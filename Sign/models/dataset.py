import torch
from torch.utils.data import Dataset
import cv2
import pandas as pd
import os
from PIL import Image
from torchvision import transforms

# Constants for frame dimensions
FRAME_HEIGHT = 160
FRAME_WIDTH = 160

class SignLanguageDataset(Dataset):
    def __init__(self, video_dir, annotations_dir, split='train', max_frames=30):
        """
        Args:
            video_dir (str): Base directory containing train/dev/test video folders
            annotations_dir (str): Directory containing all annotation Excel files
            split (str): One of 'train', 'dev', or 'test'
            max_frames (int): Maximum number of frames to use per video
        """
        self.base_video_dir = video_dir
        self.video_dir = os.path.join(video_dir, split)
        self.max_frames = max_frames
        self.split = split
        
        print(f"Initializing {split} dataset...")
        print(f"Video directory: {self.video_dir}")
        
        if not os.path.exists(self.video_dir):
            raise ValueError(f"Video directory not found: {self.video_dir}")
        
        # Load only the specific Excel file for this split
        excel_file = f"{split}.xlsx"
        excel_path = os.path.join(annotations_dir, excel_file)
        
        if not os.path.exists(excel_path):
            raise ValueError(f"Annotation file not found: {excel_path}")
            
        print(f"Loading annotations from: {excel_file}")
        self.annotations = pd.read_excel(excel_path)
        print(f"Loaded {len(self.annotations)} entries from {excel_file}")
        print(f"Columns in {excel_file}: {list(self.annotations.columns)}")
        
        # Rename columns to match expected names
        self.annotations = self.annotations.rename(columns={
            'name': 'video_name',  # Your Excel has 'name' for video filenames
            'gloss': 'gloss',      # Keep as is
            'text': 'text'         # Keep as is
        })
        
        # Clean up video names: remove 'train/' prefix and ensure .mp4 extension
        self.annotations['video_name'] = self.annotations['video_name'].apply(
            lambda x: x.replace('train/', '').replace('dev/', '').replace('test/', '') 
            if isinstance(x, str) else x
        )
        self.annotations['video_name'] = self.annotations['video_name'].apply(
            lambda x: f"{x}.mp4" if not x.endswith('.mp4') else x
        )
        
        # Debug: Print some video names from annotations
        print("\nFirst few video names in annotations (after cleanup):")
        print(self.annotations['video_name'].head())
        
        # Debug: Print some video files from directory
        print("\nFirst few video files in directory:")
        video_files = sorted(os.listdir(self.video_dir))[:5]
        print(video_files if video_files else "No video files found")
        
        # Filter annotations based on split
        self.annotations = self._filter_annotations_for_split()
        
        if len(self.annotations) == 0:
            raise ValueError(
                f"No matching videos found in {self.video_dir} for annotations in {excel_file}. "
                "Please check if video filenames match the 'name' column in the Excel file."
            )
        
        print(f"\nSuccessfully loaded {len(self.annotations)} video-annotation pairs")
        
        # Create vocabularies
        self._create_vocabularies()
        
        # Define image transformations with consistent size
        self.transform = transforms.Compose([
            transforms.Resize((FRAME_HEIGHT, FRAME_WIDTH)),  # Using constants from encoders
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _filter_annotations_for_split(self):
        """Filter annotations to only include videos from the current split."""
        # Get list of video files in the split directory
        available_videos = set(os.listdir(self.video_dir))
        
        # Debug: Print video name formats
        print("\nChecking video name formats:")
        print("Example annotation video name:", self.annotations['video_name'].iloc[0] if len(self.annotations) > 0 else "No annotations")
        print("Example directory video name:", next(iter(available_videos)) if available_videos else "No videos in directory")
        
        # Filter annotations to only include videos that exist in the split directory
        filtered_df = self.annotations[self.annotations['video_name'].isin(available_videos)]
        
        print(f"\nFound {len(filtered_df)} annotations for {self.split} split")
        if len(filtered_df) == 0:
            print("WARNING: No matching videos found. This might be due to filename mismatches.")
            print("\nAvailable video files:", sorted(list(available_videos))[:5], "...")
            print("\nExpected video names:", list(self.annotations['video_name'].head()))
        
        return filtered_df
    
    def _create_vocabularies(self):
        """Create vocabularies and word-to-index mappings."""
        # Gloss vocabulary
        self.gloss_vocab = ['<pad>', '<sos>', '<eos>'] + sorted(list(set(
            word for gloss in self.annotations['gloss'] 
            for word in str(gloss).split()
        )))
        self.gloss_to_idx = {word: idx for idx, word in enumerate(self.gloss_vocab)}
        
        # Text vocabulary
        self.text_vocab = ['<pad>', '<sos>', '<eos>'] + sorted(list(set(
            word for text in self.annotations['text'] 
            for word in str(text).split()
        )))
        self.text_to_idx = {word: idx for idx, word in enumerate(self.text_vocab)}

    def _convert_to_indices(self, text, vocab_dict, max_len=50):
        """Convert text to indices with padding."""
        words = str(text).split()
        indices = [vocab_dict['<sos>']]  # Start token
        indices.extend(vocab_dict.get(word, vocab_dict['<pad>']) for word in words[:max_len-2])
        indices.append(vocab_dict['<eos>'])  # End token
        
        # Pad sequence
        while len(indices) < max_len:
            indices.append(vocab_dict['<pad>'])
            
        return torch.tensor(indices, dtype=torch.long)

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        # Get video path and annotations
        video_name = self.annotations.iloc[idx]['video_name']
        gloss = self.annotations.iloc[idx]['gloss']
        text = self.annotations.iloc[idx]['text']
        
        # Load video frames
        video_path = os.path.join(self.video_dir, video_name)
        frames = self._load_video(video_path)
        
        # Convert gloss and text to indices
        gloss_indices = self._convert_to_indices(gloss, self.gloss_to_idx)
        text_indices = self._convert_to_indices(text, self.text_to_idx)
        
        return {
            'frames': frames,
            'gloss': gloss_indices,  # Now returns tensor
            'text': text_indices     # Now returns tensor
        }
    
    def _load_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        try:
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            sample_rate = max(1, total_frames // self.max_frames)
            frame_positions = range(0, total_frames, sample_rate)[:self.max_frames]
            
            for frame_pos in frame_positions:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                ret, frame = cap.read()
                
                if ret:
                    # Resize immediately to save memory
                    frame = cv2.resize(frame, (FRAME_HEIGHT, FRAME_WIDTH))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = Image.fromarray(frame)
                    frame = self.transform(frame)
                    frames.append(frame)
                
                if len(frames) >= self.max_frames:
                    break
                    
        except Exception as e:
            print(f"Error loading video {video_path}: {str(e)}")
        
        finally:
            cap.release()
        
        # Handle empty or partially loaded videos
        if not frames:
            dummy_frame = torch.zeros(3, FRAME_HEIGHT, FRAME_WIDTH)
            frames = [dummy_frame] * self.max_frames
        else:
            while len(frames) < self.max_frames:
                frames.append(torch.zeros_like(frames[0]))
        
        return torch.stack(frames)

    def get_vocab_sizes(self):
        """Return the sizes of gloss and text vocabularies."""
        return len(self.gloss_vocab), len(self.text_vocab)

    def get_gloss_vocab(self):
        """Create vocabulary from all gloss annotations."""
        if not hasattr(self, '_gloss_vocab'):
            unique_words = set()
            for gloss in self.annotations['gloss']:
                words = str(gloss).split()
                unique_words.update(words)
            self._gloss_vocab = ['<pad>', '<sos>', '<eos>'] + sorted(list(unique_words))
        return self._gloss_vocab

    def get_text_vocab(self):
        """Create vocabulary from all text annotations."""
        if not hasattr(self, '_text_vocab'):
            unique_words = set()
            for text in self.annotations['text']:
                words = str(text).split()
                unique_words.update(words)
            self._text_vocab = ['<pad>', '<sos>', '<eos>'] + sorted(list(unique_words))
        return self._text_vocab 