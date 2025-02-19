import argparse
import os
import torch
import gradio as gr
from train import train
from models.sign_language_model import SignLanguageTranslator
from models.dataset import SignLanguageDataset
from test_utils import calculate_metrics, translate_video
import pandas as pd

class SignLanguageGUI:
    def __init__(self, checkpoint_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model = SignLanguageTranslator(
            gloss_vocab_size=len(checkpoint['gloss_vocab']),
            text_vocab_size=len(checkpoint['text_vocab'])
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Load test dataset annotations
        self.test_data = pd.read_excel('data/data/annotations/test.xlsx')
        
        # Store vocabularies and transform
        self.gloss_vocab = checkpoint['gloss_vocab']
        self.text_vocab = checkpoint['text_vocab']
        self.transform = SignLanguageDataset.get_transform()
    
    def process_video(self, video_path):
        """Process a video and return translations and metrics."""
        try:
            # Get ground truth from test data
            video_name = os.path.basename(video_path)
            ground_truth = self.test_data[self.test_data['name'] == video_name.replace('.mp4', '')]['text'].iloc[0]
            
            # Translate video
            gloss_pred, text_pred = translate_video(
                self.model, video_path, self.transform, self.device
            )
            
            # Calculate metrics
            metrics = calculate_metrics(
                [text_pred],
                [[ground_truth.split()]]
            )
            
            # Format results
            results = {
                "Gloss Translation": gloss_pred,
                "Text Translation": text_pred,
                "Ground Truth": ground_truth,
                "Metrics": {
                    "BLEU-1": f"{metrics['BLEU-1']:.2f}%",
                    "BLEU-2": f"{metrics['BLEU-2']:.2f}%",
                    "BLEU-3": f"{metrics['BLEU-3']:.2f}%",
                    "BLEU-4": f"{metrics['BLEU-4']:.2f}%",
                    "CHRF": f"{metrics['CHRF']:.2f}%",
                    "ROUGE-1": f"{metrics['ROUGE-1']:.2f}%",
                    "ROUGE-2": f"{metrics['ROUGE-2']:.2f}%",
                    "ROUGE-L": f"{metrics['ROUGE-L']:.2f}%"
                }
            }
            
            # Add warning if BLEU-4 is below threshold
            if metrics['BLEU-4'] < 28:
                results["Warning"] = "⚠️ BLEU-4 score is below the target threshold of 28%"
            
            return results
            
        except Exception as e:
            return {
                "Error": f"Failed to process video: {str(e)}"
            }
    
    def create_interface(self):
        """Create Gradio interface."""
        with gr.Blocks(title="Sign Language Translation Testing") as interface:
            gr.Markdown("# Sign Language Translation Testing Interface")
            gr.Markdown("### Test with videos from the test dataset")
            
            with gr.Row():
                with gr.Column():
                    video_input = gr.Video(label="Upload Test Video")
                    process_btn = gr.Button("Translate and Evaluate")
                
                with gr.Column():
                    with gr.Group():
                        gr.Markdown("### Translation Results")
                        gloss_output = gr.Textbox(label="Gloss Translation")
                        text_output = gr.Textbox(label="Text Translation")
                        ground_truth = gr.Textbox(label="Ground Truth")
            
            with gr.Row():
                with gr.Group():
                    gr.Markdown("### Evaluation Metrics")
                    metrics_output = gr.JSON(label="Metrics")
                    warning_output = gr.Markdown()
            
            # Example videos from test set
            gr.Markdown("### Example Test Videos:")
            example_videos = [
                os.path.join('data/data/videos/test', video)
                for video in [
                    '01April_2010_Thursday_heute-6705.mp4',
                    '01April_2011_Friday_tagesschau-3377.mp4',
                    '01June_2010_Tuesday_tagesschau-5009.mp4'
                ]
            ]
            gr.Examples(
                examples=example_videos,
                inputs=video_input
            )
            
            def handle_translation(video):
                results = self.process_video(video)
                warning = results.get("Warning", "")
                return (
                    results.get("Gloss Translation", ""),
                    results.get("Text Translation", ""),
                    results.get("Ground Truth", ""),
                    results.get("Metrics", {}),
                    warning
                )
            
            process_btn.click(
                fn=handle_translation,
                inputs=[video_input],
                outputs=[
                    gloss_output,
                    text_output,
                    ground_truth,
                    metrics_output,
                    warning_output
                ]
            )
        
        return interface

def main():
    parser = argparse.ArgumentParser(description='Sign Language Translation')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'test', 'gui'],
                       help='Mode to run the model in')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint to load')
    
    args = parser.parse_args()

    # Data paths
    data_config = {
        'video_dir': 'data/data/videos',
        'annotations_dir': 'data/data/annotations',
    }

    if args.mode == 'train':
        train(
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            checkpoint_path=args.checkpoint,
            **data_config
        )
    elif args.mode == 'gui':
        if not args.checkpoint:
            raise ValueError("Checkpoint path must be provided for GUI mode")
        
        # Create and launch GUI
        gui = SignLanguageGUI(args.checkpoint)
        interface = gui.create_interface()
        interface.launch(share=True)

if __name__ == '__main__':
    main() 