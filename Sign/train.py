import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from models.sign_language_model import SignLanguageTranslator
from models.dataset import SignLanguageDataset
import os

def train(batch_size=None, epochs=100, learning_rate=1e-4, checkpoint_path=None, 
          video_dir=None, annotations_dir=None):
    """
    Train the sign language translation model.
    
    Args:
        batch_size (int): Batch size for training (auto-selected based on device)
        epochs (int): Number of epochs to train
        learning_rate (float): Learning rate
        checkpoint_path (str): Path to checkpoint to resume training from
        video_dir (str): Directory containing video data
        annotations_dir (str): Directory containing annotation files
    """
    if video_dir is None or annotations_dir is None:
        video_dir = 'data/data/videos'
        annotations_dir = 'data/data/annotations'
        print(f"Using default paths:\nVideos: {video_dir}\nAnnotations: {annotations_dir}")
    
    # Set up device and memory settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    is_gpu = device.type == 'cuda'
    
    # Auto-configure settings based on device
    if batch_size is None:
        batch_size = 4 if is_gpu else 1
    max_frames = 30 if is_gpu else 20
    num_workers = 2 if is_gpu else 0
    
    if is_gpu:
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
    
    print(f"Using device: {device}")
    print(f"Batch size: {batch_size}, Max frames: {max_frames}")
    
    # Initialize dataset
    train_dataset = SignLanguageDataset(
        video_dir=video_dir,
        annotations_dir=annotations_dir,
        split='train',
        max_frames=max_frames
    )
    # Get vocabulary sizes
    gloss_vocab_size, text_vocab_size = train_dataset.get_vocab_sizes()
    
    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=is_gpu,
        persistent_workers=is_gpu,
        prefetch_factor=2 if is_gpu else None
    )
    
    # Initialize model
    model = SignLanguageTranslator(
        gloss_vocab_size=gloss_vocab_size,
        text_vocab_size=text_vocab_size
    ).to(device)
    
    if is_gpu and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    # Load checkpoint if provided
    start_epoch = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from checkpoint: {checkpoint_path}")
    
    # Initialize optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = CrossEntropyLoss()
    
    print(f"Starting training with {len(train_dataset)} samples")
    
    # Training loop
    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, epochs, is_gpu
        )
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Training Loss: {train_loss:.4f}')
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f'checkpoints/checkpoint_epoch_{epoch+1}.pt'
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

def train_epoch(model, train_loader, optimizer, criterion, device, epoch, total_epochs, is_gpu):
    total_loss = 0
    batch_count = len(train_loader)
    
    for batch_idx, batch in enumerate(train_loader):
        try:
            # Move data to device
            frames = batch['frames'].to(device)
            gloss = batch['gloss'].to(device)
            text = batch['text'].to(device)
            
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward pass
            gloss_output, text_output = model(
                frames,
                gloss_targets=gloss,
                text_targets=text
            )
            
            # Calculate loss
            gloss_loss = criterion(gloss_output.view(-1, gloss_output.size(-1)), 
                                 gloss.view(-1))
            text_loss = criterion(text_output.view(-1, text_output.size(-1)), 
                                text.view(-1))
            loss = gloss_loss + text_loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Store loss value before cleanup
            current_loss = loss.item()
            total_loss += current_loss
            
            # Free up memory
            del frames, gloss, text, gloss_output, text_output, loss
            if is_gpu:
                torch.cuda.empty_cache()
            
            # Print progress using stored loss value
            if batch_idx % max(1, batch_count // 20) == 0:  # Show ~20 updates per epoch
                print(f'Epoch {epoch+1}/{total_epochs}, '
                      f'Batch {batch_idx}/{batch_count}, '
                      f'Loss: {current_loss:.4f}')
                
        except RuntimeError as e:
            if "out of memory" in str(e) and is_gpu:
                torch.cuda.empty_cache()
                print(f"WARNING: GPU OOM in batch {batch_idx}. Skipping batch...")
                continue
            else:
                raise e
    
    return total_loss / batch_count

if __name__ == '__main__':
    train() 