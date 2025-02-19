import torch
import nltk
import cv2
from PIL import Image
from nltk.translate.bleu_score import corpus_bleu
from sacrebleu import corpus_chrf, CHRF
from rouge_score import rouge_scorer
nltk.download('punkt')

def decode_sequence(output_tensor, vocab):
    """
    Convert model output tensor to text using vocabulary.
    
    Args:
        output_tensor: Model output tensor of shape (batch_size, seq_len, vocab_size)
        vocab: List of vocabulary words where index matches the model output indices
    
    Returns:
        str: Decoded text
    """
    # Get the most likely token at each position
    predictions = torch.argmax(output_tensor, dim=-1)
    
    # Convert to list for easier processing
    pred_indices = predictions[0].cpu().numpy()  # Take first sequence in batch
    
    # Convert indices to words
    words = []
    for idx in pred_indices:
        word = vocab[idx]
        if word == '<eos>':
            break
        if word not in ['<pad>', '<sos>']:
            words.append(word)
    
    return ' '.join(words)

def preprocess_video(video_path, transform):
    """
    Preprocess a video for model input.
    
    Args:
        video_path: Path to video file
        transform: Torchvision transforms to apply to frames
    
    Returns:
        torch.Tensor: Preprocessed video frames
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    max_frames = 100  # Same as in dataset class
    
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = transform(frame)
        frames.append(frame)
    
    cap.release()
    
    # Pad with zeros if video is shorter than max_frames
    while len(frames) < max_frames:
        frames.append(torch.zeros_like(frames[0]))
        
    return torch.stack(frames)

def calculate_metrics(predictions, references):
    """
    Calculate BLEU, CHRF, and ROUGE scores.
    
    Args:
        predictions: List of predicted sentences
        references: List of lists of reference sentences
    
    Returns:
        dict: Dictionary containing various metrics
    """
    # BLEU scores
    bleu1 = corpus_bleu(references, predictions, weights=(1.0, 0, 0, 0))
    bleu2 = corpus_bleu(references, predictions, weights=(0.5, 0.5, 0, 0))
    bleu3 = corpus_bleu(references, predictions, weights=(0.33, 0.33, 0.33, 0))
    bleu4 = corpus_bleu(references, predictions, weights=(0.25, 0.25, 0.25, 0.25))
    
    # CHRF score
    chrf = CHRF()
    chrf_score = chrf.corpus_score(predictions, references)
    
    # ROUGE scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(' '.join(predictions[0]), ' '.join(references[0][0]))
    
    metrics = {
        'BLEU-1': bleu1 * 100,
        'BLEU-2': bleu2 * 100,
        'BLEU-3': bleu3 * 100,
        'BLEU-4': bleu4 * 100,
        'CHRF': chrf_score.score * 100,
        'ROUGE-1': rouge_scores['rouge1'].fmeasure * 100,
        'ROUGE-2': rouge_scores['rouge2'].fmeasure * 100,
        'ROUGE-L': rouge_scores['rougeL'].fmeasure * 100
    }
    
    # Add warning if BLEU-4 is below threshold
    if metrics['BLEU-4'] < 28:
        metrics['warning'] = 'BLEU-4 score is below the target threshold of 28'
    
    return metrics

def translate_video(model, video_path, transform, device):
    """
    Translate a single video.
    
    Args:
        model: Trained SignLanguageTranslator model
        video_path: Path to video file
        transform: Torchvision transforms for preprocessing
        device: Device to run the model on
    
    Returns:
        tuple: (gloss_prediction, text_prediction)
    """
    model.eval()
    with torch.no_grad():
        # Preprocess video
        frames = preprocess_video(video_path, transform)
        frames = frames.unsqueeze(0).to(device)  # Add batch dimension
        
        # Generate translation
        gloss_output, text_output = model(frames)
        
        # Convert outputs to text
        gloss_pred = decode_sequence(gloss_output, model.gloss_vocab)
        text_pred = decode_sequence(text_output, model.text_vocab)
        
        return gloss_pred, text_pred 