import torch
import argparse
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from transformers import RobertaModel, RobertaTokenizer, HubertModel, Wav2Vec2FeatureExtractor
from models import XNormModel, EarlyFusionModel, LateFusionModel
from utils import get_iemocap_data_loaders, collate_fn_raw
import os

def evaluate_and_confusion(model_choice, checkpoint_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    emotion_labels = ['angry', 'frustrated', 'happy', 'sad', 'neutral']
    num_classes = len(emotion_labels)

    # Initialize model
    if model_choice == 'xnorm':
        text_checkpoint = 'roberta-base'
        audio_checkpoint = 'facebook/hubert-base-ls960'

        roberta = RobertaModel.from_pretrained(text_checkpoint)
        hubert = HubertModel.from_pretrained(audio_checkpoint)

        model = XNormModel(
            roberta=roberta,
            hubert=hubert,
            num_classes=num_classes,
            from_pretrained=checkpoint_path
        )

        tokenizer = RobertaTokenizer.from_pretrained(text_checkpoint)
        processor = Wav2Vec2FeatureExtractor.from_pretrained(audio_checkpoint)

        _, _, test_loader = get_iemocap_data_loaders(
            path='./iemocap',
            precomputed=False,
            batch_size=2,
            num_workers=0,
            collate_fn=lambda b: collate_fn_raw(b, tokenizer, processor),
        )

    else:
        if model_choice == 'early':
            model = EarlyFusionModel(from_pretrained=checkpoint_path)
        elif model_choice == 'late':
            model = LateFusionModel(from_pretrained=checkpoint_path)
        else:
            raise ValueError(f"Unsupported model choice: {model_choice}")

        _, _, test_loader = get_iemocap_data_loaders(
            path='./iemocap_precomputed',
            precomputed=True,
            batch_size=16,
            num_workers=0,
            collate_fn=None,
        )

    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            if model_choice == 'xnorm':
                labels = batch['labels']
                batch_text = {k: v.to(device) for k, v in batch['text_inputs'].items()}
                batch_audio = {k: v.to(device) for k, v in batch['audio_inputs'].items()}
                
                logits = model(batch_text, batch_audio)
            else:
                labels = batch['label']
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                logits = model(batch['audio_emb'], batch['text_emb'])
            
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=emotion_labels)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
    plt.title(f"Confusion Matrix - {model_choice.capitalize()} Model")

    # Save to file
    os.makedirs("./results/confusion_matrices", exist_ok=True)
    save_path = f"./results/confusion_matrices/{model_choice}_confusion_matrix.png"
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")

    # Also show it
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['xnorm', 'early', 'late'])
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()

    evaluate_and_confusion(args.model, args.checkpoint)
