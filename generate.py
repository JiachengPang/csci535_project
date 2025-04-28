import torch
from torch.optim import AdamW
from transformers import (
    RobertaModel,
    RobertaTokenizer,
    HubertModel,
    Wav2Vec2FeatureExtractor,
)
from models import XNormModel, EarlyFusionModel, LateFusionModel
from tqdm import tqdm, trange
import argparse
from decoder import ProjectionLayer, MultimodalDecoder
from utils import (
    get_iemocap_caption_data_loaders,
    collate_fn_caption,
    collate_fn_caption_precomputed,
)
from trainer import CaptioningTrainer
from decoder_main import load_encoder
import json

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Generating on {device}")
text_checkpoint = "roberta-base"
audio_checkpoint = "facebook/hubert-base-ls960"
caption_checkpoint = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

DEFAULT_PROMPT = "Describe the emotion expressed in this speech by focusing on both the speaker's words and vocal characteristics. Your response:"

def load_projector_decoder(model_choice, from_pretrained=None):
    print(f'Loading projector/decoder: model: {model_choice}, from_pretrained: {from_pretrained}')
    encoder_dim = 1536 if model_choice == 'xnorm' else 512
    
    if from_pretrained:
        pretrained = torch.load(from_pretrained, map_location='cpu')
        projector = ProjectionLayer(encoder_dim, 2048, pretrained_weights=pretrained['projector_state_dict'])
        decoder = MultimodalDecoder(pretrained_weights=pretrained['decoder_state_dict'])
    else:
        projector = ProjectionLayer(encoder_dim, 2048)
        decoder = MultimodalDecoder()
    
    return projector, decoder


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Decoder generation on {device}")
    emotion_labels = ["angry", "frustrated", "happy", "sad", "neutral"]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="xnorm", choices=["xnorm", "early", "late"]
    )
    args = parser.parse_args()
    encoder_choice = args.model

    # encoder

    encoder_ckpt = f'./results/{encoder_choice}_checkpoint.pth'
    projector_ckpt = f'./results/{encoder_choice}_projector_checkpoint.pth'
    decoder_ckpt = f'./results/{encoder_choice}_decoder_checkpoint.pth'
    encoder = load_encoder(encoder_choice, len(emotion_labels), from_pretrained=encoder_ckpt)
    projector = load_projector(encoder_choice, from_pretrained=projector_ckpt)
    decoder = load_decoder()

    decoder = MultimodalDecoder()
    caption_tokenizer = decoder.tokenizer

    # data
    if encoder_choice == "xnorm":
        text_tokenizer = RobertaTokenizer(text_checkpoint)
        audio_processor = Wav2Vec2FeatureExtractor(audio_checkpoint)

        train_loader, val_loader, test_loader = get_iemocap_caption_data_loaders(
            ds_path="./iemocap",
            precomputed=False,
            collate_fn=lambda batch: collate_fn_caption(
                batch,
                text_tokenizer=text_tokenizer,
                audio_processor=audio_processor,
                caption_tokenizer=caption_tokenizer,
            ),
            batch_size=1,
            first_n=100,
        )
    else:
        train_loader, val_loader, test_loader = get_iemocap_caption_data_loaders(
            ds_path="./iemocap_precomputed",
            precomputed=True,
            collate_fn=lambda batch: collate_fn_caption_precomputed(
                batch, caption_tokenizer=caption_tokenizer
            ),
            batch_size=1,
            first_n=100,
        )

    best_model_path = f"{encoder_choice}_best_captioning_model.pth"

    # === Load best model after training ===
    checkpoint = torch.load(best_model_path, map_location="cpu")
    projector.to("cpu")
    decoder.to("cpu")

    projector.load_state_dict(checkpoint["projector_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])

    projector.to(device)
    decoder.to(device)
    torch.cuda.empty_cache()

    print(f"Best model loaded from {best_model_path}")

    # test_loss = trainer.evaluate(test_loader)

    # print(f"Final Test Loss: {test_loss:.4f}")


if __name__ == "__main__":
    main()
