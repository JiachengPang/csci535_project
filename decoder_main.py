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
import json

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Decoder training on {device}")
text_checkpoint = "roberta-base"
audio_checkpoint = "facebook/hubert-base-ls960"
caption_checkpoint = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

DEFAULT_PROMPT = "Describe the emotion expressed in this speech by focusing on both the speaker's words and vocal characteristics. Your response:"


def load_encoder(model_choice, num_classes, from_pretrained=None):
    print(f"Loading encoder: {model_choice}, from_pretrained: {from_pretrained}")
    if model_choice == "xnorm":
        roberta = RobertaModel.from_pretrained(text_checkpoint)
        hubert = HubertModel.from_pretrained(audio_checkpoint)

        # freeze roberta and hubert
        for param in roberta.parameters():
            param.requires_grad = False
        for param in hubert.parameters():
            param.requires_grad = False

        encoder = XNormModel(
            roberta=roberta,
            hubert=hubert,
            num_classes=num_classes,
            from_pretrained=from_pretrained,
        )
    elif model_choice == "early":
        encoder = EarlyFusionModel(from_pretrained=from_pretrained)
    elif model_choice == "late":
        encoder = LateFusionModel(from_pretrained=from_pretrained)

    return encoder


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Decoder training on {device}")
    emotion_labels = ["angry", "frustrated", "happy", "sad", "neutral"]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="xnorm", choices=["xnorm", "early", "late"]
    )
    args = parser.parse_args()
    encoder_choice = args.model

    # encoder
    ckpt_path = f"{encoder_choice}_checkpoint.pth"
    encoder = load_encoder(
        encoder_choice, len(emotion_labels), from_pretrained=ckpt_path
    )

    if encoder_choice == "xnorm":
        projector = ProjectionLayer(1536, 2048)
    else:
        projector = ProjectionLayer(512, 2048)

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

    # optimizer and trainer
    optimizer = AdamW(
        list(projector.parameters()) + list(decoder.parameters()), lr=1e-4
    )
    trainer = CaptioningTrainer(
        encoder=encoder,
        projector=projector,
        decoder=decoder,
        decoder_tokenizer=caption_tokenizer,
        optimizer=optimizer,
        device=device,
    )

    num_epochs = 50
    patience = 5
    best_val_loss = float("inf")
    counter = 0

    best_model_path = f"./results/{encoder_choice}_best_captioning_model.pth"
    training_progress = []

    epoch_bar = trange(1, num_epochs + 1, desc="Training Epochs")

    for epoch in epoch_bar:
        epoch_bar.set_description(f"Epoch {epoch}/{num_epochs}")

        train_loss = trainer.train_one_epoch(train_loader, epoch)
        val_loss = trainer.evaluate(val_loader)

        # record progress
        training_progress.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
            }
        )

        # update progress bar
        epoch_bar.set_postfix(
            {
                "Train Loss": f"{train_loss:.4f}",
                "Val Loss": f"{val_loss:.4f}",
                "Best Val Loss": f"{best_val_loss:.4f}",
            }
        )

        # save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(
                {
                    "projector_state_dict": projector.state_dict(),
                    "decoder_state_dict": decoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                best_model_path,
            )
            epoch_bar.write(f"Best model saved at epoch {epoch}!")
        else:
            counter += 1
            epoch_bar.write(f"No improvement for {counter} epoch(s)")

        # Early stopping
        if counter >= patience:
            epoch_bar.write(
                f"Early stopping triggered after {patience} epochs without improvement."
            )
            break

    JSON_FILE = f"./results/{encoder_choice}_training_progress.json"
    with open(JSON_FILE, "w") as f:
        json.dump(training_progress, f, indent=2)

    print(f"Training progress saved to '{JSON_FILE}'")

    # === Load best model after training ===
    # checkpoint = torch.load(best_model_path)

    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    checkpoint = torch.load(best_model_path, map_location="cpu")
    projector.to("cpu")
    decoder.to("cpu")

    projector.load_state_dict(checkpoint["projector_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])

    projector.to(device)
    decoder.to(device)
    torch.cuda.empty_cache()

    print(
        f"Best model loaded from {best_model_path} (Validation Loss: {best_val_loss:.4f})"
    )

    test_loss = trainer.evaluate(test_loader)

    print(f"Final Test Loss: {test_loss:.4f}")


if __name__ == "__main__":
    main()
