import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, Wav2Vec2FeatureExtractor
from models.audio_text_model import ATmodel
from custom_datasets import IEMOCAPDataset

from utils import get_iemocap_data_loaders, collate_fn_raw


class EarlyStopping:
    def __init__(
        self,
        patience=5,
        min_delta=0.001,
        checkpoint_path="best_model.pth",
        verbose=True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_acc, model):
        if self.best_score is None or val_acc > self.best_score + self.min_delta:
            self.best_score = val_acc
            self.counter = 0
            torch.save(model.state_dict(), self.checkpoint_path)
            if self.verbose:
                print(
                    f"Validation accuracy improved. Saving model to {self.checkpoint_path}"
                )
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


def parse_options():
    parser = argparse.ArgumentParser(description="Audio-Text AdaptFormer Training")
    parser.add_argument("--gpu_id", type=str, default="cuda:0", help="the gpu id")
    parser.add_argument("--lr", type=float, default=3e-4, help="initial learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="batchsize")
    parser.add_argument(
        "--num_epochs", type=int, default=15, help="total training epochs"
    )
    parser.add_argument("--seed", type=int, default=1111, help="random seed")
    parser.add_argument(
        "--adapter_dim", type=int, default=8, help="dimension of the adapter"
    )
    parser.add_argument(
        "--num_latent", type=int, default=4, help="number of latent tokens"
    )
    parser.add_argument(
        "--precomputed", action="store_true", help="use precomputed features"
    )
    opts = parser.parse_args()
    torch.manual_seed(opts.seed)
    opts.device = torch.device(opts.gpu_id)
    return opts


# def collate_fn(batch):
#     if "audio_emb" in batch[0]:
#         audio = torch.stack([item["audio_emb"] for item in batch])
#         text = torch.stack([item["text_emb"] for item in batch])
#         labels = torch.stack([item["label"] for item in batch])
#         return audio, text, labels, None
#     else:
#         processor = Wav2Vec2FeatureExtractor.from_pretrained(
#             "facebook/hubert-base-ls960"
#         )
#         tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

#         audio = [item["audio_array"] for item in batch]
#         text = [item["text"] for item in batch]
#         labels = torch.tensor([item["label"] for item in batch])

#         audio_inputs = processor(
#             audio, return_tensors="pt", padding=True, sampling_rate=16000
#         )
#         text_inputs = tokenizer(
#             text, return_tensors="pt", padding=True, truncation=True
#         )

#         return (
#             audio_inputs.input_values,
#             text_inputs.input_ids,
#             labels,
#             text_inputs.attention_mask,
#         )


def train_one_epoch(loader, model, optimizer, loss_fn, device, precomputed):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0

    for batch in loader:
        a = batch["audio_inputs"].input_values
        t = batch["text_inputs"].input_ids
        l = batch["labels"]

        a, t, l = a.to(device), t.to(device), l.to(device)
        # mask = mask.to(device) if mask is not None else None

        optimizer.zero_grad()
        logits = model(a, t, None)
        loss = loss_fn(logits, l)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += (logits.argmax(dim=1) == l).sum().item()
        total_samples += l.size(0)

    return total_loss / len(loader), (total_correct / total_samples) * 100


def val_one_epoch(loader, model, loss_fn, device, precomputed):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    with torch.no_grad():
        for batch in loader:
            a = batch["audio_inputs"].input_values
            t = batch["text_inputs"].input_ids
            l = batch["labels"]

            a, t, l = a.to(device), t.to(device), l.to(device)
            # mask = mask.to(device) if mask is not None else None
            logits = model(a, t, None)
            loss = loss_fn(logits, l)
            total_loss += loss.item()
            total_correct += (logits.argmax(dim=1) == l).sum().item()
            total_samples += l.size(0)

    return total_loss / len(loader), (total_correct / total_samples) * 100


def train_test(args):
    processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    trainloader, valloader, testloader = get_iemocap_data_loaders(
        path="./iemocap",
        precomputed=False,
        batch_size=16,
        num_workers=0,
        collate_fn=lambda b: collate_fn_raw(b, tokenizer, processor),
    )

    emotion_labels = ["angry", "frustrated", "happy", "sad", "neutral"]

    num_classes = len(emotion_labels)

    model = ATmodel(
        num_classes=num_classes, num_latents=args.num_latent, dim=args.adapter_dim
    )

    model.to(args.device)

    print(
        "\t Model Loaded | Trainable Params:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    early_stopper = EarlyStopping(patience=5, checkpoint_path="best_model.pth")

    best_acc = 0
    for epoch in range(args.num_epochs):
        train_loss, train_acc = train_one_epoch(
            trainloader, model, optimizer, loss_fn, args.device, args.precomputed
        )
        val_loss, val_acc = val_one_epoch(
            valloader, model, loss_fn, args.device, args.precomputed
        )

        print(
            f"Epoch {epoch + 1}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.2f}%, Val Loss {val_loss:.4f}, Val Acc {val_acc:.2f}%"
        )

        early_stopper(val_acc, model)
        if early_stopper.early_stop:
            print("Early stopping triggered. Stopping training.")
            break

        best_acc = max(best_acc, val_acc)

    print("\nBest Validation Accuracy:", round(best_acc, 2), "%")

    # Load best model for evaluation
    model.load_state_dict(torch.load("best_model.pth"))
    final_test_loss, final_test_acc = val_one_epoch(
        testloader, model, loss_fn, args.device, args.precomputed
    )
    print(f"Final Test Accuracy (Best Model): {final_test_acc:.2f}%")


if __name__ == "__main__":
    opts = parse_options()
    train_test(opts)
