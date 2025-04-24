import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, Wav2Vec2FeatureExtractor
from models.audio_text_model import ATmodel
from custom_datasets import IEMOCAPDataset


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
    parser.add_argument("--num_classes", type=int, default=4, help="number of classes")
    parser.add_argument(
        "--precomputed", action="store_true", help="use precomputed features"
    )
    opts = parser.parse_args()
    torch.manual_seed(opts.seed)
    opts.device = torch.device(opts.gpu_id)
    return opts


# def collate_fn_raw(batch, text_tokenizer, audio_processor, sampling_rate=16000):
#     texts = [item["text"] for item in batch]
#     audios = [item["audio_array"] for item in batch]
#     labels = [item["label"] for item in batch]

#     text_inputs = text_tokenizer(
#         texts, padding=True, truncation=True, return_tensors="pt"
#     )
#     audio_inputs = audio_processor(
#         audios, sampling_rate=sampling_rate, padding=True, return_tensors="pt"
#     )
#     labels = torch.tensor(labels, dtype=torch.long)

#     return {"text_inputs": text_inputs, "audio_inputs": audio_inputs, "labels": labels}


def collate_fn(batch):
    if "audio_emb" in batch[0]:
        audio = torch.stack([item["audio_emb"] for item in batch])
        text = torch.stack([item["text_emb"] for item in batch])
        labels = torch.stack([item["label"] for item in batch])
        return audio, text, labels, None
    else:
        processor = Wav2Vec2FeatureExtractor.from_pretrained(
            "facebook/hubert-base-ls960"
        )
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

        audio = [item["audio_array"] for item in batch]
        text = [item["text"] for item in batch]
        labels = torch.tensor([item["label"] for item in batch])

        audio_inputs = processor(
            audio, return_tensors="pt", padding=True, sampling_rate=16000
        )
        text_inputs = tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )

        return (
            audio_inputs.input_values,
            text_inputs.input_ids,
            labels,
            text_inputs.attention_mask,
        )


def train_one_epoch(loader, model, optimizer, loss_fn, device, precomputed):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0

    for a, t, l, mask in loader:
        a, t, l = a.to(device), t.to(device), l.to(device)
        mask = mask.to(device) if mask is not None else None

        optimizer.zero_grad()
        logits = model(a, t, mask) if not precomputed else model(a, t, None)
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
        for a, t, l, mask in loader:
            a, t, l = a.to(device), t.to(device), l.to(device)
            mask = mask.to(device) if mask is not None else None
            logits = model(a, t, mask) if not precomputed else model(a, t, None)
            loss = loss_fn(logits, l)
            total_loss += loss.item()
            total_correct += (logits.argmax(dim=1) == l).sum().item()
            total_samples += l.size(0)

    return total_loss / len(loader), (total_correct / total_samples) * 100


def train_test(args):
    from datasets import load_from_disk

    raw_dataset = load_from_disk("iemocap")

    train_ds = IEMOCAPDataset(raw_dataset["train"], args.precomputed)
    # test_ds = IEMOCAPDataset(raw_dataset["test"], args.precomputed)

    # Split the dataset into train and test sets
    test_size = int(0.2 * len(train_ds))
    train_size = len(train_ds) - test_size
    train_ds, test_ds = torch.utils.data.random_split(
        train_ds, [train_size, test_size], generator=torch.Generator().manual_seed(42)
    )
    print(f"Train size: {len(train_ds)}, Test size: {len(test_ds)}")

    trainloader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=4,
    )
    testloader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=4,
    )

    model = ATmodel(
        num_classes=args.num_classes, num_latents=args.num_latent, dim=args.adapter_dim
    )
    model.to(args.device)
    print(
        "\t Model Loaded | Trainable Params:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0
    for epoch in range(args.num_epochs):
        train_loss, train_acc = train_one_epoch(
            trainloader, model, optimizer, loss_fn, args.device, args.precomputed
        )
        val_loss, val_acc = val_one_epoch(
            testloader, model, loss_fn, args.device, args.precomputed
        )

        print(
            f"Epoch {epoch + 1}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.2f}%, Val Loss {val_loss:.4f}, Val Acc {val_acc:.2f}%"
        )
        best_acc = max(best_acc, val_acc)

    print("\nBest Validation Accuracy:", round(best_acc, 2), "%")


if __name__ == "__main__":
    opts = parse_options()
    train_test(opts)
