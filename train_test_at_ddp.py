import argparse
from collections import Counter
import os
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import RobertaTokenizer, Wav2Vec2FeatureExtractor
from models_other.audio_text_model import ATmodel
from custom_datasets import IEMOCAPDataset

from utils import MetricsLogger, EarlyStopping
from sklearn.metrics import f1_score

MODEL = "at_mbt"


def parse_options():
    parser = argparse.ArgumentParser(description="Audio‑Text AdaptFormer DDP Training")
    parser.add_argument("--lr", type=float, default=3e-4, help="initial learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="per‑GPU batch size")
    parser.add_argument("--num_epochs", type=int, default=50, help="training epochs")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--adapter_dim", type=int, default=8, help="adapter dimension")
    parser.add_argument("--num_latent", type=int, default=4, help="latent tokens")
    parser.add_argument(
        "--precomputed", action="store_true", help="use precomputed feats"
    )
    # These are filled automatically by torchrun / torch.distributed.launch
    parser.add_argument(
        "--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0))
    )
    opts = parser.parse_args()

    torch.manual_seed(opts.seed)
    torch.cuda.set_device(opts.local_rank)
    opts.device = torch.device("cuda", opts.local_rank)
    return opts


def init_distributed(opts):
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if rank == 0:
        print("Distributed init done | world size:", world_size)
    return world_size, rank


def collate_fn(batch):
    if "audio_emb" in batch[0]:  # Pre‑extracted embeddings branch
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


def reduce_tensor(t: torch.Tensor, world_size: int):
    """Reduce a tensor from all processes and return the averaged value."""
    with torch.no_grad():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= world_size
        return t


def train_one_epoch(loader, model, optimizer, loss_fn, device, precomputed, world_size):
    model.train()
    total_loss = torch.zeros(1, device=device)
    total_correct = torch.zeros(1, device=device)
    total_samples = torch.zeros(1, device=device)

    pred = []
    true = []

    for a, t, l, mask in loader:
        a, t, l = a.to(device), t.to(device), l.to(device)
        mask = mask.to(device) if mask is not None else None

        optimizer.zero_grad(set_to_none=True)
        logits = model(a, t, mask) if not precomputed else model(a, t, None)
        loss = loss_fn(logits, l)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            total_loss += loss.detach()
            total_correct += (logits.argmax(dim=1) == l).float().sum()
            total_samples += l.size(0)
        pred.extend(logits.argmax(dim=1).cpu().numpy())
        true.extend(l.cpu().numpy())

    # Gather across all GPUs
    reduce_tensor(total_loss, world_size)
    reduce_tensor(total_correct, world_size)
    reduce_tensor(total_samples, world_size)

    avg_loss = total_loss.item() / len(loader)
    avg_acc = (total_correct.item() / total_samples.item()) * 100
    pred = np.array(pred)
    true = np.array(true)
    f1 = f1_score(true, pred, average="macro")
    return avg_loss, avg_acc, f1


def val_one_epoch(loader, model, loss_fn, device, precomputed, world_size):
    model.eval()
    total_loss = torch.zeros(1, device=device)
    total_correct = torch.zeros(1, device=device)
    total_samples = torch.zeros(1, device=device)

    pred = []
    true = []

    with torch.no_grad():
        for a, t, l, mask in loader:
            a, t, l = a.to(device), t.to(device), l.to(device)
            mask = mask.to(device) if mask is not None else None
            logits = model(a, t, mask) if not precomputed else model(a, t, None)
            loss = loss_fn(logits, l)

            total_loss += loss
            total_correct += (logits.argmax(dim=1) == l).float().sum()
            total_samples += l.size(0)
            pred.extend(logits.argmax(dim=1).cpu().numpy())
            true.extend(l.cpu().numpy())

    reduce_tensor(total_loss, world_size)
    reduce_tensor(total_correct, world_size)
    reduce_tensor(total_samples, world_size)

    avg_loss = total_loss.item() / len(loader)
    avg_acc = (total_correct.item() / total_samples.item()) * 100

    pred = np.array(pred)
    true = np.array(true)
    f1 = f1_score(true, pred, average="macro")

    return avg_loss, avg_acc, f1


def main():
    opts = parse_options()
    world_size, rank = init_distributed(opts)

    from datasets import load_from_disk

    logger = MetricsLogger(save_path=f"./results/{MODEL}_training_metrics.json")

    raw_dataset = load_from_disk("iemocap")
    # full_ds = IEMOCAPDataset(raw_dataset["train"], opts.precomputed)

    def merge_excited(example):
        if example["major_emotion"] == "excited":
            example["major_emotion"] = "happy"
        return example

    merged_ds = raw_dataset.map(merge_excited)
    target_labels = ["angry", "frustrated", "happy", "sad", "neutral"]
    merged_ds = merged_ds.filter(lambda d: d["major_emotion"] in target_labels)

    label_counts = Counter(merged_ds["major_emotion"])
    print("Distribution after filtering:")
    for label, count in label_counts.items():
        print(f"{label}: {count}")

    ds = merged_ds.train_test_split(test_size=0.2, seed=42)
    test_val = ds["test"].train_test_split(test_size=0.5, seed=42)

    train_ds = IEMOCAPDataset(ds["train"], opts.precomputed)
    val_ds = IEMOCAPDataset(test_val["train"], opts.precomputed)
    test_ds = IEMOCAPDataset(test_val["test"], opts.precomputed)

    train_sampler = DistributedSampler(train_ds, shuffle=True)
    test_sampler = DistributedSampler(test_ds, shuffle=False)
    val_sampler = DistributedSampler(val_ds, shuffle=False)

    train_sampler = DistributedSampler(train_ds, shuffle=True)
    test_sampler = DistributedSampler(test_ds, shuffle=False)
    val_sampler = DistributedSampler(val_ds, shuffle=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=opts.batch_size,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=opts.batch_size,
        sampler=test_sampler,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=opts.batch_size,
        sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    emotion_labels = ["angry", "frustrated", "happy", "sad", "neutral"]

    num_classes = len(emotion_labels)

    model = ATmodel(
        num_classes=num_classes, num_latents=opts.num_latent, dim=opts.adapter_dim
    )
    model.to(opts.device)
    model = DDP(
        model,
        device_ids=[opts.local_rank],
        output_device=opts.local_rank,
        find_unused_parameters=False,
    )

    if rank == 0:
        print(
            "Model loaded | Trainable params:",
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr)
    loss_fn = nn.CrossEntropyLoss().to(opts.device)
    early_stopper = EarlyStopping(model=MODEL, patience=10, verbose=(rank == 0))

    best_acc = 0.0
    for epoch in range(opts.num_epochs):
        train_sampler.set_epoch(epoch)
        train_loss, train_acc, train_f1 = train_one_epoch(
            train_loader,
            model,
            optimizer,
            loss_fn,
            opts.device,
            opts.precomputed,
            world_size,
        )
        val_loss, val_acc, val_f1 = val_one_epoch(
            val_loader, model, loss_fn, opts.device, opts.precomputed, world_size
        )

        logger.log_train(train_loss, train_acc, train_f1)
        logger.log_val(val_loss, val_acc, val_f1)
        logger.save()

        if rank == 0:
            print(
                f"Epoch {epoch + 1}/{opts.num_epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
            )

        # Early stopping only on rank‑0 (uses reduced metrics)
        early_stopper(val_acc, model.module if isinstance(model, DDP) else model, rank)
        best_acc = max(best_acc, val_acc)
        if early_stopper.early_stop:
            if rank == 0:
                print("Early stopping triggered.")
            break

    # Load best model for final evaluation (rank‑0 saves, all ranks load)
    map_location = {"cuda:%d" % 0: "cuda:%d" % opts.local_rank}
    model.module.load_state_dict(
        torch.load("best_model.pth", map_location=map_location)
    )

    final_loss, final_acc, final_f1 = val_one_epoch(
        test_loader, model, loss_fn, opts.device, opts.precomputed, world_size
    )
    logger.log_test(final_loss, final_acc, final_f1)
    logger.save()
    if rank == 0:
        print("\nBest Validation Accuracy: {:.2f}%".format(best_acc))
        print("Final Test Accuracy (Best Model): {:.2f}%".format(final_acc))

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
