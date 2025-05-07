import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, Wav2Vec2FeatureExtractor
from models_other.audio_text_model import ATmodel
from custom_datasets import IEMOCAPDataset

from utils import get_iemocap_data_loaders, collate_fn_raw, MetricsLogger, EarlyStopping

from sklearn.metrics import f1_score


MODEL = "mbt"


def get_params():
    class Params:
        def __init__(self):
            self.gpu_id = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = torch.device(self.gpu_id)
            self.lr = 3e-4
            self.batch_size = 8
            self.num_epochs = 50
            self.seed = 42
            self.adapter_dim = 8
            self.num_latent = 4
            self.precomputed = False

    params = Params()
    torch.manual_seed(params.seed)
    return params


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
    pred = []
    true = []

    for batch in loader:
        a = batch["audio_inputs"].input_values
        t = batch["text_inputs"].input_ids
        l = batch["labels"]
        m = batch["text_inputs"]["attention_mask"]

        a, t, l, m = a.to(device), t.to(device), l.to(device), m.to(device)
        # mask = mask.to(device) if mask is not None else None

        optimizer.zero_grad()
        logits = model(a, t, m)
        loss = loss_fn(logits, l)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += (logits.argmax(dim=1) == l).sum().item()
        total_samples += l.size(0)
        pred.extend(logits.argmax(dim=1).cpu().numpy())
        true.extend(l.cpu().numpy())

    pred = np.array(pred)
    true = np.array(true)
    f1 = f1_score(true, pred, average="macro")

    return total_loss / len(loader), (total_correct / total_samples) * 100, f1


def val_one_epoch(loader, model, loss_fn, device, precomputed):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    pred = []
    true = []
    with torch.no_grad():
        for batch in loader:
            a = batch["audio_inputs"].input_values
            t = batch["text_inputs"].input_ids
            l = batch["labels"]
            m = batch["text_inputs"]["attention_mask"]

            a, t, l, m = a.to(device), t.to(device), l.to(device), m.to(device)
            # mask = mask.to(device) if mask is not None else None
            logits = model(a, t, m)
            loss = loss_fn(logits, l)
            total_loss += loss.item()
            total_correct += (logits.argmax(dim=1) == l).sum().item()
            total_samples += l.size(0)
            pred.extend(logits.argmax(dim=1).cpu().numpy())
            true.extend(l.cpu().numpy())

    pred = np.array(pred)
    true = np.array(true)
    f1 = f1_score(true, pred, average="macro")

    return total_loss / len(loader), (total_correct / total_samples) * 100, f1


def train_test(params):
    processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    trainloader, valloader, testloader = get_iemocap_data_loaders(
        path="./iemocap",
        precomputed=False,
        batch_size=params.batch_size,
        num_workers=0,
        collate_fn=lambda b: collate_fn_raw(b, tokenizer, processor),
    )

    emotion_labels = ["angry", "frustrated", "happy", "sad", "neutral"]

    num_classes = len(emotion_labels)

    model = ATmodel(
        num_classes=num_classes, num_latents=params.num_latent, dim=params.adapter_dim
    )

    model.to(params.device)

    print(
        "\t Model Loaded | Trainable Params:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
    loss_fn = nn.CrossEntropyLoss()
    early_stopper = EarlyStopping(name=MODEL, model=model, patience=5)

    logger = MetricsLogger(save_path=f"./results/{MODEL}_training_metrics.json")

    best_acc = 0
    for epoch in range(params.num_epochs):
        train_loss, train_acc, train_f1 = train_one_epoch(
            trainloader, model, optimizer, loss_fn, params.device, params.precomputed
        )
        val_loss, val_acc, val_f1 = val_one_epoch(
            valloader, model, loss_fn, params.device, params.precomputed
        )

        logger.log_train(train_loss, train_acc, train_f1)
        logger.log_val(val_loss, val_acc, val_f1)
        logger.save()

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
    model.load_state_dict(
        torch.load(f"./results/{MODEL}_checkpoint.pth")
    )  # ./results/{encoder_choice}_checkpoint.pth
    final_test_loss, final_test_acc, final_test_f1 = val_one_epoch(
        testloader, model, loss_fn, params.device, params.precomputed
    )

    logger.log_test(final_test_loss, final_test_acc, final_test_f1)
    logger.save()

    print(f"Final Test Accuracy (Best Model): {final_test_acc:.2f}%")


if __name__ == "__main__":
    params = get_params()
    train_test(params)
