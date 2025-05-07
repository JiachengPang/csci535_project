import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from custom_datasets import IEMOCAPDataset, IEMOCAPCaptionDataset
import json
import os
from collections import Counter
import pandas as pd


def get_podcast_eval_loader(
    ds_path,
    precomputed=False,
    collate_fn=None,
    batch_size=16,
    num_workers=0,
    seed=42,
    first_n=0,
):
    ds = load_from_disk(ds_path)

    # allow smaller ds
    if first_n > 0:
        ds = ds.select(range(min(first_n, len(ds))))

    ds = IEMOCAPDataset(ds, precomputed=precomputed)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return loader


def get_iemocap_caption_data_loaders(
    ds_path,
    caption_path="gpt4o_audio_responses.csv",
    precomputed=False,
    collate_fn=None,
    batch_size=16,
    num_workers=0,
    seed=42,
    first_n=0,
):
    ds = load_from_disk(ds_path)["train"]
    caption_df = pd.read_csv(caption_path)
    caption_mapping = dict(zip(caption_df["id"], caption_df["response"]))

    def merge_excited(example):
        if example["major_emotion"] == "excited":
            example["major_emotion"] = "happy"
        return example

    ds = ds.map(merge_excited)

    target_labels = ["angry", "frustrated", "happy", "sad", "neutral"]
    ds = ds.filter(lambda d: d["major_emotion"] in target_labels)

    # allow smaller ds
    if first_n > 0:
        ds = ds.select(range(min(first_n, len(ds))))

    label_counts = Counter(ds["major_emotion"])
    print("Distribution after filtering:")
    for label, count in label_counts.items():
        print(f"{label}: {count}")

    ds = ds.train_test_split(test_size=0.2, seed=seed)
    test_val = ds["test"].train_test_split(test_size=0.5, seed=seed)

    train_ds = IEMOCAPCaptionDataset(
        ds["train"], caption_mapping, precomputed=precomputed
    )
    val_ds = IEMOCAPCaptionDataset(
        test_val["train"], caption_mapping, precomputed=precomputed
    )
    test_ds = IEMOCAPCaptionDataset(
        test_val["test"], caption_mapping, precomputed=precomputed
    )

    print(f"train, val, test sizes: {len(train_ds)}, {len(val_ds)}, {len(test_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader


def get_iemocap_data_loaders(
    path, precomputed, batch_size=16, num_workers=0, seed=42, collate_fn=None, first_n=0
):
    ds = load_from_disk(path)["train"]

    def merge_excited(example):
        if example["major_emotion"] == "excited":
            example["major_emotion"] = "happy"
        return example

    ds = ds.map(merge_excited)

    target_labels = ["angry", "frustrated", "happy", "sad", "neutral"]
    ds = ds.filter(lambda d: d["major_emotion"] in target_labels)

    # allow smaller ds
    if first_n > 0:
        ds = ds.select(range(min(first_n, len(ds))))

    label_counts = Counter(ds["major_emotion"])
    print("Distribution after filtering:")
    for label, count in label_counts.items():
        print(f"{label}: {count}")

    # split into 80% train, 10% val, 10% test
    ds = ds.train_test_split(test_size=0.2, seed=seed)
    test_val = ds["test"].train_test_split(test_size=0.5, seed=seed)

    train_ds = IEMOCAPDataset(ds["train"], precomputed=precomputed)
    val_ds = IEMOCAPDataset(test_val["train"], precomputed=precomputed)
    test_ds = IEMOCAPDataset(test_val["test"], precomputed=precomputed)

    print(f"train, val, test sizes: {len(train_ds)}, {len(val_ds)}, {len(test_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader


def collate_fn_caption_precomputed(batch, caption_tokenizer):
    text_embs = torch.stack([item["text_emb"] for item in batch], dim=0)  # (B, 768)
    audio_embs = torch.stack([item["audio_emb"] for item in batch], dim=0)  # (B, 768)

    captions = [item["caption"] for item in batch]
    labels = caption_tokenizer(
        captions, padding=True, truncation=True, max_length=128, return_tensors="pt"
    ).input_ids  # (B, L)

    labels[
        labels == caption_tokenizer.pad_token_id
    ] = -100  # ignore PAD tokens for loss

    return {
        "text_embs": text_embs,  # (B, 768)
        "audio_embs": audio_embs,  # (B, 768)
        "labels": labels,  # (B, L)
    }


def collate_fn_caption(
    batch, text_tokenizer, audio_processor, caption_tokenizer, sampling_rate=16000
):
    texts = [item["text"] for item in batch]
    audios = [item["audio_array"] for item in batch]
    captions = [item["caption"] for item in batch]

    text_inputs = text_tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_attention_mask=True,
    )  # 'input_ids', 'attention_mask'

    audio_inputs = audio_processor(
        audios, sampling_rate=sampling_rate, padding=True, return_tensors="pt"
    )  # 'input_values'

    labels = caption_tokenizer(
        captions, padding=True, truncation=True, max_length=128, return_tensors="pt"
    ).input_ids

    labels[labels == caption_tokenizer.pad_token_id] = -100

    return {"text_inputs": text_inputs, "audio_inputs": audio_inputs, "labels": labels}


def collate_fn_raw(batch, text_tokenizer, audio_processor, sampling_rate=16000):
    texts = [item["text"] for item in batch]
    audios = [item["audio_array"] for item in batch]
    labels = [item["label"] for item in batch]

    text_inputs = text_tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_attention_mask=True,
    )  # keys: 'input_ids', 'attention_mask'
    audio_inputs = audio_processor(
        audios, sampling_rate=sampling_rate, padding=True, return_tensors="pt"
    )  # keys: 'input_values'
    labels = torch.tensor(labels, dtype=torch.long)

    return {"text_inputs": text_inputs, "audio_inputs": audio_inputs, "labels": labels}


def collate_fn_precomputed(batch, caption_tokenizer):
    text_embs = torch.stack([item["text_emb"] for item in batch], dim=0)  # (B, 768)
    audio_embs = torch.stack([item["audio_emb"] for item in batch], dim=0)  # (B, 768)

    return {
        "text_embs": text_embs,  # (B, 768)
        "audio_embs": audio_embs,  # (B, 768)
    }


class MetricsLogger:
    def __init__(self, save_path):
        print(f"MetricsLogger init to {save_path}")
        self.save_path = save_path
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "train_f1": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1": [],
            "test_loss": [],
            "test_acc": [],
            "test_f1": [],
        }

    def log_train(self, loss, acc, f1):
        self.history["train_loss"].append(float(loss))
        self.history["train_acc"].append(float(acc))
        self.history["train_f1"].append(float(f1))

    def log_val(self, loss, acc, f1):
        self.history["val_loss"].append(float(loss))
        self.history["val_acc"].append(float(acc))
        self.history["val_f1"].append(float(f1))

    def log_test(self, loss, acc, f1):
        self.history["test_loss"].append(float(loss))
        self.history["test_acc"].append(float(acc))
        self.history["test_f1"].append(float(f1))

    def save(self):
        with open(self.save_path, "w") as f:
            json.dump(self.history, f, indent=2)

    def load(self):
        if os.path.exists(self.save_path):
            with open(self.save_path, "r") as f:
                self.history = json.load(f)


def get_iemocap_datasets_ddp(path, precomputed, seed=42, first_n=0):
    """
    Loads IEMOCAP data from disk, splits it, and returns Dataset objects
    suitable for use with DistributedDataParallel.

    Args:
        path (str): Path to the saved dataset directory (loadable by datasets.load_from_disk).
        precomputed (bool): Flag indicating if the dataset contains precomputed embeddings.
                            This is passed to the IEMOCAPDataset constructor.
        seed (int): Random seed for train/test splitting.
        first_n (int): If > 0, use only the first N samples for quick testing.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset) where each element
               is an instance of IEMOCAPDataset.
    """
    print(f"--- Loading IEMOCAP dataset from disk: {path} ---")

    try:
        # Load the full dataset (expecting 'train' split or similar)
        # Adjust split name if your saved dataset uses a different one
        ds = load_from_disk(path)["train"]
        print(f"Dataset loaded successfully. Original size: {len(ds)}")
    except Exception as e:
        print(f"ERROR loading dataset from disk: {e}")
        print("Please ensure the path is correct and the dataset was saved properly.")
        # Returning None or empty datasets to signal failure
        return None, None, None

    def merge_excited(example):
        if example["major_emotion"] == "excited":
            example["major_emotion"] = "happy"
        return example

    ds = ds.map(merge_excited)

    target_labels = ["angry", "frustrated", "happy", "sad", "neutral"]
    ds = ds.filter(lambda d: d["major_emotion"] in target_labels)

    # Allow using a smaller subset for testing
    if first_n > 0:
        print(f"Selecting first {min(first_n, len(ds))} samples.")
        ds = ds.select(range(min(first_n, len(ds))))

    label_counts = Counter(ds["major_emotion"])
    print("Distribution after filtering:")
    for label, count in label_counts.items():
        print(f"{label}: {count}")

    # Split into 80% train, 10% val, 10% test
    print(f"Splitting dataset (seed={seed})...")
    ds_split_test = ds.train_test_split(test_size=0.2, seed=seed, shuffle=True)
    # Split the 20% test set into 10% validation and 10% test
    ds_split_val_test = ds_split_test["test"].train_test_split(
        test_size=0.5, seed=seed, shuffle=True
    )

    # Get the actual data splits (these are Hugging Face Dataset objects)
    train_data = ds_split_test["train"]
    val_data = ds_split_val_test[
        "train"
    ]  # The 'train' part of the second split is validation
    test_data = ds_split_val_test["test"]  # The 'test' part of the second split is test

    print("Wrapping data splits with IEMOCAPDataset...")
    # Create IEMOCAPDataset instances
    train_ds = IEMOCAPDataset(train_data, precomputed=precomputed)
    val_ds = IEMOCAPDataset(val_data, precomputed=precomputed)
    test_ds = IEMOCAPDataset(test_data, precomputed=precomputed)

    print(
        f"Dataset instances created. Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}"
    )

    return train_ds, val_ds, test_ds


def collate_fn_raw_ddp(batch, tokenizer, processor, sampling_rate=16000):
    """
    Collates a batch of raw data points (audio_array, text, label) from
    IEMOCAPDataset for DDP training. Processes text and audio using the
    provided tokenizer and processor.

    Args:
        batch (list): A list of dictionaries from IEMOCAPDataset.__getitem__
                      (when precomputed=False). Expected keys: 'audio_array', 'text', 'label'.
        tokenizer: Initialized Hugging Face tokenizer.
        processor: Initialized Hugging Face feature extractor.
        sampling_rate (int): The target sampling rate expected by the processor.

    Returns:
        dict: A dictionary containing the processed batch:
              {'text_inputs': dict, 'audio_inputs': dict, 'labels': tensor}.
              Modalities dicts will be empty if no valid data is found.
    """
    # Extract data based on keys from IEMOCAPDataset (precomputed=False)
    texts = [item.get("text") for item in batch]
    audio_arrays = [item.get("audio_array") for item in batch]
    labels = [item.get("label", -1) for item in batch]  # Default label if missing

    # --- Process Text Inputs ---
    valid_texts = [t for t in texts if isinstance(t, str)]
    text_inputs_dict = {}
    if valid_texts:
        try:
            processed_texts = tokenizer(
                valid_texts,
                padding="longest",
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )
            text_inputs_dict = {
                "input_ids": processed_texts["input_ids"],
                "attention_mask": processed_texts["attention_mask"],
            }
        except Exception as e:
            print(f"ERROR during text tokenization: {e}")
            # Depending on desired behavior, could raise e or return empty
            text_inputs_dict = {}
    # else: text_inputs_dict remains empty if no valid texts

    # --- Process Audio Inputs ---
    # Filter out None or potentially invalid audio arrays
    valid_audio = [a for a in audio_arrays if a is not None]
    audio_inputs_dict = {}
    if valid_audio:
        try:
            processed_audio = processor(
                valid_audio,
                sampling_rate=sampling_rate,
                padding="longest",
                return_tensors="pt",
            )
            audio_inputs_dict = {
                "input_values": processed_audio["input_values"],
                "attention_mask": processed_audio.get(
                    "attention_mask", None
                ),  # Include mask if processor provides it
            }
            # Remove attention_mask if None (optional, depends on model needs)
            if audio_inputs_dict.get("attention_mask") is None:
                audio_inputs_dict.pop("attention_mask", None)
        except Exception as e:
            print(f"ERROR during audio processing: {e}")
            # Log types/shapes for debugging if errors occur
            # for i, a in enumerate(valid_audio): print(f"Audio item {i} type: {type(a)}, shape/len: {getattr(a, 'shape', len(a))}")
            # Depending on desired behavior, could raise e or return empty
            audio_inputs_dict = {}
    # else: audio_inputs_dict remains empty if no valid audio

    # --- Final Batch Structure ---
    final_batch = {
        "text_inputs": text_inputs_dict,
        "audio_inputs": audio_inputs_dict,
        "labels": torch.tensor(labels, dtype=torch.long),
    }

    return final_batch


class EarlyStopping:
    def __init__(
        self,
        name,
        model,
        patience=5,
        min_delta=0.001,
        verbose=True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = f"./results/{name}_checkpoint.pth"
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
