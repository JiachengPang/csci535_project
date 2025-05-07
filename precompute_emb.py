import torch
import os
from tqdm import tqdm
import torch
from datasets import load_from_disk
from transformers import (
    HubertModel,
    Wav2Vec2FeatureExtractor,
    RobertaTokenizer,
    RobertaModel,
)

save_dir = "precomputed_embeddings"
os.makedirs(save_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

audio_checkpoint = "facebook/hubert-base-ls960"
text_checkpoint = "roberta-base"

hubert_model = HubertModel.from_pretrained(audio_checkpoint).to(device)
hubert_processor = Wav2Vec2FeatureExtractor.from_pretrained(audio_checkpoint)

roberta_model = RobertaModel.from_pretrained(text_checkpoint).to(device)
roberta_tokenizer = RobertaTokenizer.from_pretrained(text_checkpoint)


def extract_hubert_features(audio_array):
    audio_tensor = torch.tensor(audio_array).to(device)
    input_values = hubert_processor(
        audio_tensor, return_tensors="pt", sampling_rate=16000
    ).input_values.to(device)

    with torch.no_grad():
        hubert_features = hubert_model(
            input_values
        ).last_hidden_state  # (1, seq_len, 768)

    return torch.mean(hubert_features, dim=1).squeeze(0).tolist()


def extract_roberta_features(transcript):
    tokens = roberta_tokenizer(
        transcript, return_tensors="pt", padding=True, truncation=True, max_length=128
    )
    tokens = {key: val.to(device) for key, val in tokens.items()}

    with torch.no_grad():
        roberta_output = roberta_model(**tokens)

    return roberta_output.last_hidden_state[:, 0, :].squeeze(0).tolist()


dataset = load_from_disk("./podcast")

emotion_labels = ["neutral", "happy", "sad", "angry", "frustrated"]
num_classes = len(emotion_labels)

label_to_idx = {label: idx for idx, label in enumerate(emotion_labels)}

print(dataset)


print(f"HuBERT Model Device: {next(hubert_model.parameters()).device}")
print(f"RoBERTa Model Device: {next(roberta_model.parameters()).device}")
print(f"Expected Device: {device}")

print("Extracting embeddings")

modified_dataset = dataset.map(
    lambda example: {
        "audio_embedding": extract_hubert_features(example["audio"]["array"]),
        "text_embedding": extract_roberta_features(example["transcription"]),
        "label_id": label_to_idx[example["major_emotion"]],
    },
    batched=False,
    load_from_cache_file=False,
)

print("Complete.")

save_path = "./podcast_precomputed"
modified_dataset.save_to_disk(save_path)
print(f"Dataset with embeddings saved to {save_path}")

print(modified_dataset)
