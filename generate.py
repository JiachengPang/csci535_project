import torch
from transformers import (
    RobertaTokenizer,
    Wav2Vec2FeatureExtractor,
)
from tqdm import tqdm
import argparse
import json
import time
import csv
import os
from datetime import datetime

# Assuming these are in your project structure
from decoder import ProjectionLayer, MultimodalDecoder
from decoder_main import load_encoder
from utils import (
    get_iemocap_caption_data_loaders,
    collate_fn_caption,
    collate_fn_caption_precomputed,
)

text_checkpoint = "roberta-base"
audio_checkpoint = "facebook/hubert-base-ls960"

DEFAULT_PROMPT = "Describe the emotion expressed in this speech by focusing on both the speaker's words and vocal characteristics. Your response:"

def load_projector_decoder(model_choice, from_pretrained=None):
    print(
        f"Loading projector/decoder: model: {model_choice}, from_pretrained: {from_pretrained}"
    )
    if model_choice == "xnorm":
        encoder_dim = 1536
    elif model_choice == "mbt":
        encoder_dim = 768
    else:  # early, late
        encoder_dim = 512

    if from_pretrained:
        pretrained_data = torch.load(from_pretrained, map_location="cpu")
        projector_state_dict = pretrained_data.get("projector_state_dict")
        decoder_state_dict = pretrained_data.get("decoder_state_dict")
        projector = ProjectionLayer(
            encoder_dim, 2048, pretrained_weights=projector_state_dict
        )
        decoder = MultimodalDecoder(pretrained_weights=decoder_state_dict)
    else:
        projector = ProjectionLayer(encoder_dim, 2048)
        decoder = MultimodalDecoder()
    return projector, decoder

def get_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Decoder generation on {device}")
    emotion_labels = ["angry", "frustrated", "happy", "sad", "neutral"] # Not used for metrics but kept for consistency

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="xnorm", choices=["xnorm", "early", "late", "mbt"]
    )
    parser.add_argument(
        "--metrics_log_file", type=str, default="efficiency_metrics.csv",
        help="CSV file to log efficiency metrics"
    )
    args = parser.parse_args()
    encoder_choice = args.model
    metrics_log_file = args.metrics_log_file

    # --- Overall Timing and Setup ---
    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        # Initial memory allocated (after imports and initial setup)
        initial_gpu_mem_allocated = torch.cuda.memory_allocated(device) / (1024**2)
        initial_gpu_mem_reserved = torch.cuda.memory_reserved(device) / (1024**2)
    else:
        initial_gpu_mem_allocated = 0
        initial_gpu_mem_reserved = 0


    # --- Model Loading Time ---
    model_load_start_time = time.time()

    encoder_ckpt = f"./results/{encoder_choice}_checkpoint.pth"
    decoder_ckpt = f"./results/{encoder_choice}_best_captioning_model.pth"

    encoder = load_encoder(
        encoder_choice, num_classes=len(emotion_labels), from_pretrained=encoder_ckpt
    )
    projector, decoder = load_projector_decoder(
        encoder_choice, from_pretrained=decoder_ckpt
    )
    caption_tokenizer = decoder.tokenizer # Needed for data loader setup

    model_load_end_time = time.time()
    model_loading_time_seconds = model_load_end_time - model_load_start_time

    # --- Get Model Parameters ---
    encoder_params = get_model_parameters(encoder)
    projector_params = get_model_parameters(projector)
    decoder_params = get_model_parameters(decoder)
    total_params = encoder_params + projector_params + decoder_params

    # --- Data Loading ---
    # Note: Data loading time is not included in "generation time" here,
    # but could be measured separately if needed.
    if encoder_choice in ("xnorm", "mbt"):
        text_tokenizer = RobertaTokenizer.from_pretrained(text_checkpoint)
        audio_processor = Wav2Vec2FeatureExtractor.from_pretrained(audio_checkpoint)
        _, _, test_loader = get_iemocap_caption_data_loaders(
            ds_path="./iemocap",
            precomputed=False,
            collate_fn=lambda batch: collate_fn_caption(
                batch,
                text_tokenizer=text_tokenizer,
                audio_processor=audio_processor,
                caption_tokenizer=caption_tokenizer,
            ),
            batch_size=16, # Consider making batch_size an arg
        )
    else:  # early, late
        _, _, test_loader = get_iemocap_caption_data_loaders(
            ds_path="./iemocap_precomputed",
            precomputed=True,
            collate_fn=lambda batch: collate_fn_caption_precomputed(
                batch, caption_tokenizer=caption_tokenizer
            ),
            batch_size=16, # Consider making batch_size an arg
        )

    generation_outputs_path = f"./temp_results/{encoder_choice}_test_generations.json"

    encoder.to(device)
    projector.to(device)
    decoder.to(device)

    encoder.eval()
    projector.eval()
    decoder.eval()

    generations_data = [] # Keep this to store actual generations if still needed for the JSON
    total_samples_processed = 0

    # --- Generation Time and GPU Memory for Inference ---
    if device == "cuda":
        torch.cuda.synchronize() # Wait for all kernels to complete before starting timer
    inference_start_time = time.time()

    progress_bar = tqdm(test_loader, desc=f"Generating Captions ({encoder_choice})")

    with torch.no_grad():
        for batch in progress_bar:
            batch_wav_ids = batch["ids"]
            current_batch_size = len(batch_wav_ids)
            total_samples_processed += current_batch_size

            if encoder_choice == "xnorm":
                text_inputs = {
                    "input_ids": batch["text_inputs"]["input_ids"].to(device),
                    "attention_mask": batch["text_inputs"]["attention_mask"].to(device),
                }
                audio_inputs = {
                    "input_values": batch["audio_inputs"]["input_values"].to(device),
                }
                features = encoder(
                    text_inputs, audio_inputs, return_features=True
                )
            elif encoder_choice == "mbt":
                a = batch["audio_inputs"]["input_values"].to(device)
                t = batch["text_inputs"]["input_ids"].to(device)
                m = batch["text_inputs"]["attention_mask"].to(device)
                features = encoder(a, t, return_features=True, text_mask=m)
            else:  # early, late
                text_embs = batch["text_embs"].to(device)
                audio_embs = batch["audio_embs"].to(device)
                features = encoder(
                    audio_emb=audio_embs, text_emb=text_embs, return_features=True
                )

            prefix_emb = projector(features)

            prompts = [DEFAULT_PROMPT] * prefix_emb.size(0)
            prompt_inputs = caption_tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).input_ids.to(device)

            generated_ids = decoder.generate(
                prefix_emb=prefix_emb,
                input_ids=prompt_inputs,
                max_new_tokens=50, # This affects generation speed
            )

            # Decoding text (part of generation process time)
            generated_texts = decoder.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            ground_truths_raw = batch["labels"].clone() # For saving actuals
            ground_truths_raw[ground_truths_raw == -100] = caption_tokenizer.pad_token_id
            ground_truth_texts = decoder.tokenizer.batch_decode(
                ground_truths_raw, skip_special_tokens=True
            )

            for i in range(len(generated_texts)):
                generations_data.append(
                    {
                        "wav_filename": batch_wav_ids[i],
                        "generated_caption": generated_texts[i].strip(),
                        "ground_truth": ground_truth_texts[i].strip(),
                    }
                )
    if device == "cuda":
        torch.cuda.synchronize() # Wait for all kernels to complete
    inference_end_time = time.time()
    total_generation_time_seconds = inference_end_time - inference_start_time

    # --- Calculate Efficiency Metrics ---
    avg_time_per_sample_seconds = (
        total_generation_time_seconds / total_samples_processed
        if total_samples_processed > 0
        else 0
    )
    samples_per_second = (
        total_samples_processed / total_generation_time_seconds
        if total_generation_time_seconds > 0
        else 0
    )

    if device == "cuda":
        peak_gpu_memory_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        peak_gpu_memory_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024**2)
        # Subtract initial to see memory specifically used by models and inference
        peak_gpu_mem_usage_models_inference_allocated_mb = peak_gpu_memory_allocated_mb - initial_gpu_mem_allocated
        peak_gpu_mem_usage_models_inference_reserved_mb = peak_gpu_memory_reserved_mb - initial_gpu_mem_reserved
    else:
        peak_gpu_memory_allocated_mb = 0
        peak_gpu_memory_reserved_mb = 0
        peak_gpu_mem_usage_models_inference_allocated_mb = 0
        peak_gpu_mem_usage_models_inference_reserved_mb = 0


    # --- Save Generations to JSON (as in original script) ---
    with open(generation_outputs_path, "w") as f:
        json.dump(generations_data, f, indent=2)
    print(
        f"{encoder_choice} decoder model generated captions saved to '{generation_outputs_path}'"
    )

    # --- Log Efficiency Metrics ---
    metrics_summary = {
        "timestamp": run_timestamp,
        "model_type": encoder_choice,
        "device": device,
        "total_samples_processed": total_samples_processed,
        "batch_size": test_loader.batch_size if test_loader else "N/A",
        "max_new_tokens": 50, # Log generation parameters that affect speed
        "model_loading_time_seconds": round(model_loading_time_seconds, 3),
        "total_generation_time_seconds": round(total_generation_time_seconds, 3),
        "avg_time_per_sample_seconds": round(avg_time_per_sample_seconds, 5),
        "samples_per_second (throughput)": round(samples_per_second, 2),
        "peak_gpu_memory_allocated_mb (total)": round(peak_gpu_memory_allocated_mb, 2),
        "peak_gpu_memory_reserved_mb (total)": round(peak_gpu_memory_reserved_mb, 2),
        "peak_gpu_memory_allocated_mb (models_inference)": round(peak_gpu_mem_usage_models_inference_allocated_mb, 2),
        "peak_gpu_memory_reserved_mb (models_inference)": round(peak_gpu_mem_usage_models_inference_reserved_mb, 2),
        "encoder_parameters": encoder_params,
        "projector_parameters": projector_params,
        "decoder_parameters": decoder_params,
        "total_trainable_parameters": total_params,
    }

    print("\n--- Efficiency Metrics Summary ---")
    for key, value in metrics_summary.items():
        print(f"{key}: {value}")

    # Append to CSV
    file_exists = os.path.isfile(metrics_log_file)
    with open(metrics_log_file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=metrics_summary.keys())
        if not file_exists or os.path.getsize(metrics_log_file) == 0:
            writer.writeheader() # Write header only once
        writer.writerow(metrics_summary)

    print(f"\nEfficiency metrics appended to '{metrics_log_file}'")


if __name__ == "__main__":
    main()