import torch
from torch.optim import AdamW
from transformers import (
    RobertaModel,
    RobertaTokenizer,
    HubertModel,
    Wav2Vec2FeatureExtractor,
)
# Assuming these are in your project structure
from models import XNormModel, EarlyFusionModel, LateFusionModel
from models_other.audio_text_model import ATmodel
from tqdm import tqdm, trange
import argparse
from decoder import ProjectionLayer, MultimodalDecoder
from utils import (
    get_iemocap_caption_data_loaders,
    collate_fn_caption,
    collate_fn_caption_precomputed,
)
from trainer import CaptioningTrainer # Assuming this is in your project structure
import json
import time
import csv
import os
from datetime import datetime

# Global device definition (already present)
# device = "cuda" if torch.cuda.is_available() else "cpu" # Defined in main
# print(f"Decoder training on {device}") # Moved to main for clarity

text_checkpoint = "roberta-base"
audio_checkpoint = "facebook/hubert-base-ls960"
# caption_checkpoint = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" # Not directly used in this script for model loading

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
        encoder = EarlyFusionModel(num_classes=num_classes, from_pretrained=from_pretrained) # Assuming num_classes is needed
    elif model_choice == "late":
        encoder = LateFusionModel(num_classes=num_classes, from_pretrained=from_pretrained) # Assuming num_classes is needed
    elif model_choice == "mbt":
        encoder = ATmodel(
            num_classes=num_classes,
            num_latents=4, # Example, ensure this matches your ATmodel definition
            dim=8,
            from_pretrained=from_pretrained,
        )
    else:
        raise ValueError(f"Unknown encoder model choice: {model_choice}")
    return encoder

def get_model_parameters(model, trainable_only=True):
    """Calculates the number of parameters in a model."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Decoder training script started on {device}")
    emotion_labels = ["angry", "frustrated", "happy", "sad", "neutral"]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="xnorm", choices=["xnorm", "early", "late", "mbt"]
    )
    parser.add_argument(
        "--metrics_csv_file", type=str, default="decoder_training_efficiency.csv",
        help="CSV file to log efficiency metrics for training"
    )
    parser.add_argument(
        "--metrics_txt_file", type=str, default="decoder_efficiency_metrics.txt",
        help="Text file to append human-readable efficiency metrics"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument(
        "--patience", type=int, default=5, help="Patience for early stopping"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training and evaluation"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer"
    )

    args = parser.parse_args()
    encoder_choice = args.model
    metrics_csv_file = args.metrics_csv_file
    metrics_txt_file = args.metrics_txt_file
    num_epochs = args.epochs
    patience = args.patience
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    # --- Overall Timing and Setup ---
    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats(device) # Reset before any model loading or training
        initial_gpu_mem_allocated = torch.cuda.memory_allocated(device) / (1024**2)
        initial_gpu_mem_reserved = torch.cuda.memory_reserved(device) / (1024**2)
    else:
        initial_gpu_mem_allocated = 0
        initial_gpu_mem_reserved = 0

    # --- Model Loading Time ---
    model_load_start_time = time.time()

    # Encoder
    ckpt_path = f"./temp_results/{encoder_choice}_checkpoint.pth"
    encoder = load_encoder(
        encoder_choice, len(emotion_labels), from_pretrained=ckpt_path
    )
    is_mbt = encoder_choice == "mbt" # This seems specific to how features are extracted later

    # Projector
    if encoder_choice == "xnorm":
        projector_input_dim = 1536
    elif encoder_choice == "mbt":
        projector_input_dim = 768 # Based on ATmodel's dim if it outputs that size
    else:  # early, late
        projector_input_dim = 512
    projector = ProjectionLayer(projector_input_dim, 2048) # Assuming 2048 is decoder's expected dim

    # Decoder
    decoder = MultimodalDecoder()
    caption_tokenizer = decoder.tokenizer

    model_load_end_time = time.time()
    model_loading_time_seconds = model_load_end_time - model_load_start_time

    # --- Get Model Parameters ---
    # Encoder parameters (total, as it's frozen)
    encoder_total_params = get_model_parameters(encoder, trainable_only=False)
    encoder_trainable_params = get_model_parameters(encoder, trainable_only=True) # Should be 0

    # Projector and Decoder parameters (these are trained)
    projector_trainable_params = get_model_parameters(projector)
    decoder_trainable_params = get_model_parameters(decoder)
    total_trained_params = projector_trainable_params + decoder_trainable_params

    # --- Data Loading ---
    # Note: Data loading time is not included in "training time" here
    if encoder_choice in ("xnorm", "mbt"):
        text_tokenizer = RobertaTokenizer.from_pretrained(text_checkpoint)
        audio_processor = Wav2Vec2FeatureExtractor.from_pretrained(audio_checkpoint)
        train_loader, val_loader, test_loader = get_iemocap_caption_data_loaders(
            ds_path="./iemocap",
            precomputed=False,
            collate_fn=lambda batch_data: collate_fn_caption(
                batch_data,
                text_tokenizer=text_tokenizer,
                audio_processor=audio_processor,
                caption_tokenizer=caption_tokenizer,
            ),
            batch_size=batch_size,
        )
    else: # early, late
        train_loader, val_loader, test_loader = get_iemocap_caption_data_loaders(
            ds_path="./iemocap_precomputed",
            precomputed=True,
            collate_fn=lambda batch_data: collate_fn_caption_precomputed(
                batch_data, caption_tokenizer=caption_tokenizer
            ),
            batch_size=batch_size,
        )

    # --- Move models to device ---
    encoder.to(device)
    projector.to(device)
    decoder.to(device)

    # --- Optimizer and Trainer ---
    # Parameters to be optimized are from projector and decoder
    params_to_optimize = list(projector.parameters()) + list(decoder.parameters())
    optimizer = AdamW(params_to_optimize, lr=learning_rate)

    trainer = CaptioningTrainer(
        encoder=encoder,
        projector=projector,
        decoder=decoder,
        decoder_tokenizer=caption_tokenizer,
        optimizer=optimizer,
        device=device,
        is_mbt=is_mbt, # Pass this to the trainer if it needs it
    )

    # --- Training Loop ---
    best_val_loss = float("inf")
    counter = 0 # For early stopping
    epochs_completed = 0

    best_model_path = f"./temp_results/{encoder_choice}_best_captioning_model.pth"
    training_progress_json_file = f"./temp_results/{encoder_choice}_training_progress.json"
    training_progress_data = []

    print(f"\nStarting training for {num_epochs} epochs...")
    if device == "cuda":
        torch.cuda.synchronize() # Ensure all previous CUDA ops are done
    training_start_time = time.time()

    epoch_bar = trange(1, num_epochs + 1, desc="Training Epochs")
    for epoch in epoch_bar:
        epochs_completed += 1
        epoch_bar.set_description(f"Epoch {epoch}/{num_epochs}")

        train_loss = trainer.train_one_epoch(train_loader, epoch)
        val_loss = trainer.evaluate(val_loader)

        training_progress_data.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )
        epoch_bar.set_postfix(
            {
                "Train Loss": f"{train_loss:.4f}",
                "Val Loss": f"{val_loss:.4f}",
                "Best Val": f"{best_val_loss:.4f}",
            }
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(
                {
                    "projector_state_dict": projector.state_dict(),
                    "decoder_state_dict": decoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                },
                best_model_path,
            )
            epoch_bar.write(f"Best model saved at epoch {epoch} (Val Loss: {best_val_loss:.4f})")
        else:
            counter += 1
            epoch_bar.write(f"No improvement for {counter} epoch(s). Best Val Loss: {best_val_loss:.4f}")

        if counter >= patience:
            epoch_bar.write(
                f"Early stopping triggered after {patience} epochs without improvement."
            )
            break
    
    if device == "cuda":
        torch.cuda.synchronize() # Ensure all training CUDA ops are done
    training_end_time = time.time()
    total_training_time_seconds = training_end_time - training_start_time

    # --- Save Training Progress (Losses) ---
    with open(training_progress_json_file, "w") as f:
        json.dump(training_progress_data, f, indent=2)
    print(f"Training progress (losses) saved to '{training_progress_json_file}'")

    # --- Calculate Efficiency Metrics ---
    avg_time_per_epoch_seconds = (
        total_training_time_seconds / epochs_completed
        if epochs_completed > 0
        else 0
    )

    if device == "cuda":
        peak_gpu_memory_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        peak_gpu_memory_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024**2)
        # Subtract initial to see memory specifically used by models and training
        peak_gpu_mem_usage_training_allocated_mb = peak_gpu_memory_allocated_mb - initial_gpu_mem_allocated
        peak_gpu_mem_usage_training_reserved_mb = peak_gpu_memory_reserved_mb - initial_gpu_mem_reserved
    else:
        peak_gpu_memory_allocated_mb = 0
        peak_gpu_memory_reserved_mb = 0
        peak_gpu_mem_usage_training_allocated_mb = 0
        peak_gpu_mem_usage_training_reserved_mb = 0
    
    # --- Load Best Model for Test Evaluation ---
    if os.path.exists(best_model_path):
        print(f"\nLoading best model from {best_model_path} for final testing...")
        checkpoint = torch.load(best_model_path, map_location=device) # Load directly to target device
        
        # Create new instances or load into existing ones carefully
        # For simplicity, assuming projector and decoder are already on the correct device
        # If they were moved to CPU before, ensure they are back on 'device'
        projector.load_state_dict(checkpoint["projector_state_dict"])
        decoder.load_state_dict(checkpoint["decoder_state_dict"])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict']) # Usually not needed for eval

        projector.to(device) # Ensure they are on the correct device
        decoder.to(device)
        encoder.to(device) # Encoder should also be on the device for the trainer

        if 'best_val_loss' in checkpoint:
             print(f"Best model loaded (Epoch: {checkpoint.get('epoch', 'N/A')}, Validation Loss: {checkpoint['best_val_loss']:.4f})")
        else:
            print(f"Best model loaded (Validation Loss from training: {best_val_loss:.4f})")

        # Re-initialize trainer with potentially loaded models if necessary, or ensure current trainer uses them
        # The current `trainer` instance should still hold references to the updated projector and decoder
        trainer.encoder.eval() # Ensure all models are in eval mode for testing
        trainer.projector.eval()
        trainer.decoder.eval()

        test_loss = trainer.evaluate(test_loader)
        print(f"Final Test Loss with best model: {test_loss:.4f}")
    else:
        print(f"No best model found at {best_model_path}. Skipping test evaluation.")
        test_loss = float('nan') # Or some other indicator

    # --- Log Efficiency Metrics ---
    metrics_summary = {
        "timestamp": run_timestamp,
        "model_type": encoder_choice,
        "device": device,
        "num_epochs_run": epochs_completed,
        "configured_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "model_loading_time_seconds": round(model_loading_time_seconds, 2),
        "total_training_time_seconds": round(total_training_time_seconds, 2),
        "avg_time_per_epoch_seconds": round(avg_time_per_epoch_seconds, 2),
        "peak_gpu_memory_allocated_mb (total)": round(peak_gpu_memory_allocated_mb, 2),
        "peak_gpu_memory_reserved_mb (total)": round(peak_gpu_memory_reserved_mb, 2),
        "peak_gpu_memory_allocated_mb (training_specific)": round(peak_gpu_mem_usage_training_allocated_mb, 2),
        "peak_gpu_memory_reserved_mb (training_specific)": round(peak_gpu_mem_usage_training_reserved_mb, 2),
        "encoder_total_parameters": encoder_total_params,
        "encoder_trainable_parameters": encoder_trainable_params, # Should be 0
        "projector_trainable_parameters": projector_trainable_params,
        "decoder_trainable_parameters": decoder_trainable_params,
        "total_trained_parameters": total_trained_params,
        "final_best_validation_loss": round(best_val_loss, 4) if best_val_loss != float('inf') else 'N/A',
        "final_test_loss": round(test_loss, 4) if not isinstance(test_loss, float) or not torch.isnan(torch.tensor(test_loss)) else 'N/A',

    }

    print("\n--- Training Efficiency Metrics Summary ---")
    for key, value in metrics_summary.items():
        print(f"{key}: {value}")

    # Append to CSV
    csv_file_exists = os.path.isfile(metrics_csv_file)
    try:
        with open(metrics_csv_file, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=metrics_summary.keys())
            if not csv_file_exists or os.path.getsize(metrics_csv_file) == 0:
                writer.writeheader()
            writer.writerow(metrics_summary)
        print(f"\nEfficiency metrics appended to CSV: '{metrics_csv_file}'")
    except IOError:
        print(f"Error writing to CSV file: {metrics_csv_file}")


    # Append human-readable summary to TXT file
    try:
        with open(metrics_txt_file, 'a') as txtfile:
            txtfile.write(f"--- Metrics for Run: {run_timestamp} ---\n")
            txtfile.write(f"Model Type: {encoder_choice}\n")
            txtfile.write(f"Device: {device}\n")
            for key, value in metrics_summary.items():
                if key not in ["timestamp", "model_type", "device"]: # Avoid redundancy
                    txtfile.write(f"{key.replace('_', ' ').capitalize()}: {value}\n")
            txtfile.write("\n") # Add a blank line for separation
        print(f"Human-readable efficiency metrics appended to TXT: '{metrics_txt_file}'")
    except IOError:
        print(f"Error writing to TXT file: {metrics_txt_file}")

    torch.cuda.empty_cache() # Clean up GPU memory at the very end
    print("Decoder training script finished.")


if __name__ == "__main__":
    main()
