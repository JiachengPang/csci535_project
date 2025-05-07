import argparse
import numpy as np
import torch
import torch.nn as nn
# from torch.utils.data import DataLoader # Not directly used, get_iemocap_data_loaders is used
from transformers import RobertaTokenizer, Wav2Vec2FeatureExtractor
from models_other.audio_text_model import ATmodel # Assuming this is your ATmodel (MBT)
# from custom_datasets import IEMOCAPDataset # Not directly used, get_iemocap_data_loaders is used

# Assuming these utils are in your project structure
from utils import get_iemocap_data_loaders, collate_fn_raw, MetricsLogger, EarlyStopping 

from sklearn.metrics import f1_score

import time # Added for timing
import csv  # Added for CSV logging
import os   # Added for file operations
from datetime import datetime # Added for timestamping
import math # For math.isnan

# MODEL_NAME = "mbt" # Use a more descriptive variable name if needed, or pass from args

def get_model_parameters(model, trainable_only=True):
    """Calculates the number of parameters in a model."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

def to_float(value):
    """Converts a value to a Python float, handling tensors and NumPy types."""
    if isinstance(value, torch.Tensor):
        return value.item()
    # Handle cases where value might already be a numpy float or int
    if hasattr(value, 'item'): # Check if it's a numpy scalar
        return value.item()
    return float(value)

def parse_options():
    parser = argparse.ArgumentParser(description="Audio-Text Model (e.g., MBT) Training")
    # Existing arguments
    parser.add_argument("--gpu_id", type=str, default="cuda:0", help="The gpu id (e.g., 'cuda:0', 'cuda', 'cpu')")
    parser.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=50, help="Total training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--adapter_dim", type=int, default=8, help="Dimension of the model (or adapter if applicable)")
    parser.add_argument("--num_latent", type=int, default=4, help="Number of latent tokens")
    parser.add_argument("--precomputed", action="store_true", help="Use precomputed features (not used in current data loading)")
    parser.add_argument("--model_name", type=str, default="mbt", help="Name of the model, used for saving files.")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping.")


    # Added arguments for paths and logging
    parser.add_argument('--results_dir', type=str, default="./results",
                        help="Directory to save checkpoints, logs, and metrics.")
    parser.add_argument('--metrics_csv_file', type=str, default="model_training_efficiency.csv", # Will be prefixed by model_name
                        help="CSV file name to log efficiency metrics.")
    parser.add_argument('--metrics_txt_file', type=str, default="model_efficiency_metrics.txt", # Will be prefixed by model_name
                        help="Text file name to append human-readable efficiency metrics.")
    parser.add_argument('--iemocap_path', type=str, default='./iemocap', help="Path to IEMOCAP dataset.")


    opts = parser.parse_args()
    torch.manual_seed(opts.seed)
    
    # Determine device
    if opts.gpu_id.lower() == "cpu":
        opts.device = torch.device("cpu")
    elif torch.cuda.is_available():
        try:
            # If just "cuda" is passed, default to cuda:0 and make the device object explicit with index
            if opts.gpu_id.lower() == "cuda":
                print("INFO: --gpu_id='cuda' provided, defaulting to device 'cuda:0'.")
                opts.device = torch.device("cuda:0") # Make the device object carry the index
            else:
                opts.device = torch.device(opts.gpu_id) # Handles 'cuda:0', 'cuda:1'
            
            # Test the device to ensure it's valid and selected
            torch.ones(1).to(opts.device) 
            print(f"Successfully set device to: {opts.device} (Index: {opts.device.index if opts.device.type == 'cuda' else 'N/A'})")
        except RuntimeError as e:
            print(f"Error setting CUDA device '{opts.gpu_id}': {e}. Defaulting to CPU.")
            opts.device = torch.device("cpu")
    else:
        print(f"CUDA not available or device '{opts.gpu_id}' not found. Defaulting to CPU.")
        opts.device = torch.device("cpu")
        
    # Ensure results directory exists
    os.makedirs(opts.results_dir, exist_ok=True)
    
    return opts


def train_one_epoch(loader, model, optimizer, loss_fn, device): 
    model.train()
    total_loss_sum, total_correct, total_samples = 0, 0, 0
    all_preds = []
    all_true = []

    for batch in loader:
        audio_input_values = batch["audio_inputs"]["input_values"].to(device)
        text_input_ids = batch["text_inputs"]["input_ids"].to(device)
        labels = batch["labels"].to(device)
        text_attention_mask = batch["text_inputs"]["attention_mask"].to(device)

        optimizer.zero_grad()
        logits = model(audio_input_values, text_input_ids, text_attention_mask) 
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss_sum += loss.item() * labels.size(0) 
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_true.extend(labels.cpu().numpy())

    avg_loss = total_loss_sum / total_samples if total_samples > 0 else 0.0
    accuracy = (total_correct / total_samples) * 100.0 if total_samples > 0 else 0.0
    f1 = f1_score(all_true, all_preds, average="macro", zero_division=0)

    return avg_loss, accuracy, f1


def val_one_epoch(loader, model, loss_fn, device): 
    model.eval()
    total_loss_sum, total_correct, total_samples = 0, 0, 0
    all_preds = []
    all_true = []
    with torch.no_grad():
        for batch in loader:
            audio_input_values = batch["audio_inputs"]["input_values"].to(device)
            text_input_ids = batch["text_inputs"]["input_ids"].to(device)
            labels = batch["labels"].to(device)
            text_attention_mask = batch["text_inputs"]["attention_mask"].to(device)

            logits = model(audio_input_values, text_input_ids, text_attention_mask)
            loss = loss_fn(logits, labels)
            
            total_loss_sum += loss.item() * labels.size(0) 
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_true.extend(labels.cpu().numpy())

    avg_loss = total_loss_sum / total_samples if total_samples > 0 else 0.0
    accuracy = (total_correct / total_samples) * 100.0 if total_samples > 0 else 0.0
    f1 = f1_score(all_true, all_preds, average="macro", zero_division=0)
    
    return avg_loss, accuracy, f1


def train_test(args):
    print(f"Starting training for model: {args.model_name} on device: {args.device}")

    # --- Efficiency Metrics Setup ---
    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cuda_device_idx_for_stats = None 
    if args.device.type == "cuda":
        cuda_device_idx_for_stats = args.device.index 
        if cuda_device_idx_for_stats is None: 
            print("WARNING: CUDA device index is None, defaulting to current_device() for stats. This might be an issue.")
            cuda_device_idx_for_stats = torch.cuda.current_device()

        print(f"DEBUG: Using CUDA device index for stats: {cuda_device_idx_for_stats} (type: {type(cuda_device_idx_for_stats)}) for device {args.device}")
        
        torch.cuda.reset_peak_memory_stats(cuda_device_idx_for_stats)
        initial_gpu_mem_allocated = torch.cuda.memory_allocated(cuda_device_idx_for_stats) / (1024**2)
        initial_gpu_mem_reserved = torch.cuda.memory_reserved(cuda_device_idx_for_stats) / (1024**2)
    else:
        initial_gpu_mem_allocated = 0.0
        initial_gpu_mem_reserved = 0.0

    # --- Model Instantiation Time ---
    model_instantiation_start_time = time.time()

    processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    emotion_labels = ["angry", "frustrated", "happy", "sad", "neutral"]
    num_classes = len(emotion_labels)

    model = ATmodel(
        num_classes=num_classes, 
        num_latents=args.num_latent, 
        dim=args.adapter_dim 
    )
    model.to(args.device) 

    model_instantiation_end_time = time.time()
    model_instantiation_time_seconds = model_instantiation_end_time - model_instantiation_start_time

    # --- Get Model Parameters ---
    model_total_params = get_model_parameters(model, trainable_only=False)
    model_trainable_params = get_model_parameters(model, trainable_only=True)

    print(f"\tModel {args.model_name} Loaded | Total Params: {model_total_params} | Trainable Params: {model_trainable_params}")

    # --- Data Loaders ---
    trainloader, valloader, testloader = get_iemocap_data_loaders(
        path=args.iemocap_path, 
        precomputed=False, 
        batch_size=args.batch_size,
        num_workers=0, 
        collate_fn=lambda b: collate_fn_raw(b, tokenizer, processor),
    )
    
    # --- Optimizer, Loss, Early Stopper, Logger ---
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    
    checkpoint_file_name = f"{args.model_name}_checkpoint.pth"
    best_model_path = os.path.join(args.results_dir, checkpoint_file_name) 

    performance_log_name = f"{args.model_name}_training_metrics.json"
    
    early_stopper = EarlyStopping(
        name=args.model_name, 
        model=model, 
        patience=args.patience
    )
    performance_logger = MetricsLogger(save_path=os.path.join(args.results_dir, performance_log_name))

    # --- Training Loop ---
    best_val_acc_tracker = 0.0 
    epochs_completed = 0

    print(f"\nStarting training loop for {args.num_epochs} epochs...")
    if args.device.type == "cuda":
        torch.cuda.synchronize(args.device) 
    training_start_time = time.time()

    for epoch in range(args.num_epochs):
        epochs_completed += 1
        train_loss, train_acc, train_f1 = train_one_epoch(
            trainloader, model, optimizer, loss_fn, args.device
        )
        val_loss, val_acc, val_f1 = val_one_epoch(
            valloader, model, loss_fn, args.device
        )

        performance_logger.log_train(train_loss, train_acc, train_f1)
        performance_logger.log_val(val_loss, val_acc, val_f1)
        performance_logger.save() 

        print(
            f"Epoch {epoch + 1}/{args.num_epochs}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.2f}%, Train F1 {train_f1:.4f} | "
            f"Val Loss {val_loss:.4f}, Val Acc {val_acc:.2f}%, Val F1 {val_f1:.4f}"
        )

        # CORRECTED: Pass the model to early_stopper's __call__ method
        early_stopper(val_acc, model) 
        if early_stopper.early_stop:
            print("Early stopping triggered. Stopping training.")
            break
        
        if val_acc > best_val_acc_tracker: 
            best_val_acc_tracker = val_acc


    if args.device.type == "cuda":
        torch.cuda.synchronize(args.device) 
    training_end_time = time.time()
    total_training_time_seconds = training_end_time - training_start_time
    
    print(f"\nTraining finished after {epochs_completed} epochs.")
    # early_stopper.best_score is the validation metric (accuracy in this case)
    # early_stopper.model should hold the best model state if your EarlyStopping class updates it
    print(f"Best Validation Accuracy achieved: {to_float(early_stopper.best_score):.2f}% (from EarlyStopper)")


    # --- Load Best Model for Final Test Evaluation ---
    # The `best_model_path` is where EarlyStopping should have saved the best model.
    # Your original script loaded from a path constructed like this:
    # model.load_state_dict(torch.load(f"./results/{MODEL}_checkpoint.pth"))
    # Ensure your EarlyStopping class saves to `best_model_path`
    final_test_loss_float, final_test_acc_float, final_test_f1_float = float('nan'), float('nan'), float('nan')

    if os.path.exists(best_model_path):
        print(f"Loading best model from: {best_model_path}")
        # It's generally safer to load the state_dict into the existing model instance
        # or re-initialize and load if EarlyStopping doesn't update the model in-place.
        # Assuming EarlyStopping class saves the model's state_dict.
        model.load_state_dict(torch.load(best_model_path, map_location=args.device))
        model.eval() 
        _final_test_loss, _final_test_acc, _final_test_f1 = val_one_epoch(
            testloader, model, loss_fn, args.device
        )
        final_test_loss_float = to_float(_final_test_loss)
        final_test_acc_float = to_float(_final_test_acc)
        final_test_f1_float = to_float(_final_test_f1)

        print(f"Final Test Results (Best Model): Loss: {final_test_loss_float:.4f}, Acc: {final_test_acc_float:.2f}%, F1: {final_test_f1_float:.4f}")
        performance_logger.log_test(final_test_loss_float, final_test_acc_float, final_test_f1_float)
    else:
        print(f"Warning: Best model checkpoint not found at {best_model_path}. Ensure EarlyStopping saves to this path or adjust loading logic.")
    
    performance_logger.save() 

    # --- Calculate Efficiency Metrics ---
    avg_time_per_epoch_seconds = (
        total_training_time_seconds / epochs_completed if epochs_completed > 0 else 0.0
    )

    if args.device.type == "cuda" and cuda_device_idx_for_stats is not None:
        peak_gpu_memory_allocated_mb = torch.cuda.max_memory_allocated(cuda_device_idx_for_stats) / (1024**2)
        peak_gpu_memory_reserved_mb = torch.cuda.max_memory_reserved(cuda_device_idx_for_stats) / (1024**2)
        peak_gpu_mem_usage_training_allocated_mb = peak_gpu_memory_allocated_mb - initial_gpu_mem_allocated
        peak_gpu_mem_usage_training_reserved_mb = peak_gpu_memory_reserved_mb - initial_gpu_mem_reserved
    else:
        peak_gpu_memory_allocated_mb = 0.0
        peak_gpu_memory_reserved_mb = 0.0
        peak_gpu_mem_usage_training_allocated_mb = 0.0
        peak_gpu_mem_usage_training_reserved_mb = 0.0

    # --- Log Efficiency Metrics ---
    efficiency_csv_filename = f"{args.model_name}_{args.metrics_csv_file}"
    efficiency_txt_filename = f"{args.model_name}_{args.metrics_txt_file}"
    metrics_csv_path = os.path.join(args.results_dir, efficiency_csv_filename)
    metrics_txt_path = os.path.join(args.results_dir, efficiency_txt_filename)

    best_score_float = to_float(early_stopper.best_score) if early_stopper.best_score is not None else float('nan')

    metrics_summary = {
        "timestamp": run_timestamp,
        "model_name": args.model_name,
        "device": args.device.type, 
        "num_epochs_run": epochs_completed,
        "configured_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "adapter_dim (model_dim)": args.adapter_dim,
        "num_latent_tokens": args.num_latent,
        "seed": args.seed,
        "model_instantiation_time_seconds": round(model_instantiation_time_seconds, 2),
        "total_training_time_seconds": round(total_training_time_seconds, 2),
        "avg_time_per_epoch_seconds": round(avg_time_per_epoch_seconds, 2),
        "peak_gpu_memory_allocated_mb_total": round(peak_gpu_memory_allocated_mb, 2),
        "peak_gpu_memory_reserved_mb_total": round(peak_gpu_memory_reserved_mb, 2),
        "peak_gpu_memory_allocated_mb_training_specific": round(peak_gpu_mem_usage_training_allocated_mb, 2),
        "peak_gpu_memory_reserved_mb_training_specific": round(peak_gpu_mem_usage_training_reserved_mb, 2),
        "model_total_parameters": model_total_params,
        "model_trainable_parameters": model_trainable_params,
        "final_best_validation_accuracy": f"{best_score_float:.2f}%" if not math.isnan(best_score_float) else 'N/A',
        "final_test_loss": round(final_test_loss_float, 4) if not math.isnan(final_test_loss_float) else 'N/A',
        "final_test_accuracy": f"{final_test_acc_float:.2f}%" if not math.isnan(final_test_acc_float) else 'N/A',
        "final_test_f1": round(final_test_f1_float, 4) if not math.isnan(final_test_f1_float) else 'N/A',
    }

    print("\n--- Model Training Efficiency Metrics Summary ---")
    for key, value in metrics_summary.items():
        print(f"{key}: {value}")

    # Append to CSV
    csv_file_exists = os.path.isfile(metrics_csv_path)
    try:
        with open(metrics_csv_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=metrics_summary.keys())
            if not csv_file_exists or os.path.getsize(metrics_csv_path) == 0:
                writer.writeheader()
            writer.writerow(metrics_summary)
        print(f"\nEfficiency metrics appended to CSV: '{metrics_csv_path}'")
    except IOError as e:
        print(f"Error writing to CSV file '{metrics_csv_path}': {e}")

    # Append human-readable summary to TXT
    try:
        with open(metrics_txt_path, 'a') as txtfile:
            txtfile.write(f"--- Metrics for {args.model_name} Training Run: {run_timestamp} ---\n")
            txtfile.write(f"Model Name: {args.model_name}\n")
            txtfile.write(f"Device: {args.device.type}\n")
            for key, value in metrics_summary.items():
                if key not in ["timestamp", "model_name", "device"]:
                    txtfile.write(f"{key.replace('_', ' ').capitalize()}: {value}\n")
            txtfile.write("\n")
        print(f"Human-readable efficiency metrics appended to TXT: '{metrics_txt_path}'")
    except IOError as e:
        print(f"Error writing to TXT file '{metrics_txt_path}': {e}")

    if args.device.type == "cuda":
        torch.cuda.empty_cache() 
    print(f"{args.model_name} training script finished.")


if __name__ == "__main__":
    opts = parse_options()
    train_test(opts)
