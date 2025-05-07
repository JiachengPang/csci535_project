import copy
import torch
from transformers import RobertaModel, RobertaTokenizer, HubertModel, Wav2Vec2FeatureExtractor
from models import XNormModel, EarlyFusionModel, LateFusionModel # Assuming these are in your project
from utils import get_iemocap_data_loaders, collate_fn_raw, MetricsLogger # Assuming these are in your project
from trainer import Trainer # Assuming this is in your project
import argparse
import time # Added for timing
import csv  # Added for CSV logging
import os   # Added for file operations
from datetime import datetime # Added for timestamping
import math # For math.isnan

def get_model_parameters(model, trainable_only=True):
    """Calculates the number of parameters in a model."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

def to_float(value):
    """Converts a value to a Python float, handling tensors and NumPy types."""
    if isinstance(value, torch.Tensor):
        return value.item()
    return float(value)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Encoder training script started on device: {device}')
    if device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False # As in original script

    emotion_labels = ['angry', 'frustrated', 'happy', 'sad', 'neutral']

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Encoder Model Training Script")
    parser.add_argument('--model', type=str, default='xnorm', choices=['xnorm', 'early', 'late'],
                        help="Type of encoder model to train.")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs.")
    parser.add_argument('--patience', type=int, default=5, help="Patience for early stopping.")
    parser.add_argument('--batch_size', type=int, default=16,
                        help="Batch size for training and evaluation. Note: XNorm might require smaller batch_size like 2-4 if memory constrained.")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="Learning rate for the optimizer.")
    parser.add_argument('--results_dir', type=str, default="./temp_results",
                        help="Directory to save checkpoints, logs, and metrics.")
    parser.add_argument('--metrics_csv_file', type=str, default="encoder_training_efficiency.csv",
                        help="CSV file name to log efficiency metrics for encoder training.")
    parser.add_argument('--metrics_txt_file', type=str, default="encoder_efficiency_metrics.txt",
                        help="Text file name to append human-readable efficiency metrics for encoder training.")
    parser.add_argument('--iemocap_path', type=str, default='./iemocap', help="Path to IEMOCAP dataset for xnorm.")
    parser.add_argument('--iemocap_precomputed_path', type=str, default='./iemocap_precomputed', help="Path to precomputed IEMOCAP dataset.")


    args = parser.parse_args()

    model_choice = args.model
    n_epoch = args.epochs
    patience = args.patience
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    results_dir = args.results_dir
    
    os.makedirs(results_dir, exist_ok=True) # Ensure results directory exists

    metrics_csv_path = os.path.join(results_dir, args.metrics_csv_file)
    metrics_txt_path = os.path.join(results_dir, args.metrics_txt_file)
    
    print(f'Model to train: {model_choice}')
    
    # Existing MetricsLogger for loss/acc/f1
    performance_logger = MetricsLogger(save_path=os.path.join(results_dir, f"{model_choice}_training_metrics.json"))

    # --- Overall Timing and GPU Setup for Efficiency Metrics ---
    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        initial_gpu_mem_allocated = torch.cuda.memory_allocated(device) / (1024**2)
        initial_gpu_mem_reserved = torch.cuda.memory_reserved(device) / (1024**2)
    else:
        initial_gpu_mem_allocated = 0
        initial_gpu_mem_reserved = 0

    # --- Model Instantiation Time ---
    model_instantiation_start_time = time.time()
    
    model = None # Initialize model variable
    text_checkpoint = 'roberta-base' # Define here for use in data loader section too
    audio_checkpoint = 'facebook/hubert-base-ls960' # Define here

    # Create model
    if model_choice == 'xnorm':
        # These are loaded onto CPU first by Hugging Face
        roberta = RobertaModel.from_pretrained(text_checkpoint)
        hubert = HubertModel.from_pretrained(audio_checkpoint)
        
        # Freeze params (as in original script)
        for param in roberta.parameters():
            param.requires_grad = False
        for param in hubert.parameters():
            param.requires_grad = False

        model = XNormModel(roberta=roberta, hubert=hubert, num_classes=len(emotion_labels))
    elif model_choice == 'early':
        # Assuming EarlyFusionModel takes num_classes
        model = EarlyFusionModel(num_classes=len(emotion_labels))
    elif model_choice == 'late':
        # Assuming LateFusionModel takes num_classes
        model = LateFusionModel(num_classes=len(emotion_labels))
    else:
        raise ValueError(f"Unsupported model choice: {model_choice}")

    model_instantiation_end_time = time.time()
    model_instantiation_time_seconds = model_instantiation_end_time - model_instantiation_start_time

    # --- Get Model Parameters ---
    # For XNorm, roberta and hubert are frozen. Trainable params are only in the fusion/classification layers.
    model_total_params = get_model_parameters(model, trainable_only=False)
    model_trainable_params = get_model_parameters(model, trainable_only=True)

    # --- Data Loaders ---
    print(f"Using batch size: {batch_size} for model {model_choice}")
    if model_choice == 'xnorm':
        # These tokenizers/processors are loaded on CPU
        text_tokenizer_for_loader = RobertaTokenizer.from_pretrained(text_checkpoint) 
        audio_processor_for_loader = Wav2Vec2FeatureExtractor.from_pretrained(audio_checkpoint) 

        train_loader, val_loader, test_loader = get_iemocap_data_loaders(
            path=args.iemocap_path,
            precomputed=False,
            batch_size=batch_size, # Use argument
            num_workers=0, # As in original
            collate_fn=lambda b: collate_fn_raw(b, text_tokenizer_for_loader, audio_processor_for_loader),
        )
    else: # early, late
        train_loader, val_loader, test_loader = get_iemocap_data_loaders(
            path=args.iemocap_precomputed_path,
            precomputed=True,
            batch_size=batch_size, # Use argument
            num_workers=0, # As in original
            collate_fn=None, # As in original for precomputed
        )
    
    # --- Optimizer and Trainer ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) # Use argument for lr
    
    # Move model to device *before* passing to Trainer if Trainer expects it there
    model.to(device)
    trainer = Trainer(model, optimizer, device=device) # Pass model already on device
    
    # --- Training Loop ---
    best_val_loss = float('inf')
    best_model_state = None
    early_stopping_counter = 0 
    epochs_completed = 0

    print(f"\nStarting training for {model_choice} model for {n_epoch} epochs...")
    if device == "cuda":
        torch.cuda.synchronize() 
    training_start_time = time.time()

    for epoch in range(1, n_epoch + 1):
        epochs_completed += 1
        # trainer.train_one_epoch and trainer.evaluate are expected to return Python floats or values convertible to float
        train_loss, train_acc, train_f1 = trainer.train_one_epoch(train_loader, epoch)
        val_loss, val_acc, val_f1 = trainer.evaluate(val_loader)

        print(f'Epoch {epoch}/{n_epoch}')
        print(f'  Train Loss: {to_float(train_loss):.4f}, Train Acc: {to_float(train_acc):.4f}, Train F1: {to_float(train_f1):.4f}')
        print(f'  Val Loss  : {to_float(val_loss):.4f}, Val Acc  : {to_float(val_acc):.4f}, Val F1  : {to_float(val_f1):.4f}')
        
        performance_logger.log_train(to_float(train_loss), to_float(train_acc), to_float(train_f1))
        performance_logger.log_val(to_float(val_loss), to_float(val_acc), to_float(val_f1))
        performance_logger.save() 

        current_val_loss_float = to_float(val_loss) # Convert once for comparison
        if current_val_loss_float < best_val_loss: # best_val_loss is already a float
            best_val_loss = current_val_loss_float
            best_model_state = copy.deepcopy(trainer.model.state_dict()) 
            early_stopping_counter = 0
            print('  Best model state updated.')
        else:
            early_stopping_counter += 1
            print(f'  No improvement for {early_stopping_counter} epoch(s). Best Val Loss: {best_val_loss:.4f}')
        
        if early_stopping_counter >= patience:
            print(f'Early stopping triggered after {epoch} epochs.')
            break
    
    if device == "cuda":
        torch.cuda.synchronize() 
    training_end_time = time.time()
    total_training_time_seconds = training_end_time - training_start_time

    # --- Save Best Model Checkpoint ---
    final_epoch_reached_for_best_model = epochs_completed - early_stopping_counter 
    if best_model_state:
        trainer.model.load_state_dict(best_model_state) 
        checkpoint_save_path = os.path.join(results_dir, f'{model_choice}_checkpoint.pth')
        torch.save({
            'epoch': final_epoch_reached_for_best_model, 
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 
            'val_loss': best_val_loss, # This is now a float
            'model_choice': model_choice, 
            'num_classes': len(emotion_labels) 
            }, checkpoint_save_path)
        print(f"Best model from epoch {final_epoch_reached_for_best_model} saved to {checkpoint_save_path} (Val Loss: {best_val_loss:.4f})")
    else:
        print("No best model state was found (e.g., training stopped before first epoch completed or an error occurred).")
    
    # --- Final Evaluation on Test Set (using the best model) ---
    test_loss_float, test_acc_float, test_f1_float = float('nan'), float('nan'), float('nan')
    if best_model_state: 
        trainer.model.eval() 
        test_loss, test_acc, test_f1 = trainer.evaluate(test_loader)
        test_loss_float = to_float(test_loss)
        test_acc_float = to_float(test_acc)
        test_f1_float = to_float(test_f1)
        print(f'Final Test Results (best model): Loss: {test_loss_float:.4f}, Acc: {test_acc_float:.4f}, F1: {test_f1_float:.4f}')
        performance_logger.log_test(test_loss_float, test_acc_float, test_f1_float)
    else:
        print("Skipping final test evaluation as no best model was saved.")
    performance_logger.save() 

    # --- Calculate Efficiency Metrics ---
    avg_time_per_epoch_seconds = (
        total_training_time_seconds / epochs_completed
        if epochs_completed > 0 else 0
    )

    if device == "cuda":
        peak_gpu_memory_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        peak_gpu_memory_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024**2)
        peak_gpu_mem_usage_training_allocated_mb = peak_gpu_memory_allocated_mb - initial_gpu_mem_allocated
        peak_gpu_mem_usage_training_reserved_mb = peak_gpu_memory_reserved_mb - initial_gpu_mem_reserved
    else:
        peak_gpu_memory_allocated_mb = 0
        peak_gpu_memory_reserved_mb = 0
        peak_gpu_mem_usage_training_allocated_mb = 0
        peak_gpu_mem_usage_training_reserved_mb = 0

    # --- Log Efficiency Metrics ---
    metrics_summary = {
        "timestamp": run_timestamp,
        "model_type": model_choice,
        "device": device,
        "num_epochs_run": epochs_completed,
        "configured_epochs": n_epoch,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "model_instantiation_time_seconds": round(model_instantiation_time_seconds, 2),
        "total_training_time_seconds": round(total_training_time_seconds, 2),
        "avg_time_per_epoch_seconds": round(avg_time_per_epoch_seconds, 2),
        "peak_gpu_memory_allocated_mb_total": round(peak_gpu_memory_allocated_mb, 2),
        "peak_gpu_memory_reserved_mb_total": round(peak_gpu_memory_reserved_mb, 2),
        "peak_gpu_memory_allocated_mb_training_specific": round(peak_gpu_mem_usage_training_allocated_mb, 2),
        "peak_gpu_memory_reserved_mb_training_specific": round(peak_gpu_mem_usage_training_reserved_mb, 2),
        "model_total_parameters": model_total_params,
        "model_trainable_parameters": model_trainable_params,
        "final_best_validation_loss": round(best_val_loss, 4) if best_val_loss != float('inf') else 'N/A',
        "final_test_loss": round(test_loss_float, 4) if not math.isnan(test_loss_float) else 'N/A',
        "final_test_accuracy": round(test_acc_float, 4) if not math.isnan(test_acc_float) else 'N/A',
        "final_test_f1": round(test_f1_float, 4) if not math.isnan(test_f1_float) else 'N/A',
    }

    print("\n--- Encoder Training Efficiency Metrics Summary ---")
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
            txtfile.write(f"--- Metrics for Encoder Training Run: {run_timestamp} ---\n")
            txtfile.write(f"Model Type: {model_choice}\n")
            txtfile.write(f"Device: {device}\n")
            for key, value in metrics_summary.items():
                if key not in ["timestamp", "model_type", "device"]:
                    txtfile.write(f"{key.replace('_', ' ').capitalize()}: {value}\n")
            txtfile.write("\n")
        print(f"Human-readable efficiency metrics appended to TXT: '{metrics_txt_path}'")
    except IOError as e:
        print(f"Error writing to TXT file '{metrics_txt_path}': {e}")

    if device == "cuda":
        torch.cuda.empty_cache()
    print("Encoder training script finished.")

if __name__ == '__main__':
    main()
