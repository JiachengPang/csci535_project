import os
import copy
import json
import time
import datetime # Added for timeout

# --- User Provided Imports ---
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch
from torch.utils.data import DataLoader # Keep DataLoader for use
from transformers import RobertaModel, RobertaTokenizer, HubertModel, Wav2Vec2FeatureExtractor
from models import XNormModel
# Use the exact function names provided by the user, assuming they exist in 'utils.py'
from utils import get_iemocap_datasets_ddp, collate_fn_raw_ddp, MetricsLogger
# Use the exact Trainer name provided by the user, assuming it exists in 'trainer_ddp.py'
from trainer_ddp import Trainer
# --- End User Provided Imports ---


# --- DDP Setup and Cleanup ---
def setup(rank, world_size):
    """Initializes the distributed process group."""
    # torchrun automatically sets MASTER_ADDR and MASTER_PORT env variables
    # We can use init_method='env://'

    # *** Trying 'gloo' backend for diagnostics ***
    backend_to_try = "gloo"
    print(f"Rank {rank}/{world_size}: Attempting to initialize process group using 'env://' (backend: {backend_to_try})...")

    # Add a timeout (e.g., 60 seconds)
    timeout_delta = datetime.timedelta(seconds=60)
    dist.init_process_group(
        backend=backend_to_try, # Use 'gloo' instead of 'nccl'
        init_method="env://", # Use environment variables set by torchrun
        world_size=world_size,
        rank=rank,
        timeout=timeout_delta
    )
    # Pin each process to a specific GPU (still recommended even with gloo if using GPUs)
    if torch.cuda.is_available():
         torch.cuda.set_device(rank)
         print(f"Rank {rank}/{world_size}: Process group initialized (backend: {backend_to_try}), assigned to cuda:{rank}")
    else:
         print(f"Rank {rank}/{world_size}: Process group initialized (backend: {backend_to_try}), running on CPU")

    # Removed barrier from here in previous step


def cleanup():
    """Destroys the distributed process group."""
    if dist.is_initialized():
        print(f"Rank {dist.get_rank()}: Destroying process group.")
        dist.destroy_process_group()

# --- Main Worker Function (Executed per GPU) ---
def main_worker(rank, world_size, args):
    """Main function executed by each DDP process."""
    print(f"Rank {rank}/{world_size}: Starting main_worker.")
    try:
        setup(rank, world_size) # Setup DDP communication
    except Exception as e:
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"Rank {rank}/{world_size}: FAILED TO SETUP DDP - {e}")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # Attempt cleanup, might not work if init failed badly
        # cleanup() # Consider if cleanup is safe here
        return # Exit the failed process

    # Determine device after setup (might be CPU if gloo is used and no CUDA)
    if torch.cuda.is_available():
        device = rank # Each rank corresponds to a GPU index (cuda:0, cuda:1, ...)
    else:
        device = 'cpu' # Fallback to CPU if CUDA not available
        print(f"Rank {rank}: CUDA not available, using CPU.")

    is_master = (rank == 0) # Flag for master process

    # Add a barrier here to ensure setup is complete before proceeding
    print(f"Rank {rank}: Waiting on barrier immediately after setup function...")
    dist.barrier()
    print(f"Rank {rank}: Passed barrier after setup function.")


    if is_master: print(f"Master process (Rank {rank}) starting setup.")
    # Only disable TF32 if using CUDA
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False

    # Logger: Initialized only on the master process using the imported class
    logger = None
    if is_master:
        print(f"Rank {rank}: Initializing MetricsLogger...")
        logger = MetricsLogger(save_path=args['log_path'])
        print(f"Rank {rank}: MetricsLogger initialized (path: {args['log_path']})")

    # --- Load Tokenizer and Processor ---
    # Add barrier before heavy loading to ensure setup is stable
    print(f"Rank {rank}: Waiting on barrier before loading tokenizers/processors...")
    dist.barrier()
    print(f"Rank {rank}: Passed barrier. Loading tokenizer ({args['text_checkpoint']})...")
    tokenizer = RobertaTokenizer.from_pretrained(args['text_checkpoint'])
    print(f"Rank {rank}: Loading processor ({args['audio_checkpoint']})...")
    processor = Wav2Vec2FeatureExtractor.from_pretrained(args['audio_checkpoint'])
    print(f"Rank {rank}: Tokenizer and processor loaded.")

    # --- Load Datasets using DDP-aware function ---
    # Only rank 0 should ideally perform the download/initial load if using HF datasets cache
    # But splitting needs to happen before dataset object creation per rank
    # Let's have rank 0 load/split and then others load from cache (if applicable)
    # This requires careful handling or assuming dataset is already cached/local
    print(f"Rank {rank}: Waiting on barrier before loading dataset...")
    dist.barrier()
    if is_master: print(f"Rank {rank}: Master calling get_iemocap_datasets_ddp...")
    # All ranks call this, assuming load_from_disk is safe for concurrent access
    # or the underlying data loading handles DDP correctly.
    train_ds, val_ds, test_ds = get_iemocap_datasets_ddp(
        path=args['data_path'], # This path comes from the args dictionary below
        precomputed=args.get('precomputed', False),
        seed=args['seed'],
        first_n=args.get('first_n', 0)
    )
    # Basic check if datasets were loaded
    if train_ds is None or val_ds is None or test_ds is None:
         print(f"ERROR Rank {rank}: Dataset loading function returned None. Exiting.")
         cleanup()
         return
    print(f"Rank {rank}: Datasets loaded/split.")

    # --- Create Distributed Samplers ---
    print(f"Rank {rank}: Creating distributed samplers...")
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=args['seed'])
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
    test_sampler = DistributedSampler(test_ds, num_replicas=world_size, rank=rank, shuffle=False)
    print(f"Rank {rank}: Samplers created.")

    # --- Create DataLoaders ---
    print(f"Rank {rank}: Creating DataLoaders...")
    # Use the imported DDP collate function
    collate_fn_wrapper = lambda b: collate_fn_raw_ddp(b, tokenizer, processor, sampling_rate=args.get('sampling_rate', 16000))

    train_loader = DataLoader(
        train_ds, batch_size=args['batch_size_per_gpu'], sampler=train_sampler,
        num_workers=args['num_workers'], collate_fn=collate_fn_wrapper, pin_memory=True if device != 'cpu' else False, # Pin memory only for GPU
        shuffle=False # Sampler handles shuffling
    )
    val_loader = DataLoader(
        val_ds, batch_size=args['batch_size_per_gpu'], sampler=val_sampler,
        num_workers=args['num_workers'], collate_fn=collate_fn_wrapper, pin_memory=True if device != 'cpu' else False,
        shuffle=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=args['batch_size_per_gpu'], sampler=test_sampler,
        num_workers=args['num_workers'], collate_fn=collate_fn_wrapper, pin_memory=True if device != 'cpu' else False,
        shuffle=False
    )
    print(f"Rank {rank}: DataLoaders created.")
    if is_master: print(f"Master Rank {rank}: Batches per epoch (approx): {len(train_loader)}")

    # --- Initialize Models ---
    # Add barrier before model loading
    print(f"Rank {rank}: Waiting on barrier before model init...")
    dist.barrier()
    print(f"Rank {rank}: Passed barrier. Initializing base models...")
    # Load base models directly
    roberta = RobertaModel.from_pretrained(args['text_checkpoint'])
    hubert = HubertModel.from_pretrained(args['audio_checkpoint'])
    print(f"Rank {rank}: Base models initialized.")

    # --- Freeze Parameters ---
    print(f"Rank {rank}: Freezing base model parameters...")
    for param in roberta.parameters(): param.requires_grad = False
    for param in hubert.parameters(): param.requires_grad = False
    print(f"Rank {rank}: Base model parameters frozen.")

    # --- Initialize Custom Model ---
    print(f"Rank {rank}: Initializing XNormModel...")
    # Use the imported XNormModel
    num_classes = len(args.get('emotion_labels', [])) # Get num_classes from args
    if num_classes == 0:
         print(f"ERROR Rank {rank}: 'emotion_labels' not found or empty in args. Cannot determine num_classes.")
         cleanup()
         return
    model = XNormModel(roberta=roberta, hubert=hubert, num_classes=num_classes)
    print(f"Rank {rank}: Moving XNormModel to device {device}...")
    model.to(device) # Move model to the assigned GPU *before* DDP wrapping
    print(f"Rank {rank}: XNormModel initialized and moved to device.")

    # --- Wrap Model with DDP ---
    # Only wrap with DDP if using CUDA and world_size > 1
    if device != 'cpu' and world_size > 1:
        print(f"Rank {rank}: Waiting on barrier before DDP wrapping...")
        # Add a barrier *before* DDP init to ensure all models are on GPUs
        dist.barrier()
        print(f"Rank {rank}: Passed barrier before DDP init. Wrapping model...")
        # Note: DDP might not work well with 'gloo' backend for GPU models.
        # If using 'gloo', often you don't wrap the model or use a different strategy.
        # However, we keep DDP here assuming the user might switch back to 'nccl' if 'gloo' works for init.
        model = DDP(model, device_ids=[device], find_unused_parameters=args.get('ddp_find_unused', False))
        print(f"Rank {rank}: Model wrapped with DDP.")
    elif world_size > 1:
         print(f"Rank {rank}: Using CPU or single GPU, skipping DDP wrapping.")
         # Need to handle model access consistently later (no .module)
         # For simplicity, we might error out if world_size > 1 and device is cpu
         if device == 'cpu':
              print(f"ERROR Rank {rank}: DDP with CPU backend is complex and not fully handled here. Exiting.")
              cleanup()
              return


    # --- Optimizer ---
    print(f"Rank {rank}: Creating optimizer (AdamW, lr={args['learning_rate']})...")
    # If DDP was used, optimizer targets model.parameters(). If not, it's just model.parameters().
    optimizer = torch.optim.AdamW(model.parameters(), lr=args['learning_rate'])
    scheduler = None # Define scheduler if needed
    print(f"Rank {rank}: Optimizer created.")

    # --- Initialize DDP-Aware Trainer ---
    print(f"Rank {rank}: Initializing DDP-aware Trainer...")
    # Use the imported Trainer class, pass rank and world_size
    # Ensure Trainer handles the case where model is not DDP-wrapped (e.g., world_size=1)
    trainer = Trainer(
        model=model, optimizer=optimizer, scheduler=scheduler, device=device
    )
    print(f"Rank {rank}: Trainer initialized.")

    # --- Training Loop ---
    best_val_loss = float('inf')
    best_model_state = None
    patience = args['patience']
    n_epoch = args['epochs']
    counter = 0

    print(f"Rank {rank}: Entering training loop for {n_epoch} epochs...")
    # Add barrier before loop starts to ensure all setup is complete
    dist.barrier()
    if is_master: print("All ranks ready. Starting training loop.")


    for epoch in range(1, n_epoch + 1):
        epoch_start_time = time.time()
        if is_master: print(f"\n===== Epoch {epoch}/{n_epoch} =====")
        print(f"Rank {rank}: Starting Epoch {epoch}...")

        # Set epoch for distributed sampler (essential for proper shuffling)
        # Check if sampler exists (might not in single GPU case if adapted)
        if hasattr(train_sampler, 'set_epoch'):
             print(f"Rank {rank}: Setting sampler epoch {epoch}...")
             train_sampler.set_epoch(epoch)
             print(f"Rank {rank}: Sampler epoch set.")

        # Train one epoch - Trainer handles DDP logic and metric aggregation
        print(f"Rank {rank}: Calling trainer.train_one_epoch...")
        train_loss, train_acc, train_f1 = trainer.train_one_epoch(train_loader, epoch)
        print(f"Rank {rank}: Finished trainer.train_one_epoch.")

        # Evaluate - Trainer handles DDP logic and metric aggregation
        print(f"Rank {rank}: Calling trainer.evaluate...")
        val_loss, val_acc, val_f1 = trainer.evaluate(val_loader, desc='Val')
        print(f"Rank {rank}: Finished trainer.evaluate.")

        # --- Logging, Checkpointing & Early Stopping (Master Process Only) ---
        if is_master:
            epoch_duration = time.time() - epoch_start_time
            print(f'--- Epoch {epoch} Summary (Rank 0) --- Time: {epoch_duration:.2f}s ---')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}')
            print(f'  Val Loss  : {val_loss:.4f}, Val Acc  : {val_acc:.4f}, Val F1  : {val_f1:.4f}')

            if logger:
                # Use the imported logger instance
                logger.log_train(train_loss, train_acc, train_f1)
                logger.log_val(val_loss, val_acc, val_f1)
                logger.save() # Save metrics after each epoch

            # Checkpointing logic
            if val_loss < best_val_loss:
                print(f'  Val loss improved ({best_val_loss:.4f} --> {val_loss:.4f}). Saving model...')
                best_val_loss = val_loss
                # Save the underlying model's state dict (unwrap from DDP if applicable)
                model_to_save = model.module if isinstance(model, DDP) else model
                best_model_state = copy.deepcopy(model_to_save.state_dict())
                counter = 0 # Reset patience counter

                # Save the best model checkpoint
                save_obj = {
                    'epoch': epoch,
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                    'world_size': world_size # Optional: Store world size
                }
                checkpoint_path = args['checkpoint_path']
                # Ensure directory exists
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save(save_obj, checkpoint_path)
                print(f"  Best model checkpoint saved to '{checkpoint_path}'")

            else:
                counter += 1
                print(f'  Val loss did not improve. Patience counter: {counter}/{patience}')

            # Early stopping check
            if counter >= patience:
                print(f'  Early stopping triggered after epoch {epoch}.')
                # Signal other processes to stop (simple approach)
                stop_signal = torch.tensor(1).to(device)
            else:
                stop_signal = torch.tensor(0).to(device)
        else:
            # Non-master processes need a placeholder tensor for broadcast
             # Ensure tensor is on the correct device (CPU if device is CPU)
            stop_signal = torch.tensor(0).to(device if device != 'cpu' else 'cpu')


        # Broadcast the stop signal from rank 0 to all other ranks
        # Handle CPU case for broadcast tensor
        if device == 'cpu':
             # If using CPU, broadcast needs tensors on CPU
             dist.broadcast(stop_signal, src=0)
        else:
             # If using GPU, broadcast needs tensors on GPU
             dist.broadcast(stop_signal, src=0)


        # All processes check the signal
        if stop_signal.item() == 1:
            if is_master: print("Broadcasting stop signal.")
            else: print(f"Rank {rank}: Received stop signal. Breaking training loop.")
            break # Exit the loop on all processes

        # Barrier to ensure all processes finish the epoch before starting the next
        print(f"Rank {rank}: Reached end of epoch barrier.")
        dist.barrier()
        print(f"Rank {rank}: Passed end of epoch barrier.")


    if is_master: print("\nTraining loop finished.")

    # --- Final Test Evaluation ---
    if is_master: print("\n===== Final Test Evaluation =====")

    # Load the best model state before final evaluation (only on master initially)
    model_to_eval = model.module if isinstance(model, DDP) else model
    if is_master and best_model_state:
        print("Master process loading best model state for final test...")
        # Load state dict into the underlying model
        model_to_eval.load_state_dict(best_model_state)
    elif is_master:
        print("No best model state found from training, using last model state for testing.")

    # Ensure all processes have the *same* model weights for the final test.
    # Rank 0 holds the desired weights (either last or best).
    # We need to broadcast these weights to all other processes.
    if world_size > 1:
        print(f"Rank {rank}: Synchronizing model state for final test...")
        # Create a list containing the state_dict on rank 0, and None elsewhere
        state_dict_to_broadcast = model_to_eval.state_dict() if is_master else None
        # Use broadcast_object_list for arbitrary objects like state_dict
        object_list = [state_dict_to_broadcast]
        dist.broadcast_object_list(object_list, src=0)
        # Load the received state_dict on non-master ranks
        if not is_master:
            model_to_eval.load_state_dict(object_list[0])
        # Barrier to ensure loading is complete everywhere before evaluation
        dist.barrier()
        if is_master: print("Model state synchronized across all ranks.")


    # Evaluate on the test set - Trainer should handle DDP aggregation
    print(f"Rank {rank}: Evaluating on test set...")
    # Pass the potentially unwrapped model if not using DDP
    test_loss, test_acc, test_f1 = trainer.evaluate(test_loader, desc='Test')
    print(f"Rank {rank}: Finished test set evaluation.")


    # Log test results (only on master)
    if is_master:
        print(f"\n--- Test Set Results (Rank 0) ---")
        print(f'  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}')
        if logger:
            # Use the imported logger instance
            logger.log_test(test_loss, test_acc, test_f1)
            logger.save()
            print(f"  Test metrics logged to {args['log_path']}")

    # --- Cleanup ---
    # Add barrier before cleanup to ensure all processes finish evaluation
    print(f"Rank {rank}: Reached final barrier before cleanup.")
    dist.barrier()
    print(f"Rank {rank}: Passed final barrier.")
    cleanup()
    print(f"--> Finished main_worker on Rank {rank}")


# --- Main Execution Block ---
if __name__ == '__main__':
    # --- Configuration Arguments ---
    # Consider using argparse for command-line configuration
    args = {
        # ======================================================================
        # ===> Path to the dataset directory saved by datasets.save_to_disk <===
        # ======================================================================
        "data_path": './iemocap', # Corrected path based on user feedback
        # ======================================================================

        "log_path": "xnorm_training_metrics_ddp.json",
        "checkpoint_path": "xnorm_checkpoint_ddp.pth",
        "audio_checkpoint": 'facebook/hubert-base-ls960',
        "text_checkpoint": 'roberta-base',
        "emotion_labels": ['neutral', 'happy', 'sad', 'angry', 'frustrated', 'excited', 'fear', 'disgust', 'surprise', 'other'],
        "batch_size_per_gpu": 16,       # Batch size FOR EACH GPU
        "num_workers": 0,               # Dataloader workers per GPU (often 0 or low for DDP)
        "learning_rate": 1e-4,          # Base learning rate
        "epochs": 20,
        "patience": 5,                  # Early stopping patience
        "seed": 42,                     # Random seed for reproducibility
        "sampling_rate": 16000,         # Audio sampling rate for processor
        "precomputed": False,           # Set to True if using precomputed embeddings
        "first_n": 0,                   # Set > 0 to use only first N samples for testing
        "ddp_find_unused": False        # Set DDP find_unused_parameters flag
    }

    # --- DDP Launch ---
    world_size = torch.cuda.device_count()
    print(f"Found {world_size} GPUs.")

    if world_size < 1 and not torch.cuda.is_available():
         print("ERROR: No GPUs found and CUDA not available. Cannot run.")
    elif world_size == 0 and torch.cuda.is_available():
         print("ERROR: CUDA available but reports 0 devices. Check CUDA setup/visibility.")
    # elif world_size == 1: # Optional: Handle single GPU case differently if desired
    #     print("WARNING: Only one GPU found. Running DDP script in single-process mode.")
    #     main_worker(0, 1, args) # Call directly for rank 0, world_size 1
    else:
         # If world_size is 0 but CUDA is not available, run on CPU (world_size=1)
         effective_world_size = world_size if world_size > 0 else 1
         if effective_world_size == 1:
              print("Running in single-process mode (either 1 GPU or CPU).")
              main_worker(0, 1, args) # Rank 0, World Size 1
         else:
              print(f"Spawning {effective_world_size} processes for DDP training...")
              mp.spawn(main_worker,
                       args=(effective_world_size, args), # Pass world_size and args dict
                       nprocs=effective_world_size,
                       join=True) # Wait for all processes to finish

    print("\nMain script execution finished.")
