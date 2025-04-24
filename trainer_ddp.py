import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import torch.distributed as dist
import collections # For gathering objects if needed (alternative to tensors)
import os # For checking environment variables

# Helper function to gather python objects (like lists of varying lengths)
# More robust than converting large lists to tensors if memory is tight or lists differ in size
def gather_objects(obj):
    """ Gather arbitrary python objects from all ranks """
    world_size = dist.get_world_size()
    # Ensure obj is pickleable
    # Size of pickled object must be less than 2GB for NCCL backend with default settings
    gathered_obj = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_obj, obj)
    # Concatenate lists if the objects are lists
    if isinstance(obj, list):
        final_list = []
        for lst in gathered_obj:
            if lst is not None: # Ensure None objects aren't extended
                final_list.extend(lst)
        return final_list
    # Handle other types if needed, otherwise just return the list of objects from all ranks
    return gathered_obj


class Trainer:
    def __init__(self, model, optimizer, scheduler=None, device='cuda'):
        """
        Initializes the Trainer.

        Args:
            model: The model to train (potentially DDP-wrapped).
            optimizer: The optimizer.
            scheduler: Optional learning rate scheduler.
            device: The device to run training on ('cuda' or 'cpu').
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss() # Define loss function

        # Determine rank and world size from distributed environment if initialized
        self.rank = 0
        self.world_size = 1
        self.is_distributed = dist.is_available() and dist.is_initialized()
        if self.is_distributed:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            print(f"Trainer initialized on Rank {self.rank}/{self.world_size}")
        else:
            print("Trainer initialized in non-distributed mode.")

    def _move_batch_to_device(self, batch):
        """Moves batch contents to the designated device."""
        # Handle text inputs
        text_inputs_on_device = {}
        if batch.get('text_inputs') and isinstance(batch['text_inputs'], dict):
             text_inputs_on_device = {
                 k: v.to(self.device) for k, v in batch['text_inputs'].items() if isinstance(v, torch.Tensor)
            }
             # Handle boolean attention mask specifically if needed by model
             if 'attention_mask' in text_inputs_on_device and text_inputs_on_device['attention_mask'].dtype != torch.bool:
                 # Some models might expect bool, others long/float. Adjust if necessary.
                 # text_inputs_on_device['attention_mask'] = text_inputs_on_device['attention_mask'].bool()
                 pass # Keep as is unless model requires bool specifically

        # Handle audio inputs
        audio_inputs_on_device = {}
        if batch.get('audio_inputs') and isinstance(batch['audio_inputs'], dict):
             audio_inputs_on_device = {
                k: v.to(self.device) for k, v in batch['audio_inputs'].items() if isinstance(v, torch.Tensor)
            }

        # Handle labels
        labels_on_device = None
        if batch.get('labels') is not None and isinstance(batch['labels'], torch.Tensor):
             labels_on_device = batch['labels'].to(self.device)

        return text_inputs_on_device, audio_inputs_on_device, labels_on_device


    def step(self, batch):
        """Performs a single training or evaluation step."""
        text_inputs, audio_inputs, labels = self._move_batch_to_device(batch)

        if labels is None:
            raise ValueError("Labels are missing or not a tensor in the batch")

        # Forward pass
        # Assumes model's forward signature matches the input dict structure
        try:
            # Use the model directly (it might be DDP-wrapped)
            logits = self.model(text_inputs=text_inputs, audio_inputs=audio_inputs)
        except TypeError as e:
             print("Error during model forward pass. Check if model signature matches input format.")
             print(f"Text Inputs keys: {text_inputs.keys() if text_inputs else 'None'}")
             print(f"Audio Inputs keys: {audio_inputs.keys() if audio_inputs else 'None'}")
             raise e

        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=-1)
        return loss, preds, labels

    def train_one_epoch(self, loader, epoch):
        """Trains the model for one epoch."""
        self.model.train() # Set model to training mode
        total_loss = 0.0
        local_preds, local_labels = [], [] # Store results for this rank only

        # Disable tqdm progress bar for non-master processes
        loop = tqdm(loader, desc=f"Epoch {epoch} [Train] Rank {self.rank}", disable=(self.is_distributed and self.rank != 0))

        for batch in loop:
            self.optimizer.zero_grad()
            loss, preds, labels = self.step(batch)

            # Backward pass and optimization
            loss.backward() # DDP handles gradient synchronization here
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            # Accumulate loss and predictions/labels for this rank
            batch_loss = loss.item()
            total_loss += batch_loss
            local_preds.extend(preds.detach().cpu().tolist())
            local_labels.extend(labels.detach().cpu().tolist())

            # Update progress bar postfix only on master rank
            if self.rank == 0:
                 loop.set_postfix(loss=batch_loss)

        # --- Aggregate Metrics Across All Ranks ---
        # Aggregate total loss
        final_loss_sum = torch.tensor(total_loss).to(self.device)
        num_batches_local = len(loader)
        num_batches_tensor = torch.tensor(num_batches_local).to(self.device)

        if self.is_distributed:
            # Sum the losses from all ranks
            dist.all_reduce(final_loss_sum, op=dist.ReduceOp.SUM)
            # Sum the number of batches processed across all ranks
            dist.all_reduce(num_batches_tensor, op=dist.ReduceOp.SUM)
            # Gather predictions and labels using object gathering
            all_preds = gather_objects(local_preds)
            all_labels = gather_objects(local_labels)
            # Ensure barrier so all ranks have finished gathering before rank 0 calculates metrics
            dist.barrier()
        else:
            # If not distributed, the local results are the final results
            all_preds = local_preds
            all_labels = local_labels

        # Calculate final metrics ONLY on Rank 0
        avg_loss = 0.0
        acc = 0.0
        f1 = 0.0
        if self.rank == 0:
            num_batches_total = num_batches_tensor.item()
            avg_loss = final_loss_sum.item() / num_batches_total if num_batches_total > 0 else 0.0
            if all_labels and all_preds: # Ensure lists are not empty
                 # Use zero_division=0 to avoid warnings if a class has no predictions/labels
                 acc = accuracy_score(all_labels, all_preds)
                 f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
                 print(f"\nEpoch {epoch} [Train] Rank {self.rank} - Global Avg Loss: {avg_loss:.4f}, Acc: {acc:.4f}, F1: {f1:.4f}")
            else:
                 print(f"Warning: Empty predictions/labels on rank 0 during training epoch {epoch}.")
                 avg_loss = final_loss_sum.item() / num_batches_total if num_batches_total > 0 else 0.0 # Still report loss

        # Ensure all processes finish the epoch before proceeding (optional but good practice)
        if self.is_distributed:
            dist.barrier()

        # Return aggregated metrics (meaningful only on rank 0)
        return avg_loss, acc, f1


    def evaluate(self, loader, desc='Val'):
        """Evaluates the model on a given dataset loader."""
        self.model.eval() # Set model to evaluation mode
        total_loss = 0.0
        local_preds, local_labels = [], [] # Store results for this rank

        # Disable tqdm progress bar for non-master processes
        loop = tqdm(loader, desc=f"{desc} Rank {self.rank}", disable=(self.is_distributed and self.rank != 0))

        with torch.no_grad(): # Disable gradient calculations
            for batch in loop:
                loss, preds, labels = self.step(batch)

                # Accumulate loss and predictions/labels for this rank
                total_loss += loss.item() # Accumulate loss item per batch
                local_preds.extend(preds.cpu().tolist())
                local_labels.extend(labels.cpu().tolist())

        # --- Aggregate Metrics Across All Ranks ---
        # Aggregate total loss
        final_loss_sum = torch.tensor(total_loss).to(self.device)
        num_batches_local = len(loader)
        num_batches_tensor = torch.tensor(num_batches_local).to(self.device)

        if self.is_distributed:
            # Sum the losses from all ranks
            dist.all_reduce(final_loss_sum, op=dist.ReduceOp.SUM)
             # Sum the number of batches processed across all ranks
            dist.all_reduce(num_batches_tensor, op=dist.ReduceOp.SUM)
            # Gather predictions and labels
            all_preds = gather_objects(local_preds)
            all_labels = gather_objects(local_labels)
            # Ensure barrier so all ranks have finished gathering before rank 0 calculates metrics
            dist.barrier()
        else:
            # If not distributed, the local results are the final results
            all_preds = local_preds
            all_labels = local_labels

        # Calculate final metrics ONLY on Rank 0
        avg_loss = 0.0
        acc = 0.0
        f1 = 0.0
        if self.rank == 0:
            num_batches_total = num_batches_tensor.item()
            avg_loss = final_loss_sum.item() / num_batches_total if num_batches_total > 0 else 0.0
            if all_labels and all_preds:
                 acc = accuracy_score(all_labels, all_preds)
                 f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
                 print(f"\n{desc} Rank {self.rank} - Global Avg Loss: {avg_loss:.4f}, Acc: {acc:.4f}, F1: {f1:.4f}")
            else:
                 print(f"Warning: Empty predictions/labels on rank 0 during {desc}.")
                 avg_loss = final_loss_sum.item() / num_batches_total if num_batches_total > 0 else 0.0 # Still report loss

        # Ensure all processes finish evaluation before proceeding (optional)
        if self.is_distributed:
            dist.barrier()

        # Return aggregated metrics (meaningful only on rank 0)
        return avg_loss, acc, f1
