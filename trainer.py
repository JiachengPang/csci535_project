import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score


class Trainer:
  def __init__(self, model, optimizer, scheduler=None, device='cuda'):
    self.model = model
    self.optimizer = optimizer
    self.scheduler = scheduler
    self.device = device
  
  def step(self, batch):
    text_inputs = {
      k: (v.to(self.device).bool() if k == 'attention_mask' else v.to(self.device))
      for k, v in batch['text_inputs'].items()
    }
    audio_inputs = {k: v.to(self.device) for k, v in batch['audio_inputs'].items()}
    labels = batch['labels'].to(self.device)

    logits = self.model(text_inputs, audio_inputs)
    loss = F.cross_entropy(logits, labels)
    preds = torch.argmax(logits, dim=-1)
    return loss, preds, labels
  
  def train_one_epoch(self, loader, epoch):
    self.model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    loop = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for batch in loop:
      self.optimizer.zero_grad()
      loss, preds, labels = self.step(batch)
      loss.backward()
      self.optimizer.step()
      if self.scheduler:
        self.scheduler.step()

      total_loss += loss.item()
      all_preds.extend(preds.detach().cpu().tolist())
      all_labels.extend(labels.detach().cpu().tolist())
      loop.set_postfix(loss=loss.item())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return total_loss / len(loader), acc, f1
  
  def evaluate(self, loader):
    self.model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
      for batch in tqdm(loader, desc='Val'):
        loss, preds, labels = self.step(batch)
        total_loss += loss
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return total_loss / len(loader), acc, f1