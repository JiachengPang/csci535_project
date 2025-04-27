import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

DEFAULT_PROMPT = "Describe the emotion expressed in the this speech by focusing on both the speaker's words and vocal characteristics"

class Trainer:
  def __init__(self, model, optimizer, scheduler=None, device='cuda'):
    self.model = model
    self.optimizer = optimizer
    self.scheduler = scheduler
    self.device = device
  
  def step(self, batch):
    if 'text_inputs' in batch and 'audio_inputs' in batch: # raw
      text_inputs = {
        k: (v.to(self.device).bool() if k == 'attention_mask' else v.to(self.device))
        for k, v in batch['text_inputs'].items()
      }
      audio_inputs = {k: v.to(self.device) for k, v in batch['audio_inputs'].items()}
      logits = self.model(text_inputs, audio_inputs)
    else: # precomputed - 'audio_emb' and 'text_emb'
      inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'label'}
      logits = self.model(**inputs)
    
    labels = batch['label'].to(self.device) if 'label' in batch else batch['labels'].to(self.device)
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


class CaptioningTrainer:
  def __init__(self, encoder, projector, decoder, decoder_tokenizer, optimizer, scheduler=None, device='cuda'):
    self.encoder = encoder.to(device)
    self.projector = projector.to(device)
    self.decoder = decoder.to(device)
    self.optimizer = optimizer
    self.scheduler = scheduler
    self.device = device
    self.prompt_tokens = decoder_tokenizer(
      DEFAULT_PROMPT,
      return_tensors='pt',
      padding=True,
      truncation=True
    ).to(device)

    # freeze encoder
    self.encoder.eval()
    for param in self.encoder.parameters():
      param.requires_grad = False
  
  def step(self, batch):
    if 'text_inputs' in batch and 'audio_inputs' in batch: # raw
      text_inputs = {k: v.to(self.device) for k, v in batch['text_inputs'].items()}
      audio_inputs = {k: v.to(self.device) for k, v in batch['audio_inputs'].items()}
      features = self.encoder(text_inputs, audio_inputs, return_features=True)
    else: # precomputed - 'audio_emb' and 'text_emb'
      audio_embs = batch['audio_embs'].to(self.device)
      text_embs = batch['text_embs'].to(self.device)
      features = self.encoder(text_embs, audio_embs, return_features=True)

    prompt_input_ids = self.prompt_tokens.input_ids.to(self.device)         # (1, prompt_len)
    prompt_attention_mask = self.prompt_tokens.attention_mask.to(self.device)  # (1, prompt_len)
    batch_size = features.size(0)
    input_ids = prompt_input_ids.expand(batch_size, -1)          # (B, prompt_len)
    attention_mask = prompt_attention_mask.expand(batch_size, -1)  # (B, prompt_len)

    labels = batch['labels'].to(self.device)

    projected = self.projector(features)
    
    outputs = self.decoder(
      prefix_emb=projected,
      input_ids=input_ids,
      attention_mask=attention_mask,
      labels=labels
    )

    loss = outputs.loss
    return loss
  
  def train_one_epoch(self, loader, epoch):
    self.projector.train()
    self.decoder.train()

    total_loss = 0
    loop = tqdm(loader, desc=f'Epoch {epoch} [Caption Train]')
    for batch in loop:
      self.optimizer.zero_grad()
      loss = self.step(batch)
      loss.backward()
      self.optimizer.step()
      if self.scheduler:
        self.scheduler.step()
      
      total_loss += loss.item()
      loop.set_postfix(loss=loss.item())
    
    return total_loss / len(loader)

  def evaluate(self, loader):
    self.projector.eval()
    self.decoder.eval()

    total_loss = 0

    with torch.no_grad():
      for batch in tqdm(loader, desc='Caption val'):
        loss = self.step(batch)
        total_loss += loss.item()
    
    return total_loss / len(loader)
