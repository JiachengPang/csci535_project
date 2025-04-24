import torch.nn as nn
from transformers import RobertaModel, HubertModel

class Classifier(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes, dropout=0.1):
    super().__init__()

    self.model = nn.Sequential(
      nn.Linear(input_size, hidden_size),
      nn.Dropout(dropout),
      nn.ReLU(),
      nn.Linear(hidden_size, hidden_size),
      nn.Dropout(dropout),
      nn.ReLU(),
      nn.Linear(hidden_size, num_classes),
      nn.Dropout(dropout)
    )
    
  def forward(self, x):
    return self.model(x)


class NormEncoder(nn.Module):
  def __init__(self, hidden_size):
    super().__init__()
    self.a = nn.Linear(hidden_size, hidden_size)
    self.b = nn.Linear(hidden_size, hidden_size)

  def forward(self, hidden_states):
    pooled = hidden_states.mean(dim=1)
    gemma = self.a(pooled).unsqueeze(1)
    beta = self.b(pooled).unsqueeze(1)
    return gemma, beta


class NormExchange(nn.Module):
  def __init__(self, hidden_size, norm_type='layer'):
    super().__init__()
    self.norm_type = norm_type
    if norm_type == 'layer':
      self.norm = nn.LayerNorm(hidden_size)
    elif norm_type == 'batch':
      self.norm = nn.BatchNorm1d(hidden_size)
    else:
      raise ValueError('Unsupported norm_type - \'layer\' or \'batch\'')
    
  def forward(self, x1, x2, gemma1, beta1, gemma2, beta2): # x: (B, L, D), gemma/beta: (B, 1, D)
    if self.norm_type == 'batch':
      # (B, L, D) -> (B, D, L) -> BN -> (B, L, D)
      x1_norm = self.norm(x1.transpose(1, 2)).transpose(1, 2)
      x2_norm = self.norm(x2.transpose(1, 2)).transpose(1, 2)
    else:
      x1_norm = self.norm(x1)
      x2_norm = self.norm(x2)

    x1_out = x1 + gemma2 * x1_norm + beta2
    x2_out = x2 + gemma1 * x2_norm + beta1

    return x1_out, x2_out


class XNormModel(nn.Module):
  def __init__(self, roberta: RobertaModel, hubert: HubertModel, num_classes, hidden_size=768, exchange_layers=[0, 6, 11], weight=0.5):
    super().__init__()
    self.text_enc = roberta
    self.audio_enc = hubert
    self.exchange_layers = exchange_layers
    self.weight = weight

    self.text_emb = roberta.embeddings
    self.audio_feature_extractor = hubert.feature_extractor
    self.audio_feature_projection = hubert.feature_projection
    self.text_layers = roberta.encoder.layer
    self.audio_layers = hubert.encoder.layers

    self.norm_enc_text = nn.ModuleList([NormEncoder(hidden_size=hidden_size) for _ in exchange_layers])
    self.norm_enc_audio = nn.ModuleList([NormEncoder(hidden_size=hidden_size) for _ in exchange_layers])
    self.norm_exchange = NormExchange(hidden_size=hidden_size)

    self.text_classifier = Classifier(input_size=hidden_size, hidden_size=hidden_size, num_classes=num_classes)
    self.audio_classifier = Classifier(input_size=hidden_size, hidden_size=hidden_size, num_classes=num_classes)

  def forward(self, text_inputs, audio_inputs):
    t_hidden = self.text_emb(input_ids=text_inputs['input_ids'])
    a_hidden = self.audio_feature_extractor(audio_inputs['input_values'])
    a_hidden = a_hidden.transpose(1, 2) # (B, D, T) -> (B, T, D)
    a_hidden = self.audio_feature_projection(a_hidden)

    for i in range(len(self.text_layers)):

      t_hidden = self.text_layers[i](t_hidden)[0]
      a_hidden = self.audio_layers[i](a_hidden)[0]

      if i in self.exchange_layers:
        idx =  self.exchange_layers.index(i)

        gamma_t, beta_t = self.norm_enc_text[idx](t_hidden)
        gamma_a, beta_a = self.norm_enc_audio[idx](a_hidden)

        t_hidden, a_hidden = self.norm_exchange(t_hidden, a_hidden, gamma_t, beta_t, gamma_a, beta_a)

    t_pooled = t_hidden[:, 0, :]
    a_pooled = a_hidden[:, 0, :]

    t_logits = self.text_classifier(t_pooled)
    a_logits = self.audio_classifier(a_pooled)

    logits = self.weight * t_logits + (1 - self.weight) * a_logits
    return logits
