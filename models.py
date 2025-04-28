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

class TextOnlyModel(nn.Module):
    """
    A unimodal model for text classification using RoBERTa.
    Modified to accept both text and audio inputs for trainer compatibility,
    but ignores audio_inputs.
    """
    def __init__(self, roberta: RobertaModel, num_classes, hidden_size=768):
        """
        Initializes the TextOnlyModel.

        Args:
            roberta (RobertaModel): A pre-trained RoBERTa model instance.
            num_classes (int): The number of output classes.
            hidden_size (int): The hidden size of the RoBERTa model.
        """
        super().__init__()
        self.text_enc = roberta
        # Ensure Classifier is defined/imported correctly
        # Replace with your actual Classifier class if different
        self.classifier = Classifier(input_size=hidden_size, hidden_size=hidden_size, num_classes=num_classes)
        print(f"TextOnlyModel initialized (compatible version) with hidden_size={hidden_size}, num_classes={num_classes}")

    # Accept audio_inputs but set a default and ignore it in the method body
    def forward(self, text_inputs, audio_inputs=None):
        """
        Performs the forward pass for text classification.

        Args:
            text_inputs (dict): A dictionary containing RoBERTa inputs
                                {'input_ids': ..., 'attention_mask': ...}.
            audio_inputs (dict, optional): Audio inputs dictionary. Ignored by this model.

        Returns:
            torch.Tensor: The output logits tensor of shape (batch_size, num_classes).
        """
        # --- Only use text_inputs ---
        if text_inputs is None:
             raise ValueError("TextOnlyModel requires 'text_inputs', but received None.")

        # Pass only relevant inputs to the RoBERTa encoder
        outputs = self.text_enc(
            input_ids=text_inputs.get('input_ids'),
            attention_mask=text_inputs.get('attention_mask'),
            return_dict=True # Recommended for easier access to outputs
        )

        # Use the hidden state of the first token ([CLS])
        if not hasattr(outputs, 'last_hidden_state'):
             raise ValueError("RoBERTa output object missing 'last_hidden_state'. Check model configuration or return values.")

        # Extract the [CLS] token's representation
        pooled_output = outputs.last_hidden_state[:, 0, :]

        # Pass the pooled output through the classifier
        logits = self.classifier(pooled_output)
        return logits

class AudioOnlyModel(nn.Module):
    """
    A unimodal model for audio classification using HuBERT.
    Modified to accept both text and audio inputs for trainer compatibility,
    but ignores text_inputs.
    """
    def __init__(self, hubert: HubertModel, num_classes, hidden_size=768):
        """
        Initializes the AudioOnlyModel.

        Args:
            hubert (HubertModel): A pre-trained HuBERT model instance.
            num_classes (int): The number of output classes.
            hidden_size (int): The hidden size of the HuBERT model.
        """
        super().__init__()
        self.audio_enc = hubert
        # Ensure Classifier is defined/imported correctly
        # Replace with your actual Classifier class if different
        self.classifier = Classifier(input_size=hidden_size, hidden_size=hidden_size, num_classes=num_classes)
        print(f"AudioOnlyModel initialized (compatible version) with hidden_size={hidden_size}, num_classes={num_classes}")

    # Accept text_inputs but set a default and ignore it in the method body
    def forward(self, text_inputs=None, audio_inputs=None):
        """
        Performs the forward pass for audio classification.

        Args:
            text_inputs (dict, optional): Text inputs dictionary. Ignored by this model.
            audio_inputs (dict): A dictionary containing HuBERT inputs
                                 {'input_values': ..., 'attention_mask': ...}.

        Returns:
            torch.Tensor: The output logits tensor of shape (batch_size, num_classes).
        """
        # --- Only use audio_inputs ---
        if audio_inputs is None:
             raise ValueError("AudioOnlyModel requires 'audio_inputs', but received None.")

        # Pass only relevant inputs to the HuBERT encoder
        outputs = self.audio_enc(
            input_values=audio_inputs.get('input_values'),
            attention_mask=audio_inputs.get('attention_mask'),
            return_dict=True # Recommended
        )

        # Ensure last_hidden_state exists in the output object
        if not hasattr(outputs, 'last_hidden_state'):
             raise ValueError("HuBERT output object missing 'last_hidden_state'. Check model configuration or return values.")

        # Pool the output features (mean pooling over time)
        pooled_output = outputs.last_hidden_state.mean(dim=1)

        # Pass the pooled output through the classifier
        logits = self.classifier(pooled_output)
        return logits