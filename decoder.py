import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, AutoTokenizer

class ProjectionLayer(nn.Module):
  def __init__(self, encoder_dim, llama_dim, prefix_len=10):
    super().__init__()
    self.prefix_len = prefix_len
    self.projection = nn.Sequential(
      nn.Linear(encoder_dim, llama_dim),
      nn.Tanh(),
      nn.Linear(llama_dim, llama_dim * prefix_len)
    )
  
  def forward(self, encoding):
    x = self.projection(encoding)
    x = x.view(x.size(0), self.prefix_len, -1)

class MultimodalLLamaDecoder(nn.Module):
  def __init__(self, llama_name='huggyllama/llama-7b', encoder_dim=1536, llama_dim=4096, prefix_len=10):
    super().__init__()
    self.llama = LlamaForCausalLM.from_pretrained(llama_name)
    self.tokenizer = AutoTokenizer.from_pretrained(llama_name)

    self.projection = ProjectionLayer(encoder_dim, llama_dim, prefix_len)

  def forward(self, features, input_ids=None, attention_mask=None, labels=None):
    batch_size = features.size(0)
    prefix_emb = self.projection(features)
    token_emb = self.llama.embed_tokens(input_ids)
    full_embeddings = torch.cat([prefix_emb, token_emb], dim=1)

    if attention_mask is not None:
      prefix_attention = torch.ones((batch_size, self.projection.prefix_len), dtype=attention_mask.dtype, device=attention_mask.device)
      attention_mask = torch.cat([prefix_attention, attention_mask], dim=1)

    # Feed into LLaMA
    outputs = self.llama(
      inputs_embeds=full_embeddings,
      attention_mask=attention_mask,
      labels=labels,
      return_dict=True,
    )

    return outputs
  
  def generate(self, features, max_new_tokens=50):
    batch_size = features.size(0)
    prefix_emb = self.projection(features)

    start_tokens = torch.full(
        (batch_size, 1),
        self.tokenizer.bos_token_id,
        dtype=torch.long,
        device=features.device,
    )

    # get BOS and append after prefix
    start_emb = self.llama.model.embed_tokens(start_tokens)
    full_embeddings = torch.cat([prefix_emb, start_emb], dim=1)
    attention_mask = torch.ones(full_embeddings.size()[:2], device=features.device)

    outputs = self.llama.generate(
        inputs_embeds=full_embeddings,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        eos_token_id=self.tokenizer.eos_token_id,
    )

    return outputs