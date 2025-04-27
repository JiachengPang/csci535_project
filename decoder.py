import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    x = x.view(x.size(0), self.prefix_len, -1) # (B, prefix_len, llama_dim).

    return x

class MultimodalDecoder(nn.Module):
  def __init__(self, llama_name='TinyLlama/TinyLlama-1.1B-Chat-v1.0'):
    super().__init__()
    self.llama = AutoModelForCausalLM.from_pretrained(llama_name)
    self.tokenizer = AutoTokenizer.from_pretrained(llama_name)

  def forward(self, prefix_emb, input_ids=None, attention_mask=None, labels=None):
    token_emb = self.llama.model.embed_tokens(input_ids) # (B, prompt_len + caption_len, llama_dim)
    full_embeddings = torch.cat([prefix_emb, token_emb], dim=1) # (B, prefix_len + prompt_len + caption_len, llama_dim)
    outputs = self.llama(
      inputs_embeds=full_embeddings,
      attention_mask=attention_mask,
      labels=labels,
      return_dict=True,
    )

    return outputs
  
  def generate(self, prefix_emb, max_new_tokens=20):
    batch_size = prefix_emb.size(0)

    start_tokens = torch.full(
      (batch_size, 1),
      self.tokenizer.bos_token_id,
      dtype=torch.long,
      device=prefix_emb.device,
    )

    # get BOS and append after prefix
    start_emb = self.llama.model.embed_tokens(start_tokens)
    full_embeddings = torch.cat([prefix_emb, start_emb], dim=1)
    attention_mask = torch.ones(full_embeddings.size()[:2], device=prefix_emb.device)

    outputs = self.llama.generate(
      inputs_embeds=full_embeddings,
      attention_mask=attention_mask,
      max_new_tokens=max_new_tokens,
      eos_token_id=self.tokenizer.eos_token_id,
      pad_token_id=self.tokenizer.pad_token_id,
    )

    return outputs
  
if __name__ == '__main__':
  llama_name='TinyLlama/TinyLlama-1.1B-Chat-v1.0'
  llama = AutoModelForCausalLM.from_pretrained(llama_name)
  print(llama)
