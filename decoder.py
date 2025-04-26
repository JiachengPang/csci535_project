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
  def __init__(self, llama_name='meta-llama/Llama-2-7b-chat-hf', encoder_dim=1536, llama_dim=4096, prefix_len=10):
    super().__init__()
    self.llama = LlamaForCausalLM.from_pretrained(llama_name)
    self.tokenizer = AutoTokenizer.from_pretrained(llama_name)

    self.projection = ProjectionLayer(encoder_dim, llama_dim, prefix_len)

  def forward(self, features, input_ids=None, attention_mask=None, labels=None):
    batch_size = features.size(0)
    
    prefix_emb = self.projection(features)

    token_emb = self.llama

if __name__ == '__main__':
  llama_name="huggyllama/llama-7b"
  llama = LlamaForCausalLM.from_pretrained(llama_name)
  print(llama)