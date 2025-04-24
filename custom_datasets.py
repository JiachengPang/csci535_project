import torch
from torch.utils.data import Dataset

class IEMOCAPDataset(Dataset):
  def __init__(self, ds, precomputed):
    super().__init__()
    self.ds = ds
    self.precomputed = precomputed
    emotion_labels = ['angry', 'frustrated', 'happy', 'sad', 'neutral']
    self.label_to_idx = {label: idx for idx, label in enumerate(emotion_labels)}
  
  def __len__(self):
    return len(self.ds)
  
  def __getitem__(self, index):
    d = self.ds[index]

    if self.precomputed:
      audio_emb = torch.tensor(d['audio_embedding'], dtype=torch.float) # (768,)
      text_emb = torch.tensor(d['text_embedding'], dtype=torch.float) # (768,)
      label = torch.tensor(d['label_id'], dtype=torch.long)

      return {'audio_emb': audio_emb, 'text_emb': text_emb, 'label': label}
    else:
      audio_array = d['audio']['array']
      text = d['transcription']
      label = self.label_to_idx[d['major_emotion']]
      
      return {'audio_array': audio_array, 'text': text, 'label': label}
