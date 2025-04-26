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
      label = torch.tensor(self.label_to_idx[d['major_emotion']], dtype=torch.long)

      return {'text_emb': text_emb, 'audio_emb': audio_emb, 'label': label}
    else:
      audio_array = d['audio']['array']
      text = d['transcription']
      label = self.label_to_idx[d['major_emotion']]
      
      return {'text': text, 'audio_array': audio_array, 'label': label}

class IEMOCAPCaptionDataset(Dataset):
  def __init__(self, ds, caption_mapping, precomputed=False):
      super().__init__()
      self.ds = ds
      self.caption_mapping = caption_mapping  # {audio file name: caption}
      self.precomputed = precomputed
      
  def __len__(self):
      return len(self.ds)
  
  def __getitem__(self, index):
    d = self.ds[index]
    
    if self.precomputed:
      audio_emb = torch.tensor(d['audio_embedding'], dtype=torch.float)  # (768,)
      text_emb = torch.tensor(d['text_embedding'], dtype=torch.float)    # (768,)
      inputs = {'text_emb': text_emb, 'audio_emb': audio_emb}
    else:
      audio_array = d['audio']['array']
      text = d['transcription']
      inputs = {'text': text, 'audio_array': audio_array}
    
    # get caption
    file_name = d['file'].split('/')[-1]  # Just the filename like 'Ses01F_impro01_F000.wav'
    caption = self.caption_mapping[file_name]
    
    return {
      **inputs, 
      'caption': caption
    }