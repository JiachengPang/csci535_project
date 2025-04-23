import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from custom_datasets import IEMOCAPDataset

def get_iemocap_data_loaders(path, precomputed, batch_size=16, num_workers=0, seed=42, collate_fn=None, first_n=0):
  ds = load_from_disk(path)['train']
  
  # allow smaller ds
  if first_n > 0:
    ds = ds.select(range(min(first_n, len(ds))))

  # split into 80% train, 10% val, 10% test
  ds = ds.train_test_split(test_size=0.2, seed=seed)
  test_val = ds['test'].train_test_split(test_size=0.5, seed=seed)

  train_ds = IEMOCAPDataset(ds['train'], precomputed=precomputed)
  val_ds = IEMOCAPDataset(test_val['train'], precomputed=precomputed)
  test_ds = IEMOCAPDataset(test_val['test'], precomputed=precomputed)

  print(f'train, val, test sizes: {len(train_ds)}, {len(val_ds)}, {len(test_ds)}')

  train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True, collate_fn=collate_fn)
  val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True, collate_fn=collate_fn)
  test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True, collate_fn=collate_fn)

  return train_loader, val_loader, test_loader


def collate_fn_raw(batch, text_tokenizer, audio_processor, sampling_rate=16000):
  texts = [item['text'] for item in batch]
  audios = [item['audio_array'] for item in batch]
  labels = [item['label'] for item in batch]

  text_inputs = text_tokenizer(texts, padding=True, truncation=True, return_tensors='pt') # keys: 'input_ids', 'attention_mask'
  audio_inputs = audio_processor(audios, sampling_rate=sampling_rate, padding=True, return_tensors='pt') # keys: 'input_values'
  labels = torch.tensor(labels, dtype=torch.long)

  return {'text_inputs': text_inputs, 'audio_inputs': audio_inputs, 'labels': labels}