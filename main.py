import copy
import torch
from transformers import RobertaModel, RobertaTokenizer, HubertModel, Wav2Vec2FeatureExtractor
from models_other import XNormModel
from utils import get_iemocap_data_loaders, collate_fn_raw, MetricsLogger
from trainer import Trainer

def main():
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print(f'on device: {device}')
  torch.backends.cuda.matmul.allow_tf32 = False
  logger = MetricsLogger(save_path="xnorm_training_metrics.json")

  emotion_labels = ['angry', 'frustrated', 'happy', 'sad', 'neutral']
  # emotion_labels = ['neutral', 'happy', 'sad', 'angry', 'frustrated', 'excited', 'fear', 'disgust', 'surprise', 'other']

  audio_checkpoint = 'facebook/hubert-base-ls960'
  text_checkpoint = 'roberta-base'

  tokenizer = RobertaTokenizer.from_pretrained(text_checkpoint)
  processor = Wav2Vec2FeatureExtractor.from_pretrained(audio_checkpoint)

  train_loader, val_loader, test_loader = get_iemocap_data_loaders(
    path='./iemocap', 
    precomputed=False, 
    batch_size=16, 
    num_workers=0,
    collate_fn=lambda b: collate_fn_raw(b, tokenizer, processor),
    # first_n=100
    )
  
  roberta = RobertaModel.from_pretrained(text_checkpoint)
  hubert = HubertModel.from_pretrained(audio_checkpoint)
  
  # freeze params
  for param in roberta.parameters():
    param.requires_grad = False

  for param in hubert.parameters():
    param.requires_grad = False

  model = XNormModel(roberta=roberta, hubert=hubert, num_classes=len(emotion_labels))
  optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
  trainer = Trainer(model.to(device), optimizer, device=device)
  
  best_val_loss = float('inf')
  best_model_state = None
  patience = 5
  counter = 0
  n_epoch = 20

  for epoch in range(1, n_epoch + 1):
    train_loss, train_acc, train_f1 = trainer.train_one_epoch(train_loader, epoch)
    val_loss, val_acc, val_f1 = trainer.evaluate(val_loader)

    print(f'epoch {epoch}')
    print(f'train loss: {train_loss:.4f}, train acc: {train_acc:.4f}, train f1: {train_f1:.4f}')
    print(f'val loss: {val_loss:.4f}, val acc: {val_acc:.4f}, val f1: {val_f1:.4f}')
    logger.log_train(train_loss, train_acc, train_f1)
    logger.log_val(val_loss, val_acc, val_f1)
    logger.save()

    if val_loss < best_val_loss:
      best_val_loss = val_loss
      best_model_state = copy.deepcopy(trainer.model.state_dict())
      counter = 0
      print('best model updated')
    else:
      counter += 1
    
    if counter >= patience:
      print(f'early stopping triggered')
      break
  
  if best_model_state:
    trainer.model.load_state_dict(best_model_state)
    torch.save({
      'epoch': epoch,
      'model_state_dict': trainer.model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'val_loss': best_val_loss,
      }, 'xnorm_checkpoint.pth')
    print("Best model saved to 'best_xnorm_model.pth'")
  
  test_loss, test_acc, test_f1 = trainer.evaluate(test_loader)
  print(f'test loss: {test_loss:.4f}, test acc: {test_acc:.4f}, test f1: {test_f1:.4f}')
  logger.log_test(test_loss, test_acc, test_f1)
  logger.save()

if __name__ == '__main__':
  main()