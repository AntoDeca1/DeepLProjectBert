from Bert import Bert
from dataset import CustomDataset
from trainer import BertTrainer
import torch
from torch import nn

# ------Parameters-------
dim_input = 128  # Dimensione degli embedding scelta
dim_output = 64  # Dimensione dei vettori q,k,v nell'Attention Head
batch_size = 3
attention_heads = 4
learning_rate = 0.001
epochs = 5
# -----------
train_path = 'polyvore_outfits/disjoint/train.json'
val_path = 'polyvore_outfits/disjoint/valid.json'
test_path = 'polyvore_outfits/disjoint/test.json'

train_dataset = CustomDataset(path=train_path)
val_dataset = CustomDataset(path=val_path)

model = Bert(dim_input, dim_output, attention_heads)
trainer = BertTrainer(model, train_dataset, val_dataset, batch_size, learning_rate, epochs)

trainer.train()
