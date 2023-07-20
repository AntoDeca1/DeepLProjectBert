#main.py
from Bert import Bert
from dataset import CustomDataset
from trainer import BertTrainer
import torch
import json
device = torch.device("cpu")
# ------Parameters-------
dim_input = 256  # Input dimension to the final Bert-like model
dim_output = 64  # Dimension of the vector q,k,v in the Attention Heads
batch_size = 32  # Number of element in a batch
attention_heads = 12  # Number of heads in a MultiHeadAttentionLayer
num_encoders = 4  # Number of stacked Encoder
learning_rate = 0.001
epochs = 10
regularization_p = 0.0001  # Parameter that tunes the strength of the regularization
# -----------
# -------------------------Paths-----------------------------
train_path = 'polyvore_outfits/disjoint/train.json'
val_path = 'polyvore_outfits/disjoint/valid.json'
test_path = 'polyvore_outfits/disjoint/test.json'
fill_in_the_blank_train_path = "polyvore_outfits/disjoint/fill_in_blank_train.json"
fill_in_the_blank_val_path = "polyvore_outfits/disjoint/fill_in_blank_valid.json"
# ---------------------------------------------------------------
train = json.load(open(train_path))
train_dict = {el['set_id']: el for el in train}
val = json.load(open(val_path))
val_dict = {el['set_id']: el for el in val}
train_dataset = CustomDataset(path=fill_in_the_blank_train_path, sets_dict=train_dict)
val_dataset = CustomDataset(path=fill_in_the_blank_val_path, sets_dict=val_dict)

model = Bert(dim_input, dim_output, num_encoders, attention_heads).to(device)
trainer = BertTrainer(model, train_dataset, val_dataset, batch_size, regularization_p, learning_rate, epochs,
                      check_point_dir="./checkpoints")
trainer.train()