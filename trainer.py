from Bert import Bert
from dataset import Dataset
from torch import nn
from torch.utils.data import DataLoader
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BertTrainer:

    def __init__(self, model, dataset, batch_size: int = 24,
                 learning_rate=0.001,
                 epochs=5, ):
        self.model = model
        self.dataset = dataset
        self.loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.015)
        self.epochs = epochs
        self.criterion = torch.nn.TripletMarginLoss(margin=1.0, p=2)

    def train(self, epoch):
        for epoch in range(self.epochs):
            for image_batch, description_batch in self.loader:
                self.optimizer.zero_grad()
                final_embedding, positive_emb, negative_embedding = self.model(image_batch, description_batch)
                loss = self.criterion(final_embedding, positive_emb, negative_embedding)
                loss.backward()
                self.optimizer.step()
