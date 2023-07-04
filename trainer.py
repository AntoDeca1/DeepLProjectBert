from torch.utils.data import DataLoader
import torch
import time
from datetime import datetime
import neptune
import os

device = torch.device("cpu")

run = neptune.init_run(
    project="antoniodecandia01/DeepLBert",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmOTEwODllMS0zMDBmLTQ0NWQtOGNhZi1iYTkyMDUxNGUzMjUifQ==",
)


class BertTrainer:

    def __init__(self, model, train_dataset, val_dataset, batch_size,
                 learning_rate=0.001,
                 epochs=5,
                 check_point_dir=None):
        self.check_point_dir = check_point_dir
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.015)
        self.epochs = epochs
        self.criterion = torch.nn.TripletMarginLoss(margin=1.0, p=2)

    def train(self):
        train_losses = []
        validation_losses = []
        for epoch in range(self.epochs):
            train_loss = 0.0
            valid_loss = 0.0
            self.model.train()
            for index, (image_batch, description_batch) in enumerate(self.train_loader):
                print(f"Batch: {index}")
                image_batch = image_batch.to(device)
                description_batch = description_batch.to(device)
                self.optimizer.zero_grad()
                final_embedding, positive_emb, negative_embeddings = self.model(image_batch, description_batch)
                loss = 0
                for negative_embedding in negative_embeddings.permute(1, 0, 2):
                    loss += self.criterion(final_embedding, positive_emb, negative_embedding)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                run['metrics/train_loss-2'].append(loss.item())
            train_loss = train_loss / len(self.train_loader.sampler)
            train_losses.append(train_loss)

            print('Epoch: {} \tTraining Loss: {:.6f} \t'.format(
                epoch, train_loss))

    def training_checkpoint(self, epoch, loss):
        """
        We should call this function at every epoch
        DA COMPLETARE
        :return:
        """
        prev = time.time()
        if self.check_point_dir is None:
            return
        name = f"bert_epoch{epoch}_{datetime.utcnow().timestamp():.0f}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, os.path.join(self.check_point_dir, name))

        print()
        print(f"Model saved as '{name}' for {time.time() - prev:.2f}s")
        print()
