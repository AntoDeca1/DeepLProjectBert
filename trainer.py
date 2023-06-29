from torch.utils.data import DataLoader
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BertTrainer:

    def __init__(self, model, train_dataset, val_dataset, batch_size,
                 learning_rate=0.001,
                 epochs=5, ):
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

        for epoch in range(self.epochs):
            train_loss = 0.0
            valid_loss = 0.0
            self.model.train()
            for image_batch, description_batch in self.train_loader:
                self.optimizer.zero_grad()
                final_embedding, positive_emb, negative_embedding = self.model(image_batch, description_batch)
                loss = self.criterion(final_embedding, positive_emb, negative_embedding)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            self.model.eval()
            for image_batch, description_batch in self.val_loader:
                final_embedding, positive_emb, negative_embedding = self.model(image_batch, description_batch)
                loss = self.criterion(final_embedding, positive_emb, negative_embedding)
                valid_loss += loss.item()
            train_loss = train_loss / len(self.train_loader.sampler)
            valid_loss = valid_loss / len(self.val_loader.sampler)

            # print training/validation statistics
            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                epoch, train_loss, valid_loss))


