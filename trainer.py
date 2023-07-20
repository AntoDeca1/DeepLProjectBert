#trainer
from torch.utils.data import DataLoader
import torch
import time
from datetime import datetime
import os

device = torch.device("cpu")

class BertTrainer:

    def __init__(self, model, train_dataset, val_dataset, batch_size, regularization_p,
                 learning_rate=0.001,
                 epochs=5,
                 check_point_dir=None):
        """
        :param model: The whole model to be trained
        :param train_dataset:  Train Dataset
        :param val_dataset:  Validation Dataset
        :param batch_size:  Dimension of the batches
        :param regularization_p: Regularization parameter
        :param learning_rate: Learning Rate
        :param epochs: Number of epochs
        :param check_point_dir: Path to store the checkpoints
        """
        self.check_point_dir = check_point_dir
        self.model = model
        self.regularization_p = regularization_p
        self.current_epoch = 0
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.015)
        self.epochs = epochs
        self.criterion = torch.nn.TripletMarginLoss(margin=1.0, p=2)

    def regularized_triplet_Loss(self, final_embedding, positive_embedding, negative_embedding, regularization_coeff=0.0001):
        """
        Introduce a penalty on the loss based on the norm of the embeddings.So it penalize vector with large norms
        :param final_embedding:
        :param positive_embedding:
        :param negative_embedding:
        :param regularization_coeff:
        :return:
        """
        pure_loss = self.criterion(final_embedding, positive_embedding, negative_embedding)

        regularization_term = regularization_coeff * (
                final_embedding.norm(p=2) + positive_embedding.norm(p=2) + negative_embedding.norm(p=2))

        # Compute total loss
        total_loss = pure_loss + regularization_term

        return total_loss

    def predict_indexes(self, final_embedding, positive_embedding, negative_embeddings):
        """
        A simple utility function.Useful to avoid to repeat the same code in self.train() and self.accuracy()
        :param final_embedding:
        :param positive_embedding:
        :param negative_embeddings:
        :return:
        """
        candidates = torch.cat((positive_embedding.unsqueeze(1), negative_embeddings), dim=1)  # 1,4,128
        dists = torch.sum((candidates - final_embedding.unsqueeze(1)) ** 2, dim=2)
        predicted_indexes = torch.argmin(dists, dim=1)
        return predicted_indexes

    def train(self):
        """
        This function when called starts the training process of the model.For each element in the batch
        the loss is computed as the sum of triplet losses between anchor,positive,negative_i.
        The number of triplets is always 3 since in each record there are 3 negative candidates
        :return:
        """
        train_losses = []
        validation_losses = []
        for self.current_epoch in range(self.current_epoch, self.current_epoch + self.epochs + 1):
            correct_predictions_training = 0.0
            correct_predictions_validation = 0.0
            train_loss = 0.0
            valid_loss = 0.0
            self.model.train()
            for index, (image_batch, description_batch) in enumerate(self.train_loader):
                print(f"Training Batch: {index}")
                image_batch = image_batch.to(device)
                description_batch = description_batch.to(device)
                self.optimizer.zero_grad()
                final_embedding, positive_emb, negative_embeddings = self.model(image_batch, description_batch)
                t_loss = 0
                for negative_embedding in negative_embeddings.permute(1, 0, 2):
                    t_loss += self.regularized_triplet_Loss(final_embedding, positive_emb, negative_embedding,
                                                    self.regularization_p)
                t_loss.backward()
                self.optimizer.step()
                train_loss += t_loss.item()
                predicted_indexes = self.predict_indexes(final_embedding, positive_emb, negative_embeddings)
                correct_predictions_training += (predicted_indexes == 0).sum().item()
            train_loss = train_loss / len(self.train_loader.sampler)
            training_accuracy = correct_predictions_training / len(self.train_loader.sampler)
            train_losses.append(train_loss)
            if self.epochs % 10 == 0:
                self.training_checkpoint(self.current_epoch, train_loss)
            torch.cuda.empty_cache()
            self.model.eval()
            with torch.no_grad():
                for index, (image_batch, description_batch) in enumerate(self.val_loader):
                    print(f"Validation Batch: {index}")
                    image_batch = image_batch.to(device)
                    description_batch = description_batch.to(device)
                    final_embedding, positive_emb, negative_embeddings = self.model(image_batch, description_batch)
                    v_loss = 0
                    for negative_embedding in negative_embeddings.permute(1, 0, 2):
                        v_loss += self.criterion(final_embedding, positive_emb, negative_embedding)
                    valid_loss += v_loss.item()
                    predicted_indexes = self.predict_indexes(final_embedding, positive_emb, negative_embeddings)
                    correct_predictions_validation += (predicted_indexes == 0).sum().item()
                valid_loss = valid_loss / len(self.val_loader.sampler)
                validation_accuracy = correct_predictions_validation / len(self.val_loader.sampler)
                validation_losses.append(valid_loss)

                print(
                    'Epoch: {} \tTraining Loss: {:.6f} \tTraining Loss: {:.6f}\tTraining Accuracy{:.6f}\tValidation Accuracy{:.6f}'.format(
                        self.current_epoch, train_loss, valid_loss, training_accuracy, validation_accuracy))

    def training_checkpoint(self, epoch, loss):
        """
        When called save the state of the model at the given path.
        Is called during the training process
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

        print(f"Model saved as '{name}' for {time.time() - prev:.2f}s")

    def accuracy(self, dataset):
        """
        Compute the accuracy:NumberOfCorrectPredictions/NumberOfPredictions
        :param dataset:
        :return:
        """
        dataloader = DataLoader(dataset, batch_size=48)
        self.model.eval()
        correct_predictions = 0
        with torch.no_grad():
            for index, (image_batch, description_batch) in enumerate(dataloader):
                print(f"Accuracy Batch {index}")
                image_batch = image_batch.to(device)
                description_batch = description_batch.to(device)
                final_embedding, positive_emb, negative_embeddings = self.model(image_batch, description_batch)
                predicted_indexes = self.predict_indexes(final_embedding, positive_emb, negative_embeddings)
                correct_predictions += (predicted_indexes == 0).sum().item()
            return correct_predictions / len(dataloader.sampler)

    def top_2_accuracy(self, dataset):
        """
        The top_2 accuracy consider the prediction correct if is in top_2 closest embeddings.(The ones with the minimum square distance)
        :param dataset:
        :return:
        """
        dataloader = DataLoader(dataset, batch_size=48)
        self.model.eval()
        correct_predictions = 0
        with torch.no_grad():
            for index, (image_batch, description_batch) in enumerate(dataloader):
                print(f"Accuracy Batch {index}")
                image_batch = image_batch.to(device)
                description_batch = description_batch.to(device)
                final_embedding, positive_emb, negative_embeddings = self.model(image_batch, description_batch)
                candidates = torch.cat((positive_emb.unsqueeze(1), negative_embeddings), dim=1)  # 1,4,128
                dists = torch.sum((candidates - final_embedding.unsqueeze(1)) ** 2, dim=2)
                predicted_indexes = torch.topk(-dists, k=2, dim=1).indices
                correct_predictions += sum(
                    [1 if prediction[0].item() == 0 or prediction[1].item() == 0 else 0 for prediction in
                     predicted_indexes])
            return correct_predictions / len(dataloader.sampler)

    def load_checkpoint(self, path):
        """
        Given a path restore a previous checkpoint of a model
        :param path:
        :return:
        """
        print(f"Restoring model {path}")
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        self.current_epoch = checkpoint['epoch']
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Model is restored.")
