import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

#inference.py
class BertInference:
    def __init__(self, model, dataset):
        """
        :param model: Model used for inference
        :param dataset: Dataset from witch we select the queries
        """
        self.model = model
        self.dataset = dataset
        self.to_pil_transform = transforms.ToPILImage()

    def inference(self, idx):
        """
        Takes an index and select the record at that given idx in the dataset.
        The record is taken as query and passed to the model that return the prediction.
        In this function we also plot the query,the candidates and at the end the prediction
        :param idx:
        :return:
        """
        categories = [item['category'] for item in self.dataset.filtered_dataset[idx]]
        query = self.dataset[idx]
        images = query[0]
        descriptions = query[1]
        self.plot_query(query[0], categories)
        images = images.unsqueeze(0)
        descriptions = descriptions.unsqueeze(0)
        final_embedding, positive_emb, negative_embeddings = self.model(images, descriptions)
        positive_emb = positive_emb.unsqueeze(1)  # 1,128--> 1,1,128
        candidates = torch.cat((positive_emb, negative_embeddings), dim=1)  # 1,4,128
        dists = torch.sum((candidates - final_embedding) ** 2, dim=2)
        predicted_index = torch.argmin(dists)
        predicted_tensor = images[0, 4 + predicted_index, :, :]
        fig, axs = plt.subplots(2, figsize=(10, 10))
        axs[0].imshow(self.to_pil_transform(predicted_tensor))
        axs[1].imshow(self.to_pil_transform(images[0, 4, :, :]))
        axs[0].set_title("Predicted Candidate")
        axs[1].set_title("True Candidate")
        plt.show()

    def plot_query(self, query, categories):
        """
        Function that given a sequence of 8 images 4(query) and 4(candidates)is able to plot them
        :param query:
        :return:
        """
        images_PIL = [self.to_pil_transform(image) for image in query]
        num_images = len(images_PIL)
        num_rows = 2
        num_cols = (num_images + 1) // num_rows

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 10))
        fig.subplots_adjust(hspace=0.05)
        fig.suptitle("Query and Candidates")

        for i, image_pil in enumerate(images_PIL):
            row = i // num_cols
            col = i % num_cols
            axs[row, col].imshow(image_pil)
            axs[row, col].axis('off')
            if row == 0:
                axs[row, col].set_title(categories[i], fontsize=12)
            else:
                axs[row, col].set_title(categories[i], fontsize=12)
        plt.show()