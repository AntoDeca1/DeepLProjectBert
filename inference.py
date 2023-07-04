import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# TODO:Migliorare
class BertInference:

    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

    def inference(self, idx):
        """
        Function that given an index(an outfit in a dataset) predict the missing item
        :param idx:
        :return:
        """
        to_pil_transform = transforms.ToPILImage()
        query = self.dataset[idx]
        images = query[0]
        descriptions = query[1]
        self.plot_query(query[0])
        images = images.unsqueeze(0)
        descriptions = descriptions.unsqueeze(0)
        final_embedding, positive_emb, negative_embeddings = self.model(images, descriptions)
        positive_emb = positive_emb.unsqueeze(1)  # 1,128--> 1,1,128
        candidates = torch.cat((positive_emb, negative_embeddings), dim=1)  # 1,4,128
        dists = torch.sum((candidates - final_embedding) ** 2, dim=2)
        predicted_index = torch.argmin(dists)
        predicted_tensor = images[0, 3 + predicted_index, :, :]
        plot = plt.imshow(to_pil_transform(predicted_tensor))
        plt.title("Prediction")
        plt.show()

    def plot_query(self, query):
        """
        Function that given a sequence of 8 images 4(query) and 4(candidates)is able to plot them
        :param query:
        :return:
        """
        to_pil_transform = transforms.ToPILImage()
        images_PIL = [to_pil_transform(image) for image in query]
        num_images = len(images_PIL)
        num_rows = 2
        num_cols = (num_images + 1) // num_rows

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 10))
        fig.subplots_adjust(hspace=0.05)

        for i, image_pil in enumerate(images_PIL):
            row = i // num_cols
            col = i % num_cols
            axs[row, col].imshow(image_pil)
            axs[row, col].axis('off')
            if row == 0:
                axs[row, col].set_title("Query", fontsize=12)
            else:
                axs[row, col].set_title("Candidate", fontsize=12)
        plt.show()
