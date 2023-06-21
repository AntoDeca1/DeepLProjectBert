import json
from PIL import Image
import torch
from torchvision import transforms
import pandas as pd

path_train_json = "./polyvore_outfits/disjoint/train.json"
path_validation_json = "./polyvore_outfits/disjoint/valid.json"
path_test_json = "./polyvore_outfits/disjoint/test.json"
from torch.utils.data import Dataset


class DataSet(Dataset):

    def __init__(self, path):
        self.path = path
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        with open(self.path, 'r') as f:
            self.json_data = json.load(f)
        self.df = self.create_id_dataset()

    def get_image(self, image_path):
        """
        Retrieve an image given a path
        :param image_path:
        :return: image(PIL)
        """
        image = Image.open(image_path)
        return image

    def create_image_path(self, id):
        """
        Create the image path
        :param id: Image Id
        :return: The path where the image is
        """
        return "./polyvore_outfits/images/" + id + ".jpg"

    def __len__(self):
        return len(self.df)

    def create_id_dataset(self):
        """
        Create a dataset containing the ids for each outfit
        :return: a DataFrame
        """
        elements = []
        for item in self.json_data:
            element_lists = item['items']
            elements_id = tuple([el['item_id'] for el in element_lists])
            elements.append(elements_id)
        return pd.DataFrame(elements)

    def __getitem__(self, idx):
        outfit = self.df.iloc[idx]
        outfit_in_tensors = []
        for element in outfit:
            if element is None:
                tensor_image = torch.randn((3, 224, 224))
                outfit_in_tensors.append(tensor_image)
            else:
                image_path = self.create_image_path(element)
                image = self.get_image(image_path)
                tensor_image = self.transform(image)
                outfit_in_tensors.append(tensor_image)
        return tuple(outfit_in_tensors)


ds = DataSet(path=path_test_json)
loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False)

