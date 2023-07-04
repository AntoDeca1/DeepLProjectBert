import torch
from transformers import DistilBertTokenizer
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
from PIL import Image
import json

chosen_categories = ["tops", "bottoms", "shoes", "jewellery"]
metadata = json.load(open("polyvore_outfits/polyvore_item_metadata.json"))



class CustomDataset(Dataset):
    OPTIMAL_LENGTH_PERCENTILE = 70

    def __init__(self, path, sets_dict):
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.cnn_preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.path = path
        self.sets_dict = sets_dict
        self.raw_dataset = json.load(open(path))
        self.filtered_dataset = self.map_set_id(self.raw_dataset)
        self.optimal_length = self.compute_optimal_length()

    def from_set_id_to_id(self, list_of_set_id, outfit_set_id):
        """
        Used in map_set_id() function.Starting from a list of setid_position return
        a list of item_id
        :param list_of_set_id:
        :return:
        """
        items = []
        for el in list_of_set_id:
            set_id, el_position = el.split("_")

            real_position = int(el_position) - 1
            real_item_id = self.sets_dict[set_id]["items"][real_position]['item_id']
            description = metadata[real_item_id]["description"] + metadata[real_item_id]['url_name']
            category = metadata[real_item_id]["semantic_category"]
            if outfit_set_id == set_id:
                items.insert(0, {"item": real_item_id, "description": description, "category": category})
            else:
                items.append({"item": real_item_id, "description": description, "category": category})
        return items

    def map_set_id(self, ds):
        """
        Starting from fill_in_the_blank_train
        Iterate through each dict.Taking in account only the outfit with 4 items in the chosen categories
        :return:
        """
        filtered_dataset = []
        for fill_in_the_blank_dict in ds:
            categories = []
            outfit = []
            for el in fill_in_the_blank_dict["question"]:
                set_id, el_position = el.split("_")
                real_position = int(el_position) - 1
                real_item_id = self.sets_dict[set_id]["items"][real_position]['item_id']
                category = metadata[real_item_id]["semantic_category"]
                if category not in chosen_categories:
                    continue
                description = metadata[real_item_id]["description"] + metadata[real_item_id]['url_name']
                outfit.append({"item": real_item_id, "description": description, "category": category})
                categories.append(category)
                if len(outfit) == 4: break
            if all(item in chosen_categories for item in categories) and len(outfit) == 4:
                answers = self.from_set_id_to_id(fill_in_the_blank_dict["answers"], set_id)
                outfit.extend(answers)
                filtered_dataset.append(outfit)
        return filtered_dataset

    def __len__(self):
        return len(self.filtered_dataset)

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

    def __getitem__(self, idx):
        images = []
        descriptions = []
        outfit = self.filtered_dataset[idx]
        for item_dict in outfit:
            item_id = item_dict['item']
            item_description = item_dict['description']
            image_path = self.create_image_path(item_id)
            image = self.get_image(image_path)
            image_tensor = self.cnn_preprocess(image)
            tokenized_description = \
                self.tokenizer(item_description, truncation=True, padding='max_length', max_length=self.optimal_length)[
                    'input_ids']
            images.append(image_tensor)
            descriptions.append(tokenized_description)

        images_tensor = torch.stack(images)
        descriptions_tensor = torch.tensor(descriptions)
        return images_tensor, descriptions_tensor

    def compute_optimal_length(self):
        """
        Iterate through the whole training set and after saving all description lengths
        compute the optimal length
        "Description bigger than optimal length will be truncated instead the smaller ones
        padded
        :return:optimal_length
        """
        lenghts = []
        for outfit in self.filtered_dataset:
            for item in outfit:
                item_description = item['description']
                tokenized_description = self.tokenizer(item_description)['input_ids']
                lenghts.append(len(tokenized_description))
        return int(np.percentile(lenghts, self.OPTIMAL_LENGTH_PERCENTILE))
