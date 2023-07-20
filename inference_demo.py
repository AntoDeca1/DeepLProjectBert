import json
from dataset import CustomDataset
from inference import BertInference

from Bert import Bert

# TODO:Inference Toy Demo
test_path = 'polyvore_outfits/disjoint/test.json'
test = json.load(open(test_path))
test_dict = {el['set_id']: el for el in test}
fill_in_the_blank_test_path = "polyvore_outfits/disjoint/fill_in_blank_test.json"
test_dataset = CustomDataset(path=fill_in_the_blank_test_path, sets_dict=test_dict)
model = Bert(128, 64, 4, 4)
inferencer = BertInference(model, test_dataset)
inferencer.inference(2)
