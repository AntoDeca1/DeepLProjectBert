import torch
from transformers import DistilBertModel


class BertEmbedder(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.text_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        cnn_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        cnn_layers = cnn_model._modules
        cnn_layers.pop('fc')
        self.cnn_model = torch.nn.Sequential(cnn_layers)
        self.embedding = torch.nn.Linear(1280, 128)

    def forward(self, imgs_batch, descs_batch):
        """
        :param imgs_batch: (batch_size,n_items,3,h,w)
        :param descs_batch:(batch_size,n_descriptions,description_len)
        (n_descriptions=n_items)There is a description for each item
        :return: An embedding sequence that merges description and corresponding image shape:(6,128)
        """
        descs_emb = [self.text_model(encoded_descs).last_hidden_state[:, 1, :] for encoded_descs in descs_batch]
        descs_emb = torch.stack(descs_emb)  # (batch_size,n_items,768)

        old_shape = imgs_batch.shape
        batch_size, n_items = old_shape[:2]
        new_shape = [batch_size * n_items, *old_shape[2:]]
        imgs = imgs_batch.reshape(new_shape)
        imgs_emb = self.cnn_model(imgs)[:, :, 0, 0]  # (batch_size*n_items, 512)
        imgs_emb = imgs_emb.reshape(batch_size, n_items, imgs_emb.shape[-1])

        emb = torch.cat((descs_emb, imgs_emb), dim=-1)

        return self.embedding(emb)
