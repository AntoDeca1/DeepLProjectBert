import torch
from transformers import DistilBertModel
from torch import nn
import math


class BertEmbedder(nn.Module):
    """
    The Embedder is composed by:
    - DISTILBERT (Pretrained-by-HuggingFace):Used to encode textual rapresentations
    - RESNET18(Pretrained-by-torchHub):Used to encode visual rappresentations
    Image and text embeddings are combined to achieve an overall representation
    """

    def __init__(self):
        super().__init__()
        self.text_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        cnn_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        cnn_layers = cnn_model._modules
        cnn_layers.pop('fc')
        self.cnn_model = nn.Sequential(cnn_layers)
        self.embedding = nn.Linear(1280, 128)

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


class AttentionHead(nn.Module):
    """
    Single Attention Head:
    1)Compute Q,K,V
    2)Matrix Multiply Q*K
    3)Apply softmax to obtain Scores
    4)Multiply the scores for V
    5)Sum up the V to obtain the new embeddings
    """

    def __init__(self, dim_inp, dim_out):
        super(AttentionHead, self).__init__()

        self.dim_inp = dim_inp

        self.w_q = nn.Linear(dim_inp, dim_out)
        self.w_k = nn.Linear(dim_inp, dim_out)
        self.w_v = nn.Linear(dim_inp, dim_out)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        3:batch_size
        6:Num of embeddings in input
        64:Embedding Dimension
        query:shape(3,6,64): q_vectors per ogni embedding
        key:shape(3,6,64) :k_vectors per ogni embedding
        value:shape(3,6,64):v_vectors per ogni embedding
        :param input_tensor:
        :return:
        """
        query, key, value = self.w_q(x), self.w_k(x), self.w_v(x)

        scale = math.sqrt(query.size(1))  # Scaling:sqrt(64)
        scores = torch.bmm(query, key.transpose(1, 2)) / scale

        attn = self.softmax(scores)
        contextualized_embeddings = torch.bmm(attn, value)

        return contextualized_embeddings


class MultiHeadAttention(nn.Module):
    """
    The multiHead attention implemented in the classical way
    """

    def __init__(self, num_heads, dim_inp, dim_out):
        super(MultiHeadAttention, self).__init__()

        self.heads = nn.ModuleList([
            AttentionHead(dim_inp, dim_out) for _ in range(num_heads)
        ])
        self.linear = nn.Linear(dim_out * num_heads, dim_inp)
        self.norm = nn.LayerNorm(dim_inp)

    def forward(self, input_tensor):
        s = [head(input_tensor) for head in self.heads]
        scores = torch.cat(s, dim=-1)
        scores = self.linear(scores)
        return self.norm(scores)


class Encoder(nn.Module):
    """
    This is the class for only one Decoder Layer
    1)MultiHeadAttention
    2)FeedForward
    3)Normalization(We should include the residual connections)
    """

    def __init__(self, dim_input, dim_output, attention_heads=4, dropout=0.1):
        super(Encoder, self).__init__()
        self.attention = MultiHeadAttention(attention_heads, dim_input, dim_output)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim_input, dim_output),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(dim_output, dim_input),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(dim_input)

    def forward(self, input_tensor):
        """
        Encoder Composed by
        1) A MultiHeadAttention
        2) FeedForwardNN
        3) LayerNorm(ResidualConnection+ Output)
        :param input_tensor:
        :return:
        """
        context = self.attention(input_tensor)
        res = self.feed_forward(context)
        return self.norm(res + input_tensor)


class Bert(nn.Module):
    def __init__(self, dim_inp, dim_out, attention_heads=4):
        super().__init__()
        self.embedding = BertEmbedder()
        self.encoder = Encoder(dim_inp, dim_out, attention_heads, dropout=0.1)
        self.final_embedding_layer = nn.Linear(dim_inp * 6, dim_inp)

    def forward(self, imgs, descs):
        embeddings = self.embedding(imgs, descs)
        encoded = self.encoder(embeddings)
        batch_size, num_elements, embedding_dim = encoded.shape
        encoded_reshaped = encoded.reshape(batch_size, num_elements * embedding_dim)
        return self.final_embedding_layer(encoded_reshaped)

# TODO:Generalizzare il codice(parametrizzare)
