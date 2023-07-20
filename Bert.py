# Bert.py
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
    After DISTILBERT and RESNET we added a sequence a FC layer and a normalization
    like suggested in the paper linked in the README.md
    """

    def __init__(self, dim_inp, dropout=0.1, ):
        """
        :param dropout: Dropout probability
        #TextEmbedder:DistilBert not fine-tuned during training
        #ImagesEmbedder:ResNet layers before the classification one.Fine-Tuned during
        training
        """
        super().__init__()
        self.text_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        # To not fine_tune DistilBert
        for param in self.text_model.parameters():
            param.requires_grad = False
        cnn_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        cnn_layers = cnn_model._modules
        cnn_layers.pop('fc')
        self.cnn_model = nn.Sequential(cnn_layers)
        self.feed_forward_distillBert = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(1024, 256),
            nn.Dropout(dropout)
        )
        self.feed_forward_resNet = nn.Sequential(
            nn.Linear(512, 1024),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(1024, 256),
            nn.Dropout(dropout)
        )
        self.embedding = nn.Linear(512, dim_inp)

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

        descs_emb = self.feed_forward_distillBert(descs_emb)
        imgs_emb = self.feed_forward_resNet(imgs_emb)
        normalized_imgs_embed = self.normalize_hypersphere(imgs_emb)
        normalized_descs_emb = self.normalize_hypersphere(descs_emb)

        emb = torch.cat((normalized_imgs_embed, normalized_descs_emb), dim=-1)

        return self.embedding(emb)

    def normalize_hypersphere(self, vec):
        """
        This function is used to normalize each vector in the batch
        This type of normalization force the norm of the vector to be unitary
        :param vec:
        :return:
        """
        batch_size, num_elements, embedding_dim = vec.shape
        reshaped_vec = vec.view(batch_size * num_elements, embedding_dim)
        norms = torch.norm(reshaped_vec, p=2, dim=-1, keepdim=True)

        normalized_tensor = reshaped_vec / norms
        normalized_tensor = normalized_tensor.view(vec.shape)

        return normalized_tensor


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

        scale = math.sqrt(query.size(2))  # Scaling:sqrt(64)
        scores = torch.bmm(query, key.transpose(1, 2)) / scale

        attn = self.softmax(scores)
        contextualized_embeddings = torch.bmm(attn, value)

        return contextualized_embeddings


class MultiHeadAttention(nn.Module):
    """
    Multihead attention computed as:
    1)Each head is computed separately
    2)The heads resulting vector are concatenated
    3)The dimensionality is taken to the input_dim through a FC layer
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
        return self.norm(input_tensor + scores)


class Encoder(nn.Module):
    """
    This is the class for only one Encoder Layer
    1)MultiHeadAttention
    2)FeedForward
    3)Normalization(We should include the residual connections)
    """

    def __init__(self, dim_input, dim_output, attention_heads=4, dropout=0.1):
        super(Encoder, self).__init__()
        self.attention = MultiHeadAttention(attention_heads, dim_input, dim_output)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim_input, dim_input * 4),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(dim_input * 4, dim_input),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(dim_input)

    def forward(self, input_tensor):
        """
        Encoder Composed by
        1) A MultiHeadAttention
        2) FeedForwardNN
        3) LayerNorm(input_tensor+ output)
        :param input_tensor:
        :return:
        """
        context = self.attention(input_tensor)
        res = self.feed_forward(context)
        return self.norm(res + input_tensor)


class Bert(nn.Module):
    def __init__(self, dim_inp, dim_out, num_encoders=4, attention_heads=4, dropout=0.1):
        """
        Whole architecture class
        :param dim_inp: Input dimension to the final Bert Model
        :param dim_out: Output dimension
        :param num_encoders: Num of encoders layers
        :param attention_heads: Num of heads in each MultiHeadAttention
        :param dropout: DropOut probability
        """
        super().__init__()
        self.attention_heads = attention_heads
        self.num_encoders = num_encoders
        self.embedding = BertEmbedder(dim_inp)
        self.encoders = nn.ModuleList([
            Encoder(dim_inp, dim_out, attention_heads, dropout=dropout) for _ in range(num_encoders)
        ])

    def forward(self, imgs, descs):
        """
        :param imgs:Batch of sequence of images
        :param descs:Batch of sequence of description
        :return: predicted_embedding,positive_pair,negative_pairs(**)
        **:N.B:Pairs since they are 3 in our specific case
        """
        embeddings = self.embedding(imgs, descs)
        input_embeddings = embeddings[:, :4, :]
        postive_pair = embeddings[:, 4, :]
        negative_pairs = embeddings[:, 5:, :]
        for layer in self.encoders:
            input_embeddings = layer(input_embeddings)
        final_embedding = input_embeddings.mean(-2)
        return final_embedding, postive_pair, negative_pairs
