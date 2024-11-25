
from .base import InteractionLayer

import torch
from torch import nn
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import copy
import math
import collections

torch.manual_seed(1)
np.random.seed(1)



class PocketTransformer(InteractionLayer):
    def __init__(self, input_dim=128, emb_size=128, emb_max_pos_size=545, dropout_rate=0.1, n_layer=2, intermediate_size=512,
                 num_attention_heads=4, attention_probs_dropout=0.1, hidden_dropout_rate=0.1, device="cpu"):
        super(PocketTransformer, self).__init__()
        self.emb = Embeddings(input_dim, emb_size, emb_max_pos_size, dropout_rate)
        self.encoder = Encoder_MultipleLayers(n_layer,
                                              emb_size,
                                              intermediate_size,
                                              num_attention_heads,
                                              attention_probs_dropout,
                                              hidden_dropout_rate)
        self.output_shape = emb_size
        self.norm = nn.LayerNorm(emb_size)
        self.device = device
        self.decoder = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(True),
            
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(True),
            
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(True),
            
            nn.BatchNorm1d(64),
            nn.Linear(64, 1)
        )


    def get_output_shape(self):
        return 1
    
    def forward(self, x, y):
        pocket, pos_index = x

        drug_embed = y[:, :y.shape[1]//2]

        protein_embed = y[:, y.shape[1]//2:]

        pos_drug_protein = torch.zeros(drug_embed.shape[0], 2, device=pocket.device)
        pos_index = torch.cat([pos_drug_protein, pos_index], dim=1)

        stacked_input = torch.cat([drug_embed.unsqueeze(1), protein_embed.unsqueeze(1), pocket], dim=1)

        stacked_input = self.norm(stacked_input)

        attention_mask = (pos_index != -1).float()
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        ex_e_mask = (1.0 - attention_mask) * -10000.0

        emb = self.emb(stacked_input, pos_index)
        encoded_layers = self.encoder(emb.float(), ex_e_mask.float())

        # Residual connection: Add the original input to the output of the transformer encoder
        encoded_layers += stacked_input

        drug_embed = encoded_layers[:, 0, :]
        protein_embed = encoded_layers[:, 1, :]

        concatenated_output = torch.cat([drug_embed, protein_embed], dim=1)
        concatenated_output = self.decoder(concatenated_output)

        return concatenated_output




class LayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta



class Embeddings(nn.Module):
    """Construct the embeddings from pre-embedded protein/target and position embeddings."""

    def __init__(self, vocab_size, hidden_size, max_position_size, dropout_rate):
        super(Embeddings, self).__init__()
        self.position_embeddings = nn.Embedding(32, 128)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, embedded_input_ids, position_ids):
        """
        embedded_input_ids: Tensor of shape [batch, dim1, hidden_size] (already embedded)
        position_ids: Tensor of shape [batch, dim1]
        """
        # Ensure position_ids are long type
        position_ids = position_ids +1
        position_ids = position_ids.long()
        embedded_input_ids = embedded_input_ids.long()
        
        # Generate position embeddings
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = embedded_input_ids + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings




class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        first_line_first_10 = attention_probs[0, 0, 0, :30].cpu().detach().numpy()
        print(first_line_first_10)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class SelfOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob):
        super(SelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Attention, self).__init__()
        self.self = SelfAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.output = SelfOutput(hidden_size, hidden_dropout_prob)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class Intermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(Intermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = F.relu(hidden_states)
        return hidden_states


class Output(nn.Module):
    def __init__(self, intermediate_size, hidden_size, hidden_dropout_prob):
        super(Output, self).__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Encoder(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                 hidden_dropout_prob):
        super(Encoder, self).__init__()
        self.attention = Attention(hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob)
        self.intermediate = Intermediate(hidden_size, intermediate_size)
        self.output = Output(intermediate_size, hidden_size, hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class Encoder_MultipleLayers(nn.Module):
    def __init__(self, n_layer, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                 hidden_dropout_prob):
        super(Encoder_MultipleLayers, self).__init__()
        layer = Encoder(hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                        hidden_dropout_prob)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layer)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            # if output_all_encoded_layers:
            #    all_encoder_layers.append(hidden_states)
        # if not output_all_encoded_layers:
        #    all_encoder_layers.append(hidden_states)
        return hidden_states
