
import copy
import math

from overrides import overrides
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

from allennlp.common.params import Params
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder


@Seq2SeqEncoder.register("transformer")
class TransformerModel(Seq2SeqEncoder):

    def __init__(self,
                 num_layers=8,
                 input_size=200, 
                 hidden_size=800, 
                 heads=8, 
                 dropout=0.1):
        super(TransformerModel, self).__init__()

        self._input_dim = input_size
        self._output_dim = input_size
        self.transformer = make_model(num_layers, input_size, hidden_size, heads, dropout)

    @overrides
    def forward(self, x, mask):
        timesteps = mask.size(-1)
        mask = mask.unsqueeze(-1).expand([-1, -1, timesteps])
        return self.transformer(x, mask)
    
    @overrides
    def get_input_dim(self) -> int:
        return self._input_dim

    @overrides
    def get_output_dim(self) -> int:
        return self._output_dim

    @overrides
    def is_bidirectional(self):
        return False

    @classmethod
    def from_params(cls, params: Params):
        num_layers = params.pop_int("num_layers", 8)
        input_size = params.pop_int("input_size", 200)
        hidden_size = params.pop_int("hidden_size", 800)
        heads = params.pop_int("heads", 8)
        dropout = params.pop_float("dropout", 0.1)
        return cls(num_layers=num_layers,
                           input_size=input_size,
                           hidden_size=hidden_size,
                           heads=heads,
                           dropout=dropout)


def make_model(num_layers=6, 
               input_size=512, # Attention size
               hidden_size=2048, # FF layer size
               heads=8,
               dropout=0.1,
               return_all_layers=False):
    "Helper: Construct a model from hyperparameters."
    attn = MultiHeadedAttention(heads, input_size)
    ff = PositionwiseFeedForward(input_size, hidden_size, dropout)
    model = TransformerEncoder(EncoderLayer(input_size, attn, ff, dropout),
                               num_layers, 
                               return_all_layers=return_all_layers)

    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

def subsequent_mask(size, device=-1):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask = torch.from_numpy(subsequent_mask) == 0
    if device >=0:
        return subsequent_mask.cuda(device)
    return subsequent_mask

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, input_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Compute the positional encodings once in log space.
        positional_encoding = torch.zeros(max_len, input_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, input_dim, 2) *
                             -(math.log(10000.0) / input_dim))
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        positional_encoding = positional_encoding.unsqueeze(0)
        self.register_buffer('positional_encoding', positional_encoding)
        
    def forward(self, x):
        return x + Variable(self.positional_encoding[:, :x.size(1)],
                            requires_grad=False)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, input_dim, ff_dim, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(input_dim, ff_dim)
        self.w_2 = nn.Linear(ff_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class TransformerEncoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, num_layers, return_all_layers=False):
        super(TransformerEncoder, self).__init__()
        self.layers = clones(layer, num_layers)
        self.norm = LayerNorm(layer.size)
        self.return_all_layers = return_all_layers
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        all_layers = []
        for layer in self.layers:
            x = layer(x, mask)
            if self.return_all_layers:
                all_layers.append(x)

        if self.return_all_layers:
            all_layers[-1] = self.norm(all_layers[-1])
            return all_layers
        return self.norm(x)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class MultiHeadedAttention(nn.Module):
    def __init__(self, heads, input_dim, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert input_dim % heads == 0
        # We assume d_v always equals d_k
        self.d_k = input_dim // heads
        self.heads = heads
        self.linears = clones(nn.Linear(input_dim, input_dim), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            # Shape (batch_size, num_heads, timesteps, timesteps)
            mask = mask.unsqueeze(1).expand([-1, self.heads, -1, -1])

        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [layer(x).view(nbatches, -1, self.heads, self.d_k).transpose(1, 2)
             for layer, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.heads * self.d_k)
        return self.linears[-1](x)