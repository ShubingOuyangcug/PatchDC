import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils import weight_norm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score,fbeta_score
from sklearn.model_selection import StratifiedKFold
import random
import datetime
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import optuna
import json
import shutil
from pytorch_lightning.callbacks import TQDMProgressBar
from copy import deepcopy
from pytorch_lightning.loggers import TensorBoardLogger
# torch.set_float32_matmul_precision('medium')

# 配置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed =42
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


# 1. 创建文件名-标签映射表
def create_file_label_mapping(txt_path, base_path):
    """创建（文件路径，标签）的元组列表"""
    with open(txt_path, 'r') as f:
        files = [os.path.join(base_path, line.strip()) for line in f]

    mapping = []
    for file_path in files:
        try:
            filename = os.path.basename(file_path)
            label_str = filename.split('_')[-1].split('.')[0]
            label = 0 if label_str == '0' else 1
            mapping.append((file_path, label))
        except:
            print(f"Invalid filename format: {file_path}")
    return mapping


# 2. 自定义数据集类
class CSVDataset(Dataset):

    def __init__(self, file_label_mapping):
        self.mapping = file_label_mapping
        self.dtypes = {
            "data_mm_jugy": np.float32,
            
            "data_mm_iqr": np.float32,
            "data_mm_range": np.float32,
            "1H_sum_mima": np.float32,
            "1H_sum_max": np.float32,
            "1H_wmedian_jugy": np.float32,
            
            "1H_wmedian_mask": np.float32,
            "1H_wmedian_range": np.float32,
            "1H_wmedian_iqr": np.float32,
           
        }

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        file_path, label = self.mapping[idx]
        try:
            df = pd.read_csv(
                file_path,
                usecols=list(self.dtypes.keys()),
                dtype=self.dtypes,
                engine='c'  # 更快的C引擎
            )

            slide_gy_data = []
            for slide_gy_i in ["data_mm_jugy", "data_mm_iqr", "data_mm_range"]:
                slide_gy_gener = torch.FloatTensor(df[slide_gy_i].head(100).values)
                slide_gy_data.append(slide_gy_gener)
            slide_gy_gener = torch.stack(slide_gy_data, dim=0)

            rain_gy_data = []
            for rain_gy_i in ["1H_sum_mima",  "1H_wmedian_jugy", "1H_wmedian_mask", "1H_sum_max","1H_wmedian_range", "1H_wmedian_iqr"]:
                rain_gy_gener = torch.FloatTensor(df[rain_gy_i].head(24 * 15).values)
                rain_gy_gener = torch.flip(rain_gy_gener, dims=[0])
                rain_gy_data.append(rain_gy_gener)
            rain_gy_gener = torch.stack(rain_gy_data, dim=0)

            return {
                "data": rain_gy_gener,
                "data100": slide_gy_gener,
                'label': torch.tensor(label, dtype=torch.long)
            }
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None

    keys = batch[0].keys()
    result = {}
    for key in keys:
        result[key] = torch.stack([item[key] for item in batch], dim=0)

    labels = [item['label'] for item in batch]
    labels = torch.stack(labels, dim=0)

    return {
        'data': result['data'],
        'data100': result['data100'],
        'labels': labels
    }


import math
import numpy as np
from typing import Optional  # , Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def PositionalEncoding(q_len, hidden_size, normalize=True):
    pe = torch.zeros(q_len, hidden_size)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, hidden_size, 2) * -(math.log(10000.0) / hidden_size)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe


SinCosPosEncoding = PositionalEncoding


def Coord2dPosEncoding(q_len, hidden_size, exponential=False, normalize=True, eps=1e-3):
    x = 0.5 if exponential else 1
    i = 0
    for i in range(100):
        cpe = (
                2
                * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** x)
                * (torch.linspace(0, 1, hidden_size).reshape(1, -1) ** x)
                - 1
        )
        if abs(cpe.mean()) <= eps:
            break
        elif cpe.mean() > eps:
            x += 0.001
        else:
            x -= 0.001
        i += 1
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe


def Coord1dPosEncoding(q_len, exponential=False, normalize=True):
    cpe = (
            2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** (0.5 if exponential else 1))
            - 1
    )
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe


def positional_encoding(pe, learn_pe, q_len, hidden_size):
    # Positional encoding
    if pe == None:
        W_pos = torch.empty(
            (q_len, hidden_size)
        )  # pe = None and learn_pe = False can be used to measure impact of pe
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == "zero":
        W_pos = torch.empty((q_len, 1))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == "zeros":
        W_pos = torch.empty((q_len, hidden_size))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == "normal" or pe == "gauss":
        W_pos = torch.zeros((q_len, 1))
        torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
    elif pe == "uniform":
        W_pos = torch.zeros((q_len, 1))
        nn.init.uniform_(W_pos, a=0.0, b=0.1)
    elif pe == "lin1d":
        W_pos = Coord1dPosEncoding(q_len, exponential=False, normalize=True)
    elif pe == "exp1d":
        W_pos = Coord1dPosEncoding(q_len, exponential=True, normalize=True)
    elif pe == "lin2d":
        W_pos = Coord2dPosEncoding(
            q_len, hidden_size, exponential=False, normalize=True
        )
    elif pe == "exp2d":
        W_pos = Coord2dPosEncoding(q_len, hidden_size, exponential=True, normalize=True)
    elif pe == "sincos":
        W_pos = PositionalEncoding(q_len, hidden_size, normalize=True)
    else:
        raise ValueError(
            f"{pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
        'zeros', 'zero', uniform', 'lin1d', 'exp1d', 'lin2d', 'exp2d', 'sincos', None.)"
        )
    return nn.Parameter(W_pos, requires_grad=learn_pe)


class Flatten_Head(nn.Module):
    """
    Flatten_Head
    """

    def __init__(self, individual, n_vars, nf, h, c_out, head_dropout=0):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars
        self.c_out = c_out

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, h * c_out))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, h * c_out)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x hidden_size x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])  # z: [bs x hidden_size * patch_num]
                z = self.linears[i](z)  # z: [bs x h]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)  # x: [bs x nvars x h]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x


class _ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)
    """

    def __init__(
            self, hidden_size, n_heads, attn_dropout=0.0, res_attention=False, lsa=False, causal=True
    ):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = hidden_size // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa
        self.causal = causal
    def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            prev: Optional[torch.Tensor] = None,
            key_padding_mask: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        if self.causal:
            seq_len = q.size(2)
            # 创建下三角掩码（允许关注当前位置及之前位置）
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
            if attn_mask is None:
                attn_mask = causal_mask
        """
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        """
        if not self.res_attention:
            # Use torch's built-in flash attention for efficient computation
            # Note: This will not return attention weights/scores
            # The shapes of q, k, v must be: [batch, n_heads, seq_len, head_dim]
            # Reshape q, k, v into [batch*n_heads, seq_len, head_dim] as required by torch.nn.functional.scaled_dot_product_attention
            bs, n_heads, seq_len, head_dim = q.shape
            q_ = q.reshape(bs * n_heads, seq_len, head_dim)
            k_ = k.permute(0, 1, 3, 2).reshape(bs * n_heads, seq_len, head_dim)
            v_ = v.reshape(bs * n_heads, seq_len, head_dim)
            # If attn_mask exists, convert it to the appropriate format for flash attention (e.g. [batch*n_heads, seq_len, seq_len])
            if attn_mask is not None:
                attn_mask = attn_mask.repeat(bs * n_heads, 1, 1)
            output = F.scaled_dot_product_attention(
                q_,
                k_,
                v_,
                attn_mask=attn_mask,
                dropout_p=self.attn_dropout.p,
                is_causal=False,
            )
            # Restore the original shape
            output = output.reshape(bs, n_heads, seq_len, head_dim)
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            return output, None
        else:
            # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
            attn_scores = (
                    torch.matmul(q, k) * self.scale
            )  # attn_scores : [bs x n_heads x max_q_len x q_len]

            # Add pre-softmax attention scores from the previous layer (optional)
            if prev is not None:
                attn_scores = attn_scores + prev

            # Attention mask (optional)
            if (
                    attn_mask is not None
            ):  # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
                if attn_mask.dtype == torch.bool:
                    attn_scores.masked_fill_(attn_mask, -np.inf)
                else:
                    attn_scores += attn_mask

            # Key padding mask (optional)
            if (
                    key_padding_mask is not None
            ):  # mask with shape [bs x q_len] (only when max_w_len == q_len)
                attn_scores.masked_fill_(
                    key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf
                )

            # normalize the attention weights
            attn_weights = F.softmax(
                attn_scores, dim=-1
            )  # attn_weights   : [bs x n_heads x max_q_len x q_len]
            attn_weights = self.attn_dropout(attn_weights)

            # compute the new values given the attention weights
            output = torch.matmul(
                attn_weights, v
            )  # output: [bs x n_heads x max_q_len x d_v]

            return output, attn_weights, attn_scores


class _MultiheadAttention(nn.Module):
    """
    _MultiheadAttention
    """

    def __init__(
            self,
            hidden_size,
            n_heads,
            d_k=None,
            d_v=None,
            res_attention=False,
            attn_dropout=0.0,
            proj_dropout=0.0,
            qkv_bias=True,
            lsa=False,
    ):
        """
        Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x hidden_size]
            K, V:    [batch_size (bs) x q_len x hidden_size]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = hidden_size // n_heads if d_k is None else d_k
        d_v = hidden_size // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(hidden_size, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(hidden_size, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(hidden_size, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(
            hidden_size,
            n_heads,
            attn_dropout=attn_dropout,
            res_attention=self.res_attention,
            lsa=lsa,
        )

        # Poject output
        self.to_out = nn.Sequential(
            nn.Linear(n_heads * d_v, hidden_size), nn.Dropout(proj_dropout)
        )

    def forward(
            self,
            Q: torch.Tensor,
            K: Optional[torch.Tensor] = None,
            V: Optional[torch.Tensor] = None,
            prev: Optional[torch.Tensor] = None,
            key_padding_mask: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):

        bs = Q.size(0)
        if K is None:
            K = Q
        if V is None:
            V = Q

        # Linear (+ split in multiple heads)
        q_s = (
            self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        )  # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = (
            self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3, 1)
        )  # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = (
            self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)
        )  # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(
                q_s,
                k_s,
                v_s,
                prev=prev,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )
        else:
            output, attn_weights = self.sdp_attn(
                q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask
            )
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = (
            output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v)
        )  # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights


class Transpose(nn.Module):
    """
    Transpose
    """

    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)


class TSTEncoder(nn.Module):
    """
    TSTEncoder
    """

    def __init__(
            self,
            q_len,
            hidden_size,
            n_heads,
            d_k=None,
            d_v=None,
            linear_hidden_size=None,
            norm="BatchNorm",
            attn_dropout=0.0,
            dropout=0.0,
            activation="gelu",
            res_attention=False,
            n_layers=1,
            pre_norm=False,
            store_attn=False,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                TSTEncoderLayer(
                    q_len,
                    hidden_size,
                    n_heads=n_heads,
                    d_k=d_k,
                    d_v=d_v,
                    linear_hidden_size=linear_hidden_size,
                    norm=norm,
                    attn_dropout=attn_dropout,
                    dropout=dropout,
                    activation=activation,
                    res_attention=res_attention,
                    pre_norm=pre_norm,
                    store_attn=store_attn,
                )
                for i in range(n_layers)
            ]
        )
        self.res_attention = res_attention

    def forward(
            self,
            src: torch.Tensor,
            key_padding_mask: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        output = src
        scores = None
        if self.res_attention: #false
            for mod in self.layers:
                output, scores = mod(
                    output,
                    prev=scores,
                    key_padding_mask=key_padding_mask,
                    attn_mask=attn_mask,
                )
            return output
        else:
            for mod in self.layers:
                output = mod(
                    output, key_padding_mask=key_padding_mask, attn_mask=attn_mask
                )
            return output


def get_activation_fn(activation):
    if callable(activation):
        return activation()
    elif activation.lower() == "relu":
        return nn.ReLU()
    elif activation.lower() == "gelu":
        return nn.GELU()
    raise ValueError(
        f'{activation} is not available. You can use "relu", "gelu", or a callable'
    )


class TSTEncoderLayer(nn.Module):
    """
    TSTEncoderLayer
    """

    def __init__(
            self,
            q_len,
            hidden_size,
            n_heads,
            d_k=None,
            d_v=None,
            linear_hidden_size=256,
            store_attn=False,
            norm="BatchNorm",
            attn_dropout=0,
            dropout=0.0,
            bias=True,
            activation="gelu",
            res_attention=False,
            pre_norm=False,
    ):
        super().__init__()
        assert (
            not hidden_size % n_heads
        ), f"hidden_size ({hidden_size}) must be divisible by n_heads ({n_heads})"
        d_k = hidden_size // n_heads if d_k is None else d_k
        d_v = hidden_size // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(
            hidden_size,
            n_heads,
            d_k,
            d_v,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
            res_attention=res_attention,
        )

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(hidden_size), Transpose(1, 2)
            )
        else:
            self.norm_attn = nn.LayerNorm(hidden_size)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, linear_hidden_size, bias=bias),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(linear_hidden_size, hidden_size, bias=bias),
        )

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(hidden_size), Transpose(1, 2)
            )
        else:
            self.norm_ffn = nn.LayerNorm(hidden_size)

        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def forward(
            self,
            src: torch.Tensor,
            prev: Optional[torch.Tensor] = None,
            key_padding_mask: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):  # -> Tuple[torch.Tensor, Any]:

        # Multi-Head attention sublayer
        if self.pre_norm:#False
            src = self.norm_attn(src)
        ## Multi-Head attention
        if self.res_attention: #False
            src2, attn, scores = self.self_attn(
                src,
                src,
                src,
                prev,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )
        else:
            src2, attn = self.self_attn(
                src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask
            )
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src = src + self.dropout_attn(
            src2
        )  # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(
            src2
        )  # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:#False
            return src, scores
        else:
            return src


class TSTiEncoder(nn.Module):  # i means channel-independent
    """
    TSTiEncoder
    """

    def __init__(
            self,
            c_in,
            patch_num,
            patch_len,
            max_seq_len=1024,
            n_layers=3,
            hidden_size=128,
            n_heads=16,
            d_k=None,
            d_v=None,
            linear_hidden_size=256,
            norm="BatchNorm",
            attn_dropout=0.0,
            dropout=0.0,
            act="gelu",
            store_attn=False,
            key_padding_mask="auto",
            padding_var=None,
            attn_mask=None,
            res_attention=True,
            pre_norm=False,
            pe="zeros",
            learn_pe=True,
    ):
        super().__init__()

        self.patch_num = patch_num
        self.patch_len = patch_len

        # Input encoding
        q_len = patch_num
        self.W_P = nn.Linear(
            patch_len, hidden_size
        )  # Eq 1: projection of feature vectors onto a d-dim vector space
        self.seq_len = q_len

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, q_len, hidden_size)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(
            q_len,
            hidden_size,
            n_heads,
            d_k=d_k,
            d_v=d_v,
            linear_hidden_size=linear_hidden_size,
            norm=norm,
            attn_dropout=attn_dropout,
            dropout=dropout,
            pre_norm=pre_norm,
            activation=act,
            res_attention=res_attention,
            n_layers=n_layers,
            store_attn=store_attn,
        )

    def forward(self, x) -> torch.Tensor:  # x: [bs x nvars x patch_len x patch_num]

        n_vars = x.shape[1]
        # Input encoding
        x = x.permute(0, 1, 3, 2)  # x: [bs x nvars x patch_num x patch_len]
        x = self.W_P(x)  # x: [bs x nvars x patch_num x hidden_size]

        u = torch.reshape(
            x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        )  # u: [bs * nvars x patch_num x hidden_size]
        u = self.dropout(u + self.W_pos)  # u: [bs * nvars x patch_num x hidden_size]

        # Encoder
        z = self.encoder(u)  # z: [bs * nvars x patch_num x hidden_size]
        z = torch.reshape(
            z, (-1, n_vars, z.shape[-2], z.shape[-1])
        )  # z: [bs x nvars x patch_num x hidden_size]
        z = z.permute(0, 1, 3, 2)  # z: [bs x nvars x hidden_size x patch_num]

        return z


class RevIN(nn.Module):
    """RevIN (Reversible-Instance-Normalization)"""

    def __init__(
            self,
            num_features: int,
            eps=1e-5,
            affine=False,
            subtract_last=False,
            non_norm=False,
    ):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        :param substract_last: if True, the substraction is based on the last value
                               instead of the mean in normalization
        :param non_norm: if True, no normalization performed.
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        if self.non_norm:
            return x
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.non_norm:
            return x
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


class PatchTST_backbone(nn.Module):
    """
    PatchTST_backbone
    """

    def __init__(
            self,
            c_in: int,
            c_out: int,
            input_size: int,
            h: int,
            patch_len: int,
            stride: int,
            max_seq_len: Optional[int] = 1024,
            n_layers: int = 3,
            hidden_size=128,
            n_heads=16,
            d_k: Optional[int] = None,
            d_v: Optional[int] = None,
            linear_hidden_size: int = 256,
            norm: str = "BatchNorm",
            attn_dropout: float = 0.0,
            dropout: float = 0.0,
            act: str = "gelu",
            key_padding_mask: str = "auto",
            padding_var: Optional[int] = None,
            attn_mask: Optional[torch.Tensor] = None,
            res_attention: bool = True,
            pre_norm: bool = False,
            store_attn: bool = False,
            pe: str = "zeros",
            learn_pe: bool = True,
            fc_dropout: float = 0.0,
            head_dropout=0,
            padding_patch=None,
            pretrain_head: bool = False,
            head_type="flatten",
            individual=False,
            revin=True,
            affine=True,
            subtract_last=False,
    ):

        super().__init__()

        # RevIn
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((input_size - patch_len) / stride + 1)
        if padding_patch == "end":  # can be modified to general case #None
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1

        # Backbone
        self.backbone = TSTiEncoder(
            c_in,
            patch_num=patch_num,
            patch_len=patch_len,
            max_seq_len=max_seq_len,
            n_layers=n_layers,
            hidden_size=hidden_size,
            n_heads=n_heads,
            d_k=d_k,
            d_v=d_v,
            linear_hidden_size=linear_hidden_size,
            attn_dropout=attn_dropout,
            dropout=dropout,
            act=act,
            key_padding_mask=key_padding_mask,
            padding_var=padding_var,
            attn_mask=attn_mask,
            res_attention=res_attention,
            pre_norm=pre_norm,
            store_attn=store_attn,
            pe=pe,
            learn_pe=learn_pe,
        )

        # Head
        self.head_nf = hidden_size * patch_num
        self.n_vars = c_in
        self.c_out = c_out
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual

        if self.pretrain_head:
            self.head = self.create_pretrain_head(
                self.head_nf, c_in, fc_dropout
            )  # custom head passed as a partial func with all its kwargs
        elif head_type == "flatten":
            self.head = Flatten_Head(
                self.individual,
                self.n_vars,
                self.head_nf,
                h,
                c_out,
                head_dropout=head_dropout,
            )

    def forward(self, z):  # z: [bs x nvars x seq_len]
        # norm
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, "norm")
            z = z.permute(0, 2, 1)

        # do patching
        if self.padding_patch == "end":
            z = self.padding_patch_layer(z)
        z = z.unfold(
            dimension=-1, size=self.patch_len, step=self.stride
        )  # z: [bs x nvars x patch_num x patch_len]
        z = z.permute(0, 1, 3, 2)  # z: [bs x nvars x patch_len x patch_num]

        # model
        z = self.backbone(z)  # z: [bs x nvars x hidden_size x patch_num]
        z = self.head(z)  # z: [bs x nvars x h]

        # denorm
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, "denorm")
            z = z.permute(0, 2, 1)
        return z

    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(nn.Dropout(dropout), nn.Conv1d(head_nf, vars, 1))


class ConvNet(nn.Module):
    def __init__(self, c_in1=3, c_in2=7, input_size1=100, input_size2=360,
                 hidden_size=128, num_classes=2, encoder_layers=3, n_heads=8,
                 patch_len=16, stride=8, revin=False, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(3, 8)  # 第一全连接层 (p → 8)
        self.relu = nn.ReLU()               # ReLU激活函数
        self.fc2 = nn.Linear(8, 1)

        # 分支1处理3个特征，100时间步
        self.branch1 = PatchTST_backbone(
            c_in=c_in1,
            c_out=1,  # 输出通道设为1，配合h参数得到特征维度
            input_size=input_size1,
            h=hidden_size,  # 特征维度
            patch_len=patch_len,
            stride=stride,
            n_layers=encoder_layers,
            hidden_size=hidden_size,
            n_heads=n_heads,
            revin=revin,
            head_type='flatten',
            individual=False,  # 共享头部
            dropout=dropout
        )
        self.hidden_size =hidden_size
        # 分支2处理9个特征，360时间步
        self.branch2 = PatchTST_backbone(
            c_in=c_in2,
            c_out=1,
            input_size=input_size2,
            h=hidden_size,
            patch_len=patch_len,
            stride=stride,
            n_layers=encoder_layers,
            hidden_size=hidden_size,
            n_heads=n_heads,
            revin=revin,
            head_type='flatten',
            individual=False,
            dropout=dropout
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(10, num_classes)
        )
        # self.classifier = nn.Sequential(
        #     nn.Linear(12*hidden_size, num_classes)
        # )
        self.projector = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 处理分支1 [batch, 3, 100]
        out1 = self.branch1(x['data100'])  # 输出形状 [batch, 3, hidden_size]
        ji =  self.fc2(self.relu(self.fc1( x['data'][:, :3, :].permute(0,2,1)))).permute(0,2,1)
        ji = torch.cat([x['data'], ji], dim=1)
        # 处理分支2 [batch, 9, 360]
        out2 = self.branch2(ji)  # 输出形状 [batch, 9, hidden_size]


        # 拼接特征
        combined = torch.cat([out1, out2], dim=1)
        combined = self.dropout(combined)
        # combined = self.projector(combined)
        combined = torch.mean(combined, dim=2)
        # 分类
        logits = self.classifier(combined)
        return logits

class CustomProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        # 返回一个禁用的进度条
        bar = super().init_validation_tqdm()
        bar.disable = True  # 禁用验证进度条
        return bar


# PyTorch Lightning 数据模块 (修复了持久化工作进程问题)
class RainfallDataModule(pl.LightningDataModule):
    def __init__(self, train_mapping, val_mapping, test_mapping, batch_size=200, num_workers=4):
        super().__init__()
        self.train_mapping = train_mapping
        self.val_mapping = val_mapping
        self.test_mapping = test_mapping
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # 确保总是设置所有数据集
        self.train_dataset = CSVDataset(self.train_mapping)
        self.val_dataset = CSVDataset(self.val_mapping)
        self.test_dataset = CSVDataset(self.test_mapping)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0

        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0
        )


class LossStabilityCallback(pl.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        current_loss = trainer.callback_metrics["train_loss_epoch"]
        if not hasattr(self, 'loss_history'):
            self.loss_history = []
        self.loss_history.append(current_loss.item())


# PyTorch Lightning 模型模块
class RainfallModel(pl.LightningModule):
    def __init__(self, lr=0.001, dropout=0.2, hidden_size=None, encoder_layers=None, n_heads=None, patch_len=None, stride=None,
                 class_weights=None):
        super().__init__()
        self.save_hyperparameters()

        self.model = ConvNet(
            dropout=dropout,
            hidden_size=hidden_size,
            encoder_layers=encoder_layers,
            n_heads=n_heads,
            patch_len=patch_len,
            stride=stride,
        )

        self.class_weights = class_weights
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights))
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.lr = lr

        # 用于存储每折的验证结果
        self.validation_step_outputs = []
        # 用于测试
        self.test_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs = {k: v for k, v in batch.items() if k != 'labels'}
        labels = batch['labels']

        outputs = self(inputs)
        loss = self.criterion(outputs, labels)

        # 计算训练准确率
        _, preds = torch.max(outputs, 1)
        acc = (preds == labels).float().mean()

        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = {k: v for k, v in batch.items() if k != 'labels'}
        labels = batch['labels']

        outputs = self(inputs)
        loss = self.criterion(outputs, labels)

        # 计算验证准确率
        _, preds = torch.max(outputs, 1)
        acc = (preds == labels).float().mean()

        # 存储结果用于epoch结束时的计算
        results = {
            'val_loss': loss,
            'val_acc': acc,
            'labels': labels,
            'preds': preds
        }
        self.validation_step_outputs.append(results)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

        return results

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        if not outputs:
            return

        # 使用与重新评估相同的计算方式
        all_labels = torch.cat([x['labels'] for x in outputs])
        all_preds = torch.cat([x['preds'] for x in outputs])
        print("all_preds:", len(all_preds))
        # 添加zero_division参数确保一致性
        precision_1 = precision_score(all_labels.cpu().numpy(),
                                      all_preds.cpu().numpy(),
                                      pos_label=1,
                                      zero_division=0)

        recall_1 = recall_score(all_labels.cpu().numpy(),
                                all_preds.cpu().numpy(),
                                pos_label=1,
                                zero_division=0)

        f1_1 = fbeta_score(all_labels.cpu().numpy(),
                 all_preds.cpu().numpy(),
                 beta=2,  # 添加这个关键参数
                 pos_label=1,
                 zero_division=0)

        # 存储到模块属性中
        self.avg_val_precision_1 = precision_1
        self.avg_val_recall_1 = recall_1
        self.avg_val_f1 = f1_1
    

        self.log('avg_val_precision_1', precision_1, prog_bar=True)
        self.log('avg_val_recall_1', recall_1, prog_bar=True)
        self.log('avg_val_f1', f1_1, prog_bar=True)
        
        print(" call_current_f1", f1_1)

        self.validation_step_outputs.clear()
        return {'val_precision_1': precision_1, 'val_recall_1': recall_1, 'val_f1': f1_1}

    def test_step(self, batch, batch_idx):
        inputs = {k: v for k, v in batch.items() if k != 'labels'}
        labels = batch['labels']

        outputs = self(inputs)
        probs = torch.softmax(outputs, dim=1)
        pos_probs = probs[:, 1]  # 正类概率
        preds = (pos_probs >= 0.5).long()  # 根据阈值0.5进行二值化

        result = {'labels': labels, 'preds': preds}
        self.test_step_outputs.append(result)
        return result

    def on_test_epoch_end(self):
        # 聚合所有测试步骤的结果
        labels = torch.cat([x['labels'] for x in self.test_step_outputs])
        preds = torch.cat([x['preds'] for x in self.test_step_outputs])

        precision_1 = precision_score(labels.cpu().numpy(), preds.cpu().numpy(), pos_label=1, zero_division=0)
        # 计算召回率
        recall_1 = recall_score(labels.cpu().numpy(), preds.cpu().numpy(), pos_label=1, zero_division=0)
        # 记录召回率
        self.log('test_precision_1', precision_1, 'test_recall_1', recall_1, prog_bar=True)

        # 清空
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# 自定义回调函数，用于在满足召回率条件下选择F1最高的模型
class RecallConditionalModelCheckpoint(pl.Callback):
    def __init__(self, recall_threshold=0.85, val_mapping=None, save_dir=None):
        super().__init__()
        self.recall_threshold = recall_threshold
        self.save_dir = save_dir or os.getcwd()
        os.makedirs(self.save_dir, exist_ok=True)

        # 存储当前epoch状态
        self.current_epoch_state = None
        self.best_model_path = None

        # 最佳模型指标
        self.best_f1 = 0.0
        self.best_recall = 0.0
        self.best_epoch = -1
        self.val_mapping = val_mapping

    def on_train_epoch_end(self, trainer, pl_module):
        """保存当前训练结束时的模型状态"""
        # 深拷贝模型状态
        model_state = deepcopy(pl_module.state_dict())

        # 获取当前优化器状态
        optimizer_state = None
        if trainer.optimizers:
            optimizer_state = deepcopy(trainer.optimizers[0].state_dict())

        # 获取Lightning版本
        from pytorch_lightning import __version__ as lightning_version

        # 构建完整状态字典
        self.current_epoch_state = {
            'epoch': trainer.current_epoch,
            'global_step': trainer.global_step,
            'pytorch-lightning_version': lightning_version,
            'state_dict': model_state,
            'optimizer_states': [optimizer_state] if optimizer_state else [],
            'lr_schedulers': [],
            'callbacks': {},  # 空回调状态
            'hparams_name': 'hparams',  # 必需字段
            'hyper_parameters': pl_module.hparams
        }

    def on_validation_epoch_end(self, trainer, pl_module):
        """验证结束时更新最佳模型"""
        # 1. 获取当前验证指标
        metrics = trainer.callback_metrics
        current_recall = metrics.get('avg_val_recall_1', 0.0)
        current_f1 = metrics.get('avg_val_f1', 0.0)
        epoch = trainer.current_epoch

        # 2. 更新最佳模型条件
        should_save = False
        if current_recall >= self.recall_threshold:
            if current_f1 > self.best_f1:
                should_save = True
                self.best_f1 = current_f1
                self.best_recall = current_recall
                self.best_epoch = epoch

        # 3. 保存最佳模型
        if should_save and self.current_epoch_state:
            # 准备保存路径
            filename = f"best_epoch={epoch}_recall={current_recall:.4f}_f1={current_f1:.4f}.ckpt"
            best_candidate_path = os.path.join(self.save_dir, filename)

            # 保存完整检查点
            torch.save(self.current_epoch_state, best_candidate_path)
            self.best_model_path = best_candidate_path
            print(f"💾 保存最佳模型到: {best_candidate_path}")



    def on_train_end(self, trainer, pl_module):
        """训练结束时总结最佳模型"""
        if self.best_model_path:
            print("\n🏆 训练完成 - 最佳模型总结:")
            print(f"   - 路径: {self.best_model_path}")
            print(f"   - Epoch: {self.best_epoch}")
            print(f"   - 召回率: {self.best_recall:.4f}")
            print(f"   - F1分数: {self.best_f1:.4f}")
        else:
            print("⚠️ 未找到满足条件的模型")

# 贝叶斯优化目标函数
def objective(trial, train_mapping, val_mapping, test_mapping, fold_idx, save_dir):
    # 定义超参数搜索空间
    hidden_size = trial.suggest_categorical("hidden_size", [32,64, 128]) #C
    encoder_layers = trial.suggest_int("encoder_layers",  1, 3)#C
    n_heads = trial.suggest_categorical("n_heads", [2, 4,8])#C
    patch_len = trial.suggest_int("patch_len", 8, 24, step=4)
    stride = trial.suggest_int("stride", 3, 7, step=2)#C

    lr = trial.suggest_float('lr', 1e-3, 1e-2, log=True)
    dropout = trial.suggest_categorical('dropout', [0.1, 0.3, 0.5])
    batch_size = trial.suggest_categorical('batch_size', [100])
    class_weights = [1.0, trial.suggest_categorical('class_weight', [11])]
    # 创建数据模块
    data_module = RainfallDataModule(
        train_mapping=train_mapping,
        val_mapping=val_mapping,
        test_mapping=test_mapping,
        batch_size=batch_size
    )

    # 创建模型
    model = RainfallModel(
        lr=lr,
        dropout=dropout,
        hidden_size=hidden_size,
        encoder_layers=encoder_layers,
        n_heads=n_heads,
        patch_len=patch_len,
        stride=stride,
        class_weights=class_weights
    )

    # 创建日志和模型保存路径
    # logger = CSVLogger(save_dir, name=f"fold_{fold_idx}")
    logger = TensorBoardLogger(
        save_dir=os.path.join(save_dir, f"fold_{fold_idx}"),
        name=f"trial_{trial.number}",
        default_hp_metric=False  # 避免重复记录超参数
    )
    checkpoint_dir = os.path.join(logger.log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 自定义回调函数 - 在满足召回率条件下选择F1最高的模型
    recall_checkpoint = RecallConditionalModelCheckpoint(
        recall_threshold=0.8,
        val_mapping=val_mapping,
    save_dir = checkpoint_dir
    )

    # 早停回调 - 监控召回率
    early_stop_callback = EarlyStopping(
        monitor='avg_val_f1',
        min_delta=0.001,
        patience=40,
        verbose=True,
        mode='max',
        stopping_threshold=0.95
    )
    model_checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='best_model-{epoch}-{avg_val_f1:.4f}',
        monitor='avg_val_f1',
        mode='max',
        save_top_k=1
    )
    progress_bar = CustomProgressBar()
    # 创建训练器
    trainer = pl.Trainer(
        max_epochs=200,
        logger=logger,
        callbacks=[LossStabilityCallback(), recall_checkpoint, early_stop_callback, progress_bar],
        enable_progress_bar=True,
        log_every_n_steps=5,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=[0],

    )

    # 训练模型
    trainer.fit(model, datamodule=data_module)

    # 获取最佳模型路径
    try:
        best_model_path = recall_checkpoint.best_model_path or recall_checkpoint.best_f1_model_path
    except:
        best_model_path = None

    # 手动加载最佳模型并在测试集上评估
    if best_model_path:
        print(f"📦 加载最佳模型: {best_model_path}")
        model = RainfallModel.load_from_checkpoint(best_model_path)
        model.to(device)
        model.eval()

        # 创建测试数据集
        val_dataset = CSVDataset(val_mapping)
        print("val_dataset", len(val_dataset))
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn,
            persistent_workers=True,
            pin_memory=True
        )

        # 评估模型在验证集上的表现
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue

                inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        # 计算验证集上的评估指标
        precision_1 = precision_score(all_labels, all_preds, pos_label=1, zero_division=0)
        recall_1 = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
        f11 = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        ac = accuracy_score(all_labels, all_preds)
        print(f"验证集：f1: {f1},精确度：{precision_1:.4f}，召回率: {recall_1:.4f}，F1_1分数: {f11:.4f}")
        return f11

    else:
        print("❌ 未找到最佳模型路径")
        return 0.0


# 主程序
if __name__ == "__main__":
    # 配置路径
    base_path ="../trainall"
    train_txt_path = "train.txt"
    test_txt_path = "test.txt"

    # 创建完整数据集
    train_mapping = create_file_label_mapping(train_txt_path, base_path)
    test_mapping = create_file_label_mapping(test_txt_path, base_path)

    # 创建数据集以获取标签
    dataset = CSVDataset(train_mapping)
    labels = [item['label'].item() for item in dataset if item is not None]

    # 五折交叉验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    base_save_dir = os.path.join('./results', current_time)

    # 创建结果目录
    os.makedirs(base_save_dir, exist_ok=True)

    # 存储所有折的最佳模型路径
    best_models = []
    best_model_path = None
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n🚀 开始第 {fold_idx + 1}/5 折训练")

        # 创建当前折的保存目录
        fold_save_dir = os.path.join(base_save_dir, f"fold_{fold_idx + 1}")
        os.makedirs(fold_save_dir, exist_ok=True)

        # 创建当前折的训练映射和验证映射
        fold_train_mapping = [train_mapping[i] for i in train_idx]
        fold_val_mapping = [train_mapping[i] for i in val_idx]
        print("fold_val_mapping:", len(fold_val_mapping))

        # 创建Optuna研究
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )

        # 优化目标函数
        study.optimize(
            lambda trial: objective(trial, fold_train_mapping, fold_val_mapping, test_mapping, fold_idx + 1,
                                    fold_save_dir),
            n_trials=30,
            show_progress_bar=True
        )

        # 保存最佳超参数
        best_params = study.best_params
        with open(os.path.join(fold_save_dir, 'best_params.json'), 'w') as f:
            json.dump(best_params, f, indent=4)

        # 获取最佳模型路径
        # 注意：目标函数中已经保存了最佳模型路径
        # 这里我们假设目标函数返回的模型路径保存在回调函数中

        best_f1 = 0

        # 遍历所有版本目录寻找真正的"最佳模型"
        lightning_logs_dir = os.path.join(fold_save_dir, f"fold_{fold_idx + 1}")
        if os.path.exists(lightning_logs_dir):
            for version in os.listdir(lightning_logs_dir):
                version_dir = os.path.join(lightning_logs_dir, version)
                checkpoint_dir = os.path.join(version_dir, "version_0/checkpoints")

                if os.path.exists(checkpoint_dir):
                    # 检查模型文件
                    for model_file in os.listdir(checkpoint_dir):
                        if model_file.startswith("best_epoch") and model_file.endswith(".ckpt"):
                            # 从文件名解析F1值（根据您文件名中的约定）
                            try:
                                parts = model_file.split('_')

                                f1_value = float(parts[-1].split('=')[1].replace('.ckpt', ''))

                                # 检查是否优于当前最佳模型
                                if f1_value > best_f1:
                                    best_f1 = f1_value
                                    best_model_path = os.path.join(checkpoint_dir, model_file)

                            except:
                                # 如果解析失败，至少记录路径
                                print(f"Warning: Failed to parse F1 from {model_file}")

        if best_model_path:
            print(f"✅ Found best model for fold {fold_idx + 1} with F1={best_f1:.4f}: {best_model_path}")
            best_models.append(best_model_path)
            print(best_models)
        else:
            print(f"❌ No suitable model found for fold {fold_idx + 1}")
            best_models.append(None)

    # 使用所有折的最佳模型进行最终测试评估
    final_results = []
    for fold_idx, model_path in enumerate(best_models):
        if model_path is None:
            print(f"\n❌ 第 {fold_idx + 1} 折没有找到模型，跳过评估")
            continue

        print(f"\n🔍 评估第 {fold_idx + 1} 折的最佳模型在测试集上的表现")
        print(f"模型路径: {model_path}")

        # 加载模型
        model = RainfallModel.load_from_checkpoint(model_path)
        model.to(device)
        model.eval()

        # 获取参数量
        param_count = model.count_parameters()

        # 获取当前折的超参数
        fold_save_dir = os.path.join(base_save_dir, f"fold_{fold_idx + 1}")
        hyperparams_path = os.path.join(fold_save_dir, 'best_params.json')
        try:
            with open(hyperparams_path, 'r') as f:
                hyperparams = json.load(f)
            # 将超参数转换为字符串格式
            hyperparams_str = json.dumps(hyperparams, ensure_ascii=False)
        except Exception as e:
            print(f"❌ 无法读取超参数文件: {e}")
            hyperparams_str = "{}"

        # 创建测试数据加载器
        test_dataset = CSVDataset(test_mapping)
        test_loader = DataLoader(
            test_dataset,
            batch_size=40,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn,
            persistent_workers=True
        )

        # 评估模型
        all_labels = []
        all_probs = []
        all_preds = []

        with torch.no_grad():
            for batch in test_loader:
                if batch is None:
                    continue

                inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].to(device)

                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # 正类概率
                all_preds.extend(preds.cpu().numpy())

        # 计算指标
        test_acc = (np.array(all_preds) == np.array(all_labels)).mean()
        test_precision_1 = precision_score(all_labels, all_preds, pos_label=1, zero_division=0)
        test_recall_1 = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
        test_f1 = fbeta_score(all_labels, all_preds, beta=2,pos_label=1, zero_division=0)

        # 计算模型参数量
        param_count = model.count_parameters()

        # 保存结果
        result = {
            'fold': fold_idx + 1,
            'test_acc': test_acc,
            'test_precision_1': test_precision_1,
            'test_recall_1': test_recall_1,
            'test_f1': test_f1,
            'param_count': param_count,
            'hyperparams': hyperparams_str,
            'model_path': model_path
        }

        final_results.append(result)

        print(f"📊 第 {fold_idx + 1} 折测试结果:")
        print(f"准确率: {test_acc:.4f}")
        print(f"精确率 (类别1): {test_precision_1:.4f}")
        print(f"召回率 (类别1): {test_recall_1:.4f}")
        print(f"F1分数 (类别1): {test_f1:.4f}")
        print(f"参数量: {param_count}")


        import csv
        test_name = f'test_probabilities_{fold_idx}_{current_time.replace(":", "-")}.csv'
        test_name = os.path.join(base_save_dir, test_name)
        with open(test_name, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['File Path', 'True Label', 'Probability (Class 1)'])
            for (file_path, label), prob in zip(test_mapping, all_probs):
                writer.writerow([file_path, label, f"{prob:.4f}"])


    # 保存最终结果
    results_df = pd.DataFrame(final_results)
    results_path = os.path.join(base_save_dir, 'final_results.csv')
    results_df.to_csv(results_path, index=False)

    print("\n🎉 所有折完成! 最终结果保存于:", results_path)

