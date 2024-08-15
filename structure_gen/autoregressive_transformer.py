import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import math
from tqdm import tqdm


# Ref: https://github.com/YatingMusic/compound-word-transformer/blob/main/workspace/uncond/cp-linear/main-cp.py
# https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
# https://gist.github.com/danimelchor/bcad4d7f79b98464c4d4481d62d27622

class Embeddings(nn.Module):
    """
    Get embeddings for edge tokens
    """
    def __init__(self, n_token, d_model):
        super(Embeddings, self).__init__()
        # print(n_token)
        self.lut = nn.Embedding(n_token, d_model, padding_idx=0)
        self.d_model = d_model

    def forward(self, x):
        # print(n_token)
        # print(x.shape)
        # print(torch.max(x))
        # print(torch.min(x))
        # print(self.d_model)
        # print(self.lut(x))
        return self.lut(x) * math.sqrt(self.d_model)
    
    
class PositionalEncoding(nn.Module):
    """
    Get positional encodings
    """
    def __init__(self, d_model, dropout=0.1, max_len=20000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
        


class AutoregressiveTransformer(nn.Module):
    def __init__(self, n_token):
        super(AutoregressiveTransformer, self).__init__()
        
        # --- params config --- #
        self.n_token = n_token   
        self.d_model = D_MODEL 
        self.d_feedforward = D_FEEDFW
        self.n_layer = N_LAYER
        self.dropout = 0.1
        self.n_head = N_HEAD
        self.d_head = D_MODEL // N_HEAD
        self.d_inner = 1024
        # self.loss_func = nn.CrossEntropyLoss(reduction='none')
        self.emb_sizes = [128, 128, 12, 16, 16]
        
        
        # --- modules config --- #
        # embeddings
        print('>>>>>:', self.n_token)
        self.emb_i = Embeddings(self.n_token[0], self.emb_sizes[0])
        self.emb_j = Embeddings(self.n_token[1], self.emb_sizes[1])
        self.emb_edge_type = Embeddings(self.n_token[2], self.emb_sizes[2])
        self.emb_i_size = Embeddings(self.n_token[3], self.emb_sizes[3])
        self.emb_j_size = Embeddings(self.n_token[4], self.emb_sizes[4])
        self.pos_emb = PositionalEncoding(self.d_model, self.dropout)

        # linear 
        self.in_linear = nn.Linear(np.sum(self.emb_sizes), self.d_model)
        
        # encoder
        self.transformer = nn.Transformer(
            d_model=self.d_model,
            nhead=self.n_head,
            num_encoder_layers=self.n_layer,
            num_decoder_layers=self.n_layer,
            dim_feedforward=self.d_feedforward,
            dropout=self.dropout,
        )

        # individual output
        self.proj_i    = nn.Linear(self.d_model, self.n_token[0])        
        self.proj_j    = nn.Linear(self.d_model, self.n_token[1])
        self.proj_edge_type  = nn.Linear(self.d_model, self.n_token[2])
        self.proj_i_size     = nn.Linear(self.d_model, self.n_token[3])
        self.proj_j_size    = nn.Linear(self.d_model, self.n_token[4])
    
    def forward(self, src, tgt, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
        '''
        linear transformer: b x s x f
        x.shape=(bs, nf)
        '''
        
        # print(f"src shape: {src.shape}, target shape: {tgt.shape}")
    
        # src embeddings
        emb_i_src =    self.emb_i(src[..., 0])
        emb_j_src =    self.emb_j(src[..., 1])
        emb_edge_type_src =  self.emb_edge_type(src[..., 2])
        emb_i_size_src =     self.emb_i_size(src[..., 3])
        emb_j_size_src =    self.emb_j_size(src[..., 4])

        embs_src = torch.cat(
            [
                emb_i_src,
                emb_j_src,
                emb_edge_type_src,
                emb_i_size_src,
                emb_j_size_src,
            ], dim=-1)

        emb_linear_src = self.in_linear(embs_src)
        pos_emb_src = self.pos_emb(emb_linear_src)
        
        
        # tgt embeddings
        emb_i_tgt =    self.emb_i(tgt[..., 0])
        emb_j_tgt =    self.emb_j(tgt[..., 1])
        emb_edge_type_tgt =  self.emb_edge_type(tgt[..., 2])
        emb_i_size_tgt =     self.emb_i_size(tgt[..., 3])
        emb_j_size_tgt =    self.emb_j_size(tgt[..., 4])

        embs_tgt = torch.cat(
            [
                emb_i_tgt,
                emb_j_tgt,
                emb_edge_type_tgt,
                emb_i_size_tgt,
                emb_j_size_tgt,
            ], dim=-1)

        emb_linear_tgt = self.in_linear(embs_tgt)
        pos_emb_tgt = self.pos_emb(emb_linear_tgt)
        
        # target embeddings
    
        # transformer
        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        # print(pos_emb_src.shape)
        # print(pos_emb_tgt.shape)
        pos_emb_src = pos_emb_src.permute(1,0,2)
        pos_emb_tgt = pos_emb_tgt.permute(1,0,2)
        # print(f"Transformer input dim: {pos_emb_src.shape}, target dim: {pos_emb_tgt.shape}")
        transformer_out = self.transformer(pos_emb_src, pos_emb_tgt, 
                                           tgt_mask=tgt_mask, 
                                           src_key_padding_mask=src_pad_mask, 
                                           tgt_key_padding_mask=tgt_pad_mask)
        # print(f"Transformer output dim: {transformer_out.shape}")
        

        y_i    = self.proj_i(transformer_out)
        y_j    = self.proj_j(transformer_out)
        y_edge_type  = self.proj_edge_type(transformer_out)
        y_i_size    = self.proj_i_size(transformer_out)
        y_j_size = self.proj_j_size(transformer_out)

        return  y_i, y_j, y_edge_type, y_i_size, y_j_size
    
    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a square matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        
        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]
        
        return mask
    
    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return (matrix == pad_token)
    
    
    
    
        

