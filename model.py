import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()

        self.dropout = nn.Dropout(dropout_p)

        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)

        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)

        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        # Saving bufer: GPU에서 처리되고 state_dict()에도 존재하지만, 업데이트해야 할 파라미터가 아니므로 parameter가 아닌 buffer에 저장
        pos_encoding = pos_encoding.unsqueeze(0)
        self.register_buffer("pos_encoding", pos_encoding)
    
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + self.pos_encoding
        return self.dropout(token_embedding + self.pos_encoding)

class TransformerEnc(nn.Module):
    def __init__(self, dim_model, num_heads, num_encoder_layers, dropout_p, device):
        super().__init__()

        self.model_type = "Transformer"
        self.dim_model = dim_model
        self.device = device

        self.positional_encoder = PositionalEncoding(dim_model=126, dropout_p=dropout_p, max_len=125)

        # self.jointdim = nn.Linear(128, 8) #ALTERNATIVE
        self.conv1 = nn.Conv2d(1,126,(51,9),(1,1),(0,4)) # 25 * 3 = 75 -> 17 * 3 = 51
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.bn = nn.BatchNorm1d(dim_model) #B,S,D
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=num_heads)
        self.transformer0 = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        
        self.out = nn.Linear(128, 8)

        self.jointdim = nn.Linear(128, 8) #ALTERNATIVE

    def forward(self, src, condval, src_pad_mask=None, tgt_pad_mask=None):
        '''
        Src size must be (batch_size, src sequence length)
        Tgt size must be (batch_size, tgt sequence length)
        '''
        src = src.float() * math.sqrt(self.dim_model)
        src = torch.unsqueeze(src, dim=1).permute(0, 1, 3, 2)
        src = self.conv1(src)[:, :, 0, :].permute(0, 2, 1)
        src = self.positional_encoder(src)
        src = src.to(torch.float32).to(self.device)

        condition = torch.ones(src.shape[0], src.shape[1], 2).to(self.device)
        condition = condition * condval.reshape(condval.shape[0], 1, 1)
        condition = condition.to(torch.float32).to(self.device)

        src = torch.cat([src, condition], dim=2) # torch.cuda.FloatTensor
        src = src.permute(1, 0, 2)

        transformer_out = self.transformer0(src, src_pad_mask)

        out = self.out(transformer_out)
        out = out.permute(1, 2, 0)

        return out

