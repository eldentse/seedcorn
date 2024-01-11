import torch
import torch.nn as nn

from models.base_layers.transformer_layers import Transformer_Encoder, PositionalEncoding
from models.base_layers.MLP import MLP


class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 dims=[10,10], softmax=False):
        super().__init__()
        self.projecttion_layer = MLP(base_neurons=[1, d_model, d_model], 
                                    out_dim=d_model, 
                                    softmax=False)
        self.positional_encoder = PositionalEncoding(d_model)
        self.transformer_encoder = Transformer_Encoder(d_model, 
                                                       nhead, 
                                                       num_encoder_layers,
                                                       dim_feedforward,
                                                       dropout, 
                                                       activation, 
                                                       normalize_before)
        
        self.apply_softmax = softmax
        self.final_layer = MLP(base_neurons=[dims[0]*d_model, d_model, d_model], 
                               out_dim=dims[1], 
                               softmax=self.apply_softmax)
        
    
    def forward(self, x):
        B = x.size()[0]
        if x.dim() < 3:
            x = torch.unsqueeze(x, dim=-1)
        x_embedding = self.projecttion_layer(x)
        x_pe = self.positional_encoder(x_embedding)
        x, _ = self.transformer_encoder(src=x_embedding, 
                                        src_pos=x_pe, 
                                        key_padding_mask=None, 
                                        attn_mask=None, 
                                        verbose=False)
        x_flatten = torch.reshape(x, [B, -1])
        out = self.final_layer(x_flatten)
        return out
    

class MT_Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 dims=[10,10], softmax=False):
        super().__init__()
        self.projecttion_layer = MLP(base_neurons=[1, d_model, d_model], 
                                    out_dim=d_model, 
                                    softmax=False)
        self.positional_encoder = PositionalEncoding(d_model)
        self.transformer_encoder = Transformer_Encoder(d_model, 
                                                       nhead, 
                                                       num_encoder_layers,
                                                       dim_feedforward,
                                                       dropout, 
                                                       activation, 
                                                       normalize_before)
        
        self.apply_softmax = softmax
        self.final_layer = MLP(base_neurons=[dims[0]*d_model, d_model, d_model], 
                               out_dim=sum(dims[1:]), 
                               softmax=self.apply_softmax)
        
    
    def forward(self, x):
        B = x.size()[0]
        if x.dim() < 3:
            x = torch.unsqueeze(x, dim=-1)
        x_embedding = self.projecttion_layer(x)
        x_pe = self.positional_encoder(x_embedding)
        x, _ = self.transformer_encoder(src=x_embedding, 
                                        src_pos=x_pe, 
                                        key_padding_mask=None, 
                                        attn_mask=None, 
                                        verbose=False)
        x_flatten = torch.reshape(x, [B, -1])
        out = self.final_layer(x_flatten)
        pred = {}
        pred['SUMINS'] = out[:, :8]
        pred['FILE_NAME'] = out[:, 8:]
        return pred