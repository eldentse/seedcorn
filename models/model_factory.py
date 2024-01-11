from models.base_layers.MLP import MLP
from models.Transformer import Transformer, MT_Transformer

def get_model(model_name, dims, dropout, multi_task):
    softmax = True if dims[-1] > 1 else False
   
    if model_name == "MLP":
        return MLP(base_neurons=[dims[0], 512, 512], out_dim=dims[1], softmax=softmax)
    
    if model_name == "Transformer" and not multi_task:
        return Transformer(d_model=256, 
                           nhead=2,
                           num_encoder_layers=2,
                           dim_feedforward=512,
                           dropout=dropout,
                           activation="relu",
                           normalize_before=True,
                           dims = dims,
                           softmax=softmax)

    if model_name == "Transformer" and multi_task:
        return MT_Transformer(d_model=256, 
                              nhead=2,
                              num_encoder_layers=2,
                              dim_feedforward=512,
                              dropout=dropout,
                              activation="relu",
                              normalize_before=True,
                              dims = dims,
                              softmax=softmax)
    
    else:
        raise ValueError("Model %s is not implemented!" % model_name)