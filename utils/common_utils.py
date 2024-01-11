import torch
import torch.nn.functional as torch_f
import torch.nn as nn


def loss_str2func():
    return {'l1': torch_f.l1_loss,
            'l2': torch_f.mse_loss,
            'ce': torch_f.cross_entropy}


def act_str2func():
    return {'softmax': nn.Softmax(),
            'elu': nn.ELU(),
            'leakyrelu': nn.LeakyReLU(),
            'relu': nn.ReLU()}


def torch2numpy(input):
    if input is None:
        return None
    
    elif torch.is_tensor(input):
        input = input.detach().cpu().numpy()
        return input
    
    elif type(input) == dict:
        # Convert CUDA tensor dictionary to CPU
        cpu_dict = {key: tensor.cpu() for key, tensor in input.items()}
        # Convert CPU tensor dictionary to NumPy array dictionary
        numpy_dict = {key: tensor.numpy() for key, tensor in cpu_dict.items()}
        return numpy_dict
    
    else:
        assert False, "Something is wrong here!"