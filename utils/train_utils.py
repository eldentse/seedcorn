import random
import torch
import numpy as np
import os
import warnings
import traceback
from tqdm import tqdm
from sklearn import metrics

from utils.model_utils import load_checkpoint
# from data.queries import BaseQueries


def set_all_seeds(seed):
    """Sets the seed for generating random numbers."""
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def reload_model(model, resume_checkpoint, optimizer=None):
    if resume_checkpoint:
        start_epoch, _ = load_checkpoint(
            model, optimizer=optimizer, resume_path=resume_checkpoint, strict=False, as_parallel=False
        )
    else:
        start_epoch = 0
    return start_epoch


def reload_optimizer(resume_path, optimizer, scheduler=None):
    if os.path.isfile(resume_path):
        print("=> loading checkpoint '{}'".format(resume_path))
        checkpoint = torch.load(resume_path)
    try:
        missing_states = set(optimizer.state_dict().keys()) - set(checkpoint["optimizer"].keys())
        if len(missing_states) > 0:
            warnings.warn("Missing keys in optimizer ! : {}".format(missing_states))
        optimizer.load_state_dict(checkpoint["optimizer"])
    except ValueError:
        traceback.print_exc()
        warnings.warn("Couldn't load optimizer from {}".format(resume_path))

    if not scheduler is None:
        try:
            scheduler.load_state_dict(checkpoint["scheduler"].state_dict())
        except ValueError:
            traceback.print_exc()
            warnings.warn("Couldn't load scheduler from {}".format(resume_path))


def freeze_batchnorm_stats(model):
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.momentum = 0
    for name, child in model.named_children():
        freeze_batchnorm_stats(child)


def get_accuracy(y_true, y_prob):
    y_true =np.argmax(np.vstack(y_true), axis=1)
    y_prob =np.argmax(np.vstack(y_prob), axis=1)
    accuracy = metrics.accuracy_score(y_true, y_prob)
    return accuracy


def train(dataset, model, optimizer, loss_fn):
    loss_total = 0
    preds = []
    labels = []
    model.train()
    for i, snapshot in enumerate(dataset):
        y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr[0])
        y_gt = snapshot.y[0]
        
        preds.append(y_hat.detach().numpy())
        labels.append(y_gt.detach().numpy())

        loss = loss_fn(y_hat, y_gt)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_total += loss.item()
    acc = get_accuracy(labels, preds)
    return model, optimizer, loss_total, acc
    

def val(dataset, model, loss_fn):
    loss_total = 0
    preds = []
    labels = []
    model.eval()
    for i, snapshot in enumerate(dataset):
        y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr[0])
        y_gt = snapshot.y[0]
        preds.append(y_hat.detach().numpy())
        labels.append(y_gt.detach().numpy())
        loss = loss_fn(y_hat, y_gt)

        loss_total += loss.item()
    
    acc = get_accuracy(labels, preds)
    return model, loss_total, acc