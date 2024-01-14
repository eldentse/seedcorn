import argparse
from datetime import datetime

# from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn import metrics
import numpy as np
# from torch.utils.tensorboard import SummaryWriter

from utils import train_utils, argutils, common_utils, model_utils
from models import model_factory
from data import data_factory


def main(args):
    best_score = None
    # Initialise random seeds
    train_utils.set_all_seeds(args.manual_seed)

    # Initialise hosting
    now = datetime.now()
    experiment_tag = args.experiment_tag
    exp_id = f"{args.cache_folder}"+experiment_tag+"/"

    # Initialise local checkpoint folder
    argutils.save_args(args, exp_id, "opt")
    # board_writer = SummaryWriter(log_dir=exp_id) 

    # Initialise training dataset
    print("Get datset split", args.train_split)

    train_dataset, test_dataset, data_helper = data_factory.get_dataset(
        dataset_folder=args.dataset_folder, 
        split=args.train_split,
        split_ratio=args.split_ratio,
        split_type=args.split_type,
        path_to_txt=exp_id
        )

    
    # Initialise model
    print("Get model", args.model)
    model = model_factory.get_model(
        model_name=args.model, 
        dims=data_helper,
        dropout=args.dropout
        )

    ## For checks only
    # input = next(iter(train_dataset))
    # output = model(input.x, input.edge_index, input.edge_weight)

    # Resume training
    if args.train_cont:
        start_epoch = train_utils.reload_model(model, args.resume_path)       
    else:
        start_epoch = 0
    start_epoch += 1

    # GPU training
    use_multiple_gpu = False
    if args.use_GPU:
        use_multiple_gpu= torch.cuda.device_count() > 1
        if use_multiple_gpu:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model).cuda()
        else:
            print("Using one GPU!")
            model.cuda()

    # Optionally freeze batchnorm  
    if args.freeze_batch_norm:
        train_utils.freeze_batchnorm_stats(model)

    # Number of parameters
    print("Numbers of parameters to update:", model_utils.count_parameters(model))


    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.BCEWithLogitsLoss()


    print(f"{train_dataset.snapshot_count} Training Data Points | {test_dataset.snapshot_count} Validation Data Points")
    for epoch in range(args.epochs):
        model, optimizer, train_loss_total, train_acc = train_utils.train(train_dataset, 
                                                              model, 
                                                              optimizer,
                                                              loss_fn)
        model, val_loss_total, val_acc = train_utils.val(test_dataset, 
                                                        model,
                                                        loss_fn)
        print(f"EPOCH: {epoch}/{args.epochs} | TRAIN LOSS: {train_loss_total:.2f} | TRAIN ACC: {train_acc:.2f} | VAL LOSS: {val_loss_total:.2f} | VAL ACC: {val_acc:.2f}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument('--experiment_tag', default='hello') 
    parser.add_argument('--dataset_folder', default='/home/eldentse94/Documents/VSCodeProjects/SeedCorn/data')
    parser.add_argument('--cache_folder', default='checkpoints/')
    parser.add_argument('--resume_path', default=None)

    # Dataset parameters
    parser.add_argument("--train_split", 
                        default="train", choices=["test", "train"])
    parser.add_argument("--split_ratio", type=float, default=0.3)
    parser.add_argument("--split_type", 
                        default="base", 
                        help="Detailed implementation of split type can be found in dataset_factory")
    
    # Training parameters
    parser.add_argument("--train_cont", 
                        action="store_true", help="Continue from previous training")
    parser.add_argument("--manual_seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--workers", 
                        type=int, default=8, help="Number of workers for multiprocessing")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr_decay_gamma", 
                        type=float, default= 0.5,help="Learning rate decay factor, if 1, no decay is effectively applied")
    parser.add_argument("--lr_decay_step", type=float, default=15)
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--snapshot", type=int, default=5, help="How often to save intermediate models (epochs)" )
    parser.add_argument("--freeze_batch_norm", default=False)
    parser.add_argument("--use_GPU", default=False)

    # Model
    parser.add_argument("--model", default="STGCN")
    parser.add_argument("--dropout", default=0.1)

    # Loss
    parser.add_argument("--loss_function", default="ce", choices=['l1', 'l2', 'ce'])

    args = parser.parse_args()
    for key, val in sorted(vars(args).items(), key=lambda x: x[0]):
        print(f"{key}: {val}")

    main(args)
    print("All done !")