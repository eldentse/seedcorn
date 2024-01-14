import argparse
from datetime import datetime

# from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
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

    # Optimise unfrozen parts of the network
    model_params = filter(lambda p: p.requires_grad, model.parameters())   

    # Initialise optimiser
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model_params, lr=args.lr, weight_decay=args.weight_decay
            )
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model_params, lr=args.lr, momentum=args.momentum,
            weight_decay=args.weight_decay
            )
    
    if args.lr_decay_gamma:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, args.lr_decay_step, gamma=args.lr_decay_gamma
            )


    if args.train_cont:
        train_utils.reload_optimizer(
            args.resume_path,optimizer,scheduler
            )
    
    # Get loss function
    loss_function = common_utils.loss_str2func()[args.loss_function]

    for epoch in tqdm(range(start_epoch, args.epochs+1), desc="epoch"):
        print(f"***Epoch #{epoch}")
        train_utils.epoch_pass(
            trainloader,
            model,
            loss_function,
            data_helper,
            train=True,
            optimizer=optimizer,
            scheduler=scheduler,
            lr_decay_gamma=args.lr_decay_gamma,
            use_GPU=args.use_GPU,
            use_multiple_gpu=use_multiple_gpu,
            tensorboard_writer=board_writer,
            is_demo=False,
            epoch=epoch,
            freeze_batchnorm=args.freeze_batch_norm,
            path_to_png=exp_id,
            weight_SUMINS=args.weight_SUMINS,
            weight_FILENAME=args.weight_FILENAME
            )

        if epoch%args.snapshot == 0:
            print("Forward pass testing split")
            _, test_avg_meters = train_utils.epoch_pass(
                testloader,
                model,
                loss_function,
                data_helper,
                train=False,
                optimizer=optimizer,
                scheduler=scheduler,
                lr_decay_gamma=args.lr_decay_gamma,
                use_GPU=args.use_GPU,
                use_multiple_gpu=use_multiple_gpu,
                tensorboard_writer=board_writer,
                is_demo=False,
                epoch=epoch,
                freeze_batchnorm=args.freeze_batch_norm,
                path_to_png=exp_id,
                weight_SUMINS=args.weight_SUMINS,
                weight_FILENAME=args.weight_FILENAME
                )
            
            test_dict = {
                meter_name: meter.avg
                for meter_name, meter in test_avg_meters.average_meters.items()
                }
            best_metric = list(test_dict.keys())[0]
            if best_score is None:
                best_score = test_dict[best_metric]
            is_best = test_dict[best_metric] < best_score
            best_score = min(test_dict[best_metric], best_score)


            if args.experiment_tag != 'debug':
                model_utils.save_checkpoint(
                    {
                        "epoch": epoch, 
                        "network": args.model,
                        "state_dict": model.module.state_dict() if use_multiple_gpu else model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler,
                    },
                    is_best=is_best,
                    checkpoint=exp_id,
                    snapshot=args.snapshot
                    )

    board_writer.close()


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