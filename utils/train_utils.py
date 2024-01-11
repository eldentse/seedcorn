import random
import torch
import numpy as np
import os
import warnings
import traceback
from tqdm import tqdm

from utils.model_utils import load_checkpoint
from utils.eval_utils import ClassificationEvaluator, AverageMeters
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


def epoch_pass(
    loader,
    model,
    loss_function,
    data_helper,
    train,
    optimizer,
    scheduler,
    lr_decay_gamma,
    use_GPU,
    use_multiple_gpu,
    tensorboard_writer,
    is_demo,
    epoch,
    freeze_batchnorm,
    path_to_png,
    weight_SUMINS,
    weight_FILENAME
):
    
    # Switch to correct model mode
    if train:
        prefix = "train"
        if freeze_batchnorm:
            model.eval()
        else:
            model.train()
    else:
        prefix = "test"
    
    if data_helper.multi_task:
        if BaseQueries.SUMINS in data_helper.queries:
            SUMINS_evaluator =  ClassificationEvaluator(data_helper.label_info['SUMINS'])
        if BaseQueries.FILE_NAME in data_helper.queries:
            FILE_NAME_evaluator =  ClassificationEvaluator(data_helper.label_info['FILE_NAME'])
    else:
        evaluator =  ClassificationEvaluator(data_helper.label_info) 
    

    avg_meters = AverageMeters() # TODO
     
     
    for _, batch in enumerate(tqdm(loader)): 
        losses = {}

        # Get inputs 
        if data_helper.multi_task:
            inputs = batch['inputs']
            targets = {}
            if BaseQueries.SUMINS in data_helper.queries:
                targets['SUMINS'] = batch['SUMINS']
                targets['SUMINS'] = targets['SUMINS'].cuda() if use_GPU else targets['SUMINS']
            if BaseQueries.FILE_NAME in data_helper.queries:
                targets['FILE_NAME'] = batch['FILE_NAME']
                targets['FILE_NAME'] = targets['FILE_NAME'].cuda() if use_GPU else targets['FILE_NAME']
        else:
            inputs, targets = batch
            targets = targets.cuda() if use_GPU else targets
        inputs = inputs.cuda() if use_GPU else inputs
    
        
        if train:
            # Compute loss
            outputs = model(inputs) # TODO final version can entengale outputs and loss once target task is confirmed
            if data_helper.multi_task:
                SUMINS_loss = loss_function(outputs['SUMINS'], targets['SUMINS'])
                FILEAME_loss = loss_function(outputs['FILE_NAME'], targets['FILE_NAME'])

                losses['SUMINS_loss'] = SUMINS_loss
                losses['FILEAME_loss'] = FILEAME_loss
                loss = weight_SUMINS*SUMINS_loss + weight_FILENAME*FILEAME_loss
            else:
                loss = loss_function(outputs, targets) # TODO use dict if we have multiple loss terms in the future
                for items in data_helper.queries:
                    temp = items.value + '_loss'
                    locals()[temp] = {temp: loss}
                    losses.update(locals()[temp])
        else:
            with torch.no_grad():
                model.eval()
                # Compute loss
                outputs = model(inputs) 
                if not data_helper.classification_task:
                    outputs = data_helper.y_sc.inverse_transform(outputs.cpu())
                    targets = data_helper.y_sc.inverse_transform(targets.cpu())
                    loss = loss_function(torch.from_numpy(outputs), torch.from_numpy(targets))
                else:
                    if data_helper.multi_task:
                        SUMINS_loss = loss_function(outputs['SUMINS'], targets['SUMINS'])
                        FILEAME_loss = loss_function(outputs['FILE_NAME'], targets['FILE_NAME'])

                        losses['SUMINS_loss'] = SUMINS_loss
                        losses['FILEAME_loss'] = FILEAME_loss
                        loss = weight_SUMINS*SUMINS_loss + weight_FILENAME*FILEAME_loss
                    else:
                        loss = loss_function(outputs, targets)

                if not data_helper.multi_task:    
                    for items in data_helper.queries:
                        temp = items.value + '_loss'
                        locals()[temp] = {temp: loss}
                        losses.update(locals()[temp])

        if use_multiple_gpu:
            loss=loss.mean()
            for k,v in losses.items():
                if v is not None:
                    v=v.mean() 
        
        if train:
            if torch.isnan(loss):
                raise ValueError(f"Loss made of {losses} became nan!")
            optimizer.zero_grad()                
            loss.backward()            
            optimizer.step()
 
        # Add loss value into average meters 
        # which tracks running average within a batch
        for loss_name, loss_val in losses.items():
            if loss_val is not None:
                avg_meters.add_loss_value(loss_name, loss_val.mean().item())
                
        if not train:
            if data_helper.classification_task and not data_helper.multi_task:
                evaluator.feed(gt_labels=targets,
                            pred_labels=outputs) 
            else:
                if BaseQueries.SUMINS in data_helper.queries:
                    SUMINS_evaluator.feed(gt_labels=targets['SUMINS'],
                                          pred_labels=outputs['SUMINS']) 
                if BaseQueries.FILE_NAME in data_helper.queries:
                    FILE_NAME_evaluator.feed(gt_labels=targets['FILE_NAME'],
                                             pred_labels=outputs['FILE_NAME']) 


    save_dict = {}
    if train and lr_decay_gamma and scheduler is not None:
        save_dict['learning_rate']=scheduler.get_last_lr()[0]
        scheduler.step()
   
    
    for loss_name, avg_meter in avg_meters.average_meters.items():
        # Grab tracking average
        loss_val = avg_meter.avg
        save_dict[loss_name] = loss_val
    
    if not train:
        if data_helper.classification_task:
            if data_helper.multi_task:
                save_dict['y_pred_raw_SUMINS'] = SUMINS_evaluator.y_pred_raw
                save_dict['y_pred_raw_FILE_NAME'] = FILE_NAME_evaluator.y_pred_raw
            else:
                save_dict['y_pred_raw'] = evaluator.y_pred_raw # TODO
        else: 
            pass # TODO
        if data_helper.multi_task:
            sampled_outputs = data_helper.sampled_forward_pass_MT(model, use_GPU)
        else:
            sampled_outputs = data_helper.sampled_forward_pass(model, use_GPU)

    
    if not tensorboard_writer is None: 
        for k,v in save_dict.items():
            if k in losses.keys() or k in ['learning_rate','total_loss']:
                print(prefix+'/'+k,v,epoch)
                tensorboard_writer.add_scalar(prefix+'/'+k, v, epoch)
        
        # Add eval results to tensorboard
        if not train:
            if data_helper.classification_task:
                if data_helper.multi_task:
                    SUMINS_classification_result, SUMINS_report, SUMINS_count_matrix = SUMINS_evaluator.get_results()
                    tensorboard_writer.add_image("SUMINS Confusion matrix",
                                                SUMINS_evaluator.draw(
                                                    path = path_to_png+'matrix_'+str(epoch)
                                                    ),
                                                epoch) # TODO
                    
                    tensorboard_writer.add_text("SUMINS Classification report",
                                                SUMINS_report,
                                                epoch)
                    tensorboard_writer.add_text("SUMINS Count matrix",
                                                SUMINS_count_matrix,
                                                epoch)
                    
                    for k,v in SUMINS_classification_result.items():
                        tensorboard_writer.add_scalar(prefix+'/'+'SUMINS_classification/'+k, v, epoch)
                        save_dict['SUMINS_classification_'+k] = v

                    FILE_NAME_classification_result, FILE_NAME_report, FILE_NAME_count_matrix = FILE_NAME_evaluator.get_results()
                    tensorboard_writer.add_image("FILE_NAME Confusion matrix",
                                                FILE_NAME_evaluator.draw(
                                                    path = path_to_png+'matrix_'+str(epoch)
                                                    ),
                                                epoch) # TODO
                    
                    tensorboard_writer.add_text("FILE_NAME Classification report",
                                                FILE_NAME_report,
                                                epoch)
                    tensorboard_writer.add_text("FILE_NAME Count matrix",
                                                FILE_NAME_count_matrix,
                                                epoch)
                    
                    for k,v in FILE_NAME_classification_result.items():
                        tensorboard_writer.add_scalar(prefix+'/'+'FILE_NAME_classification/'+k, v, epoch)
                        save_dict['FILE_NAME_classification_'+k] = v
                else:
                    classification_result, report, count_matrix = evaluator.get_results()
                    tensorboard_writer.add_image("Confusion matrix",
                                                evaluator.draw(
                                                    path = path_to_png+'matrix_'+str(epoch)
                                                    ),
                                                epoch) # TODO
                    
                    tensorboard_writer.add_text("Classification report",
                                                report,
                                                epoch)
                    tensorboard_writer.add_text("Count matrix",
                                                count_matrix,
                                                epoch)
                    
                    for k,v in classification_result.items():
                        tensorboard_writer.add_scalar(prefix+'/'+'classification/'+k, v, epoch)
                        save_dict['classification_'+k] = v
                    
                    print("Classification performance, TP: {:d}, \
                        Total: {:d}, \
                        Accuracy {:.2f}".format(
                            int(save_dict["classification_total_tp"]),
                            int(save_dict["classification_total_samples"]),
                            save_dict["classification_accuracy"]*100))
            else:
                pass #TODO

            # tensorboard_writer.add_text("Sampled data",
            #                             data_helper.array_to_markdown(sampled_outputs),
            #                             epoch)
    

    return save_dict, avg_meters

