# -*- coding: utf-8 -*-

"""
Train routine.
"""

# Standard lib imports
import os
import time
import numpy as np
import os.path as osp


# PyTorch imports
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
from torch.utils.data.distributed import DistributedSampler

# Local imports
from utils.meters import average_accuracy
from utils import AverageMeter
from utils import compute_mask_IoU
from models.panoptic_narrative_grounding import PanopticNarrativeGroundingBaseline
from data import PanopticNarrativeGroundingDataset
import utils.distributed as distributed

from sklearn.metrics import accuracy_score

# Other imports
import pprint
from tqdm import tqdm


def train_epoch(train_loader, model, optimizer, loss_function, epoch, cfg):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): train loader.
        model (model): the model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        loss_functions (loss): the loss function to optimize.
        epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            PNG/config/defaults.py
    """
    if distributed.is_master_proc():
        print('-' * 89)
        print('Training epoch {:5d}'.format(epoch))
        print('-' * 89)

    # Enable train mode.
    model.train()
    total_loss = AverageMeter()
    epoch_loss_stats = AverageMeter()
    time_stats = AverageMeter()
    loss = 0

    # Use cuda if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    pbar = tqdm(total=len(train_loader))
    for (batch_idx, (caption, boxes, mask_features, annotations, ann_types, noun_phrases)) in enumerate(train_loader):
        boxes = boxes.to(device)
        mask_features = mask_features.to(device)
        annotations = annotations.to(device)
        ann_types = ann_types.to(device)
        noun_phrases = noun_phrases.to(device)
        
        # Perform the forward pass
        start_time = time.time()
        predictions = model(caption, boxes, mask_features, noun_phrases)
    
        predictions = torch.bmm(noun_phrases.transpose(1, 2), predictions)
        predictions = predictions / (noun_phrases.sum(dim=1).unsqueeze(dim=2).repeat(1, 1, predictions.shape[2]) + 0.0000001)

        loss = loss_function(predictions, annotations.float())

        # Perform the backward pass.
        optimizer.zero_grad()
        loss.backward()

        # Update the parameters.
        optimizer.step()

        # Gather all the predictions across all the devices.
        if cfg.NUM_GPUS > 1:
            loss = distributed.all_reduce([loss])[0]
        
        time_stats.update(time.time() - start_time, 1)
        total_loss.update(loss, 1)
        epoch_loss_stats.update(loss, 1)

        if distributed.is_master_proc():
            pbar.update(1)

            if (batch_idx % cfg.LOG_PERIOD == 0):
                elapsed_time = time_stats.avg
                print(' [{:5d}] ({:5d}/{:5d}) | ms/batch {:.4f} |'
                    ' loss {:.6f} | avg loss {:.6f} | lr {:.7f}'.format(
                        epoch, batch_idx, len(train_loader),
                        elapsed_time * 1000, total_loss.avg,
                        epoch_loss_stats.avg,
                        optimizer.param_groups[0]["lr"]))
                total_loss.reset()

            start_time = time.time()
    pbar.close()
    
    # Save checkpoint
    if distributed.is_master_proc():
        checkpoint_path = osp.join(cfg.OUTPUT_DIR, 'checkpoint.pth')
        checkpoint = {
            "epoch": epoch,
            "model_state": model.module.state_dict() if cfg.NUM_GPUS > 1 else model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)

    return epoch_loss_stats.avg

@torch.no_grad()
def evaluate(val_loader, model, epoch, cfg):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        cfg (CfgNode): configs. Details can be found in
            PNG/config/defaults.py
    """
    if distributed.is_master_proc():
        print('-' * 89)
        print('Evaluation on val set epoch {:5d}'.format(epoch))
        print('-' * 89)
    
    # Enable eval mode.
    model.eval()
    sigmoid = nn.Sigmoid()

    # Use cuda if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    instances_iou = []
    pbar = tqdm(total=len(val_loader))
    for (batch_idx, (caption, boxes, instances, mask_features, annotations, ann_types, ann_categories, noun_phrases, grounding_instances, image_info)) in enumerate(val_loader):
        boxes = boxes.to(device)
        instances = instances.to(device)
        mask_features = mask_features.to(device)
        annotations = annotations.to(device)
        ann_types = ann_types.to(device)
        ann_categories = ann_categories.to(device)
        noun_phrases = noun_phrases.to(device)
        grounding_instances = grounding_instances.to(device)
        height = image_info['height'].to(device)
        width = image_info['width'].to(device)
        
        # Perform the forward pass
        scores = model(caption, boxes, mask_features, noun_phrases)

        if cfg.NUM_GPUS > 1:
            scores, grounding_instances, annotations, instances = distributed.all_gather(
                [scores, grounding_instances, annotations, instances]
            )
            height, width = distributed.all_gather(
                [height, width]
            )

        # Evaluation
        words_mask = ann_types == 1

        scores = torch.bmm(noun_phrases.transpose(1, 2), scores)
        scores = scores / (noun_phrases.sum(dim=1).unsqueeze(dim=2).repeat(1, 1, scores.shape[2]) + 0.0000001)
        
        scores = sigmoid(scores)
        index = torch.argmax(scores, dim=2).cpu().numpy()
        predictions = instances[torch.arange(instances.shape[0]).unsqueeze(-1), index]
        predictions = predictions[words_mask]
        targets = grounding_instances[words_mask]
        words_index = words_mask.nonzero()

        if len(predictions.shape) < 3:
            predictions = predictions.unsqueeze(0)
            targets = targets.unsqueeze(0)
            words_index = words_index.unsqueeze(0)

        plurals_mask = ann_types == 2
        for p in plurals_mask.nonzero():
            plural_instance = torch.zeros([predictions.shape[1], predictions.shape[2]]).to(device)
            if not cfg.TEST.ORACLE:
                plural_instances = (scores[p[0], p[1], :] > 0.1).nonzero()
                plural_instances = plural_instances.squeeze() if len(plural_instances.shape) > 1 else plural_instances
            else:
                plural_instances = annotations[p[0], p[1]].nonzero().squeeze()
            if plural_instances.nelement() > 0:
                plural_instance = instances[p[0], plural_instances]
                if len(plural_instance.shape) == 3:
                    plural_instance, _ = plural_instance.max(dim=0)
            predictions = torch.cat([predictions, plural_instance.unsqueeze(0)])
            targets = torch.cat([targets, grounding_instances[p[0], p[1]].unsqueeze(0)])
            words_index = torch.cat([words_index, p.unsqueeze(0)])

        for p, t, (i, _) in zip(predictions, targets, words_index):
            mask_transform = Resize((int(height[i].cpu().item()), int(width[i].cpu().item())))
            p = mask_transform(p.unsqueeze(dim=0)).squeeze()
            t = mask_transform(t.unsqueeze(dim=0)).squeeze()
            _, _, instance_iou = compute_mask_IoU(p, t)
            instances_iou.append(instance_iou.cpu().item())

        if distributed.is_master_proc():
            pbar.update(1)
            if batch_idx % cfg.LOG_PERIOD == 0:
                tqdm.write('acc@0.5: {:.5f} | AA: {:.5f}'.format(accuracy_score(np.ones([len(instances_iou)]), np.array(instances_iou) > 0.5), average_accuracy(instances_iou))) 

    pbar.close()

    # Final evaluation metrics
    AA = average_accuracy(instances_iou)
    accuracy = accuracy_score(np.ones([len(instances_iou)]), np.array(instances_iou) > 0.5)
    if distributed.is_master_proc():
        print('| epoch {:5d} | final acc@0.5: {:.5f} | final AA: {:.5f}  |'.format(
                                               epoch,
                                               accuracy,
                                               AA))
    return AA

def train(cfg):
    """
    Train a model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            /config/defaults.py
    """
    # Set up environment.
    distributed.init_distributed_training(cfg)

    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Print config.
    if distributed.is_master_proc():
        print("Train with config:")
        print(pprint.pformat(cfg))

    # Create train and val loaders.
    train_dataset = PanopticNarrativeGroundingDataset(cfg, cfg.DATA.TRAIN_SPLIT, train=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.TRAIN.BATCH_SIZE / max(1, cfg.NUM_GPUS)),
        shuffle=(False if cfg.NUM_GPUS > 1 else True),
        sampler=(DistributedSampler(train_dataset) if cfg.NUM_GPUS > 1 else None),
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY
    )
    if cfg.DATA.VAL_SPLIT is not None:
        val_dataset = PanopticNarrativeGroundingDataset(cfg, cfg.DATA.VAL_SPLIT, train=False)
        val_loader = DataLoader(
            val_dataset,
            batch_size=(1 if cfg.NUM_GPUS > 1 else cfg.TRAIN.BATCH_SIZE),
            shuffle=False,
            sampler=(DistributedSampler(val_dataset) if cfg.NUM_GPUS > 1 else None),
            num_workers=cfg.DATA_LOADER.NUM_WORKERS,
            pin_memory=cfg.DATA_LOADER.PIN_MEMORY
        )

    # Build the model and print model statistics.
    # Use cuda if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Construct the model
    model = PanopticNarrativeGroundingBaseline(cfg, device=device)
    # Determine the GPU used by the current process
    cur_device = torch.cuda.current_device()
    # Transfer the model to the current GPU device
    model = model.cuda(device=cur_device)
    # Use multi-process data parallel model in the multi-gpu setting
    if cfg.NUM_GPUS > 1:
        # Make model replica operate on the current device
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, device_ids=[cur_device], output_device=cur_device,
            find_unused_parameters=True
        )
    if cfg.LOG_MODEL_INFO and distributed.is_master_proc():
        print("Model:\n{}".format(model))
        print("Params: {:,}".format(np.sum([p.numel() for p in model.parameters()]).item()))
        print("Mem: {:,} MB".format(torch.cuda.max_memory_allocated() / 1024 ** 3))
        print("nvidia-smi")
        os.system("nvidia-smi")

    if cfg.MODEL.BERT_FREEZE:
        if cfg.NUM_GPUS > 1:
            for param in model.module.bert_encoder.model.bert.encoder.layer.parameters():
                param.requires_grad = False
        else:
            for param in model.bert_encoder.model.bert.encoder.layer.parameters():
                param.requires_grad = False

    # Construct the optimizer.
    def optimizer_wrapper(Optim, **kwargs):
        def init_func(model):
            return Optim(model.parameters(), **kwargs)
        return init_func

    optimizers = {
        "adamax": (
            optimizer_wrapper(optim.Adamax, lr=cfg.SOLVER.BASE_LR),
            lambda optim: optim.param_groups[0]["lr"],
        ),
        "adam": (
            optimizer_wrapper(optim.Adam, lr=cfg.SOLVER.BASE_LR),
            lambda optim: optim.param_groups[0]["lr"],
        ),
        "sgd": (
            optimizer_wrapper(optim.SGD, lr=cfg.SOLVER.BASE_LR, momentum=0.9),
            lambda optim: optim.param_groups[0]["lr"],
        ),
    }

    if cfg.SOLVER.OPTIMIZING_METHOD not in optimizers:
        cfg.SOLVER.OPTIMIZING_METHOD = 'adam'
        if distributed.is_master_proc():
            print("{0} not defined in available optimizer list, fallback to Adam")

    optimizer, _ = optimizers[cfg.SOLVER.OPTIMIZING_METHOD]
    optimizer = optimizer(model)
    if distributed.is_master_proc():
        print('optimizer: {}'.format(optimizer))

    # Load a checkpoint to resume training if applicable.
    checkpoint_path = osp.join(cfg.OUTPUT_DIR, 'checkpoint.pth')
    if osp.exists(checkpoint_path):
        if distributed.is_master_proc():
            print('Resuming training: loading model from: {0}'.format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if cfg.NUM_GPUS > 1:
            model.module.load_state_dict(checkpoint['model_state'])
        else:
            model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch'] + 1
        model_final_path = osp.join(cfg.OUTPUT_DIR, 'model_final.pth')
        if osp.exists(model_final_path):
            model_final = torch.load(model_final_path)
            best_val_score = model_final['accuracy']
        else:
            best_val_score = None
    elif osp.exists(cfg.TRAIN.CHECKPOINT_FILE_PATH):
        if distributed.is_master_proc():
            print('Loading model from: {0}'.format(cfg.TRAIN.CHECKPOINT_FILE_PATH))
        checkpoint = torch.load(cfg.TRAIN.CHECKPOINT_FILE_PATH, map_location="cpu")
        if cfg.NUM_GPUS > 1:
            model.module.load_state_dict(checkpoint['model_state'])
        else:
            model.load_state_dict(checkpoint['model_state'])
        start_epoch, best_val_score = 0, None
    else: 
        start_epoch, best_val_score = 0, None

    # Define loss function
    loss_function = nn.BCEWithLogitsLoss()

    if distributed.is_master_proc():
        print('Train begins...')
    if cfg.TRAIN.EVAL_FIRST:
        accuracy = evaluate(val_loader, model, -1, cfg)
        if best_val_score is None or accuracy > best_val_score:
            best_val_score = accuracy
    try:
        # Perform the training loop
        for epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
            epoch_start_time = time.time()
            # Shuffle the dataset
            if cfg.NUM_GPUS > 1:
                train_loader.sampler.set_epoch(epoch)
            # Train for one epoch
            train_loss = train_epoch(train_loader, model, optimizer, loss_function, epoch, cfg)
            accuracy = evaluate(val_loader, model, epoch, cfg) 

            if distributed.is_master_proc():
                # Save best model in the validation set
                if best_val_score is None or accuracy > best_val_score:
                    best_val_score = accuracy
                    model_final_path = osp.join(cfg.OUTPUT_DIR, 'model_final.pth')
                    model_final = {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "accuracy": accuracy
                    }
                    torch.save(model_final, model_final_path)
                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s '
                        '| epoch loss {:.6f} |'.format(
                            epoch, time.time() - epoch_start_time, train_loss))
                print('-' * 89)
    except KeyboardInterrupt:
        if distributed.is_master_proc():
            print('-' * 89)
            print('Exiting from training early')
