# -*- coding: utf-8 -*-

"""
Test.
"""

# Standard lib imports
import os
import numpy as np
import os.path as osp


# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
from torch.utils.data.distributed import DistributedSampler

# Local imports
from utils.meters import average_accuracy
from utils import compute_mask_IoU
from models.panoptic_narrative_grounding import PanopticNarrativeGroundingBaseline
from data import PanopticNarrativeGroundingDataset
import utils.distributed as distributed

from sklearn.metrics import accuracy_score

# Other imports
from tqdm import tqdm

@torch.no_grad()
def perform_test(test_loader, model, cfg):
    """
    Perform testing. 
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        cfg (CfgNode): configs. Details can be found in
            /config/defaults.py
    """
    if distributed.is_master_proc():
        print('-' * 89)
        print('Evaluation on test set')
        print('-' * 89)

    # Enable eval mode.
    model.eval()
    sigmoid = nn.Sigmoid()

    # Use cuda if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    instances_iou = []
    singulars_iou = []
    plurals_iou = []
    things_iou = []
    stuff_iou = []
    pbar = tqdm(total=len(test_loader))
    for (batch_idx, (caption, boxes, instances, mask_features, annotations, ann_types, ann_categories, noun_phrases, grounding_instances, image_info)) in enumerate(test_loader):
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
        
        if not cfg.TEST.ORACLE:
            # Perform the forward pass
            scores = model(caption, boxes, mask_features, noun_phrases)

            if cfg.NUM_GPUS > 1:
                scores = distributed.all_gather(
                    [scores]
                )
        if cfg.NUM_GPUS > 1:
            grounding_instances, annotations, ann_types, instances = distributed.all_gather(
                [grounding_instances, annotations, ann_types, instances]
            )
            height, width = distributed.all_gather(
                [height, width]
            )

        # Evaluation
        words_mask = ann_types == 1
        if not cfg.TEST.ORACLE:
            scores = torch.bmm(noun_phrases.transpose(1, 2), scores)
            scores = scores / (noun_phrases.sum(dim=1).unsqueeze(dim=2).repeat(1, 1, scores.shape[2]) + 0.0000001)
        
            scores = sigmoid(scores)
            index = torch.argmax(scores, dim=2).cpu().numpy()
        else:
            index = torch.argmax(annotations, dim=2).cpu().numpy()
        predictions = instances[torch.arange(instances.shape[0]).unsqueeze(-1), index]
        predictions = predictions[words_mask]
        targets = grounding_instances[words_mask]
        words_index = words_mask.nonzero()
        
        is_singular = torch.ones([words_index.shape[0]])
        is_thing =  ann_categories[words_mask]

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
        is_singular = torch.cat([is_singular, torch.zeros([plurals_mask.nonzero().shape[0]])])
        is_thing = torch.cat([is_thing, ann_categories[plurals_mask]])

        batch_iou = []
        for p, t, (i, _), s, th in zip(predictions, targets, words_index, is_singular, is_thing):
            mask_transform = Resize((int(height[i].cpu().item()), int(width[i].cpu().item())))
            p = mask_transform(p.unsqueeze(dim=0)).squeeze()
            t = mask_transform(t.unsqueeze(dim=0)).squeeze()
            _, _, instance_iou = compute_mask_IoU(p, t)
            instances_iou.append(instance_iou.cpu().item())
            if s == 1:
                singulars_iou.append(instance_iou.cpu().item())
            else:
                plurals_iou.append(instance_iou.cpu().item())
            if th == 1:
                things_iou.append(instance_iou.cpu().item())
            else:
                stuff_iou.append(instance_iou.cpu().item())
            batch_iou.append(instance_iou.cpu().item())
    
        if distributed.is_master_proc():
            pbar.update(1)
            if batch_idx % cfg.LOG_PERIOD == 0:
                tqdm.write('acc@0.5: {:.5f} | AA: {:.5f}'.format(accuracy_score(np.ones([len(instances_iou)]), np.array(instances_iou) > 0.5), average_accuracy(instances_iou))) 
    
    # Final evaluation metrics
    AA = average_accuracy(instances_iou, save_fig=True, output_dir=cfg.OUTPUT_DIR, filename='overall')
    AA_singulars = average_accuracy(singulars_iou, save_fig=True, output_dir=cfg.OUTPUT_DIR, filename='singulars')
    AA_plurals = average_accuracy(plurals_iou, save_fig=True, output_dir=cfg.OUTPUT_DIR, filename='plurals')
    AA_things = average_accuracy(things_iou, save_fig=True, output_dir=cfg.OUTPUT_DIR, filename='things')
    AA_stuff = average_accuracy(stuff_iou, save_fig=True, output_dir=cfg.OUTPUT_DIR, filename='stuff')
    accuracy = accuracy_score(np.ones([len(instances_iou)]), np.array(instances_iou) > 0.5)
    if distributed.is_master_proc():
        print('| final acc@0.5: {:.5f} | final AA: {:.5f} |  AA singulars: {:.5f} | AA plurals: {:.5f} | AA things: {:.5f} | AA stuff: {:.5f} |'.format(
                                               accuracy,
                                               AA,
                                               AA_singulars,
                                               AA_plurals,
                                               AA_things,
                                               AA_stuff))
    return AA


def test(cfg):
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    distributed.init_distributed_training(cfg)

    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Print config.
    if distributed.is_master_proc():
        print("Test with config:")
        print(cfg)

    # Build the model and print model statistics.
    # Use cuda if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Construct the model
    model = PanopticNarrativeGroundingBaseline(cfg, device=device)
    # Determine the GPU used by the current process
    cur_device = torch.cuda.current_device()
    # Transfer the model to the current GPU device
    model = model.cuda(device=cur_device)
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

    # Load a checkpoint to test if applicable.
    checkpoint_path = osp.join(cfg.OUTPUT_DIR, 'model_final.pth')
    if cfg.TEST.CHECKPOINT_FILE_PATH != "":
        checkpoint_path = cfg.TEST.CHECKPOINT_FILE_PATH
    if osp.exists(checkpoint_path):
        if distributed.is_master_proc():
            print('Loading model from: {0}'.format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if cfg.NUM_GPUS > 1:
            model.module.load_state_dict(checkpoint['model_state'])
        else:
            model.load_state_dict(checkpoint['model_state'])
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH != "":
        # If no checkpoint found in TEST.CHECKPOINT_FILE_PATH or in the current
        # checkpoint folder, try to load checkpoint from
        # TRAIN.CHECKPOINT_FILE_PATH and test it.
        checkpoint_path = cfg.TRAIN.CHECKPOINT_FILE_PATH
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if cfg.NUM_GPUS > 1:
            model.module.load_state_dict(checkpoint['model_state'])
        else:
            model.load_state_dict(checkpoint['model_state'])
    else:
        if distributed.is_master_proc():
            print("Testing with random initialization. Only for debugging.")

    # Create testing loaders.
    test_dataset = PanopticNarrativeGroundingDataset(cfg, cfg.DATA.VAL_SPLIT, train=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(cfg.TRAIN.BATCH_SIZE / max(1, cfg.NUM_GPUS)),
        shuffle=False,
        sampler=(DistributedSampler(test_dataset) if cfg.NUM_GPUS > 1 else None),
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY
    )
    
    if distributed.is_master_proc():
        print("Testing model for {} iterations".format(len(test_loader)))

    # Perform test on the entire dataset.
    perform_test(test_loader, model, cfg)