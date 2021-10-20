import os
import json
import torch
import torch.nn.functional as F
import os.path as osp
from skimage import io

import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms import Resize

from utils import compute_box_IoU

import matplotlib.pyplot as plt
from tqdm import tqdm


class PanopticNarrativeGroundingDataset(Dataset):
    """Panoptic Narrative Grounding dataset."""

    def __init__(self, cfg, split, train=True):
        """
        Args:
            Args:
            cfg (CfgNode): configs.
            train (bool):
        """
        self.cfg = cfg
        self.train = train
        self.split = split

        self.boxes = {}
        self.mask_features_path = osp.join(
            cfg.DATA.PATH_TO_FEATURES_DIR, split, "mask_features"
        )
        self.sem_seg_features_path = osp.join(
            cfg.DATA.PATH_TO_FEATURES_DIR, split, "sem_seg_features"
        )
        self.panoptic_pred_path = osp.join(
            cfg.DATA.PATH_TO_FEATURES_DIR, split, "panoptic_seg_predictions"
        )

        self.semantic_pool = nn.AdaptiveAvgPool2d((224, 224))
        self.mask_transform = Resize((256, 256))

        self.ann_dir = osp.join(cfg.DATA.PATH_TO_DATA_DIR, "annotations")
        self.panoptic = self.load_json(
            osp.join(self.ann_dir, "panoptic_{:s}.json".format(split))
        )
        self.images = self.panoptic["images"]
        self.images = {i["id"]: i for i in self.images}
        self.panoptic_anns = self.panoptic["annotations"]
        self.panoptic_anns = {a["image_id"]: a for a in self.panoptic_anns}
        if not osp.exists(
            osp.join(self.ann_dir, "png_coco_{:s}_dataloader.json".format(split),)
        ):
            print("No such a dataset")
        else:
            self.panoptic_narrative_grounding = self.load_json(
                osp.join(self.ann_dir, "png_coco_{:s}_dataloader.json".format(split),)
            )
        self.panoptic_narrative_grounding = [
            ln
            for ln in self.panoptic_narrative_grounding
            if (
                torch.tensor([item for sublist in ln["labels"] for item in sublist])
                != -2
            ).any()
        ]

    ## General helper functions
    def load_json(self, filename):
        with open(filename, "r") as f:
            data = json.load(f)
        return data

    def save_json(self, filename, data):
        with open(filename, "w") as f:
            json.dump(data, f)

    def load_jsonl(self, filename):
        with open(filename, "r") as f:
            data = [json.loads(l) for l in list(f)]
        return data

    def inside_image(self, coord):
        coord = max(coord, 0)
        coord = min(coord, 1)
        return coord

    def __len__(self):
        return len(self.panoptic_narrative_grounding)

    def __getitem__(self, idx):
        if not self.train:
            num_preds = self.cfg.TEST.NUM_BOXES
        else:
            num_preds = self.cfg.MODEL.NUM_BOXES
        localized_narrative = self.panoptic_narrative_grounding[idx]
        caption = localized_narrative["caption"]
        image_id = int(localized_narrative["image_id"])
        image_info = self.images[image_id]

        mask_features = torch.zeros([num_preds, 256 * 14 * 14])
        img_mask_features = torch.load(
            osp.join(self.mask_features_path, "{}.pth".format(image_id)),
            map_location=torch.device("cpu"),
        )
        num_instances = img_mask_features.shape[0]
        if num_instances > 0:
            mask_features[:num_instances, :] = img_mask_features.view(num_instances, -1)

        img_sem_seg_features = torch.load(
            osp.join(self.sem_seg_features_path, "{}.pth".format(image_id)),
            map_location=torch.device("cpu"),
        )
        num_labels = img_sem_seg_features.shape[0]
        if num_labels > 0:
            img_sem_seg_features = self.semantic_pool(img_sem_seg_features)
            mask_features[
                num_instances : num_instances + num_labels, :
            ] = img_sem_seg_features.view(num_labels, -1)

        annotations = torch.zeros(
            [self.cfg.MODEL.MAX_SEQUENCE_LENGTH, num_preds]
        ).long()
        ann_types = torch.zeros([self.cfg.MODEL.MAX_SEQUENCE_LENGTH]).long()
        for i, l in enumerate(localized_narrative["labels"]):
            l = torch.tensor(l)
            if (l != -2).any():
                annotations[i + 1, l[l > -1]] = 1
                ann_types[i + 1] = 1 if (l != -2).sum() == 1 else 2

        panoptic_pred = io.imread(
            osp.join(self.panoptic_pred_path, "{:012d}.png".format(image_id))
        )[:, :, 0]
        panoptic_pred = torch.tensor(panoptic_pred).long()
        masks = torch.zeros(
            (panoptic_pred.max() + 1, panoptic_pred.shape[0], panoptic_pred.shape[1])
        )
        masks = masks.scatter(0, panoptic_pred.unsqueeze(0), 1).long()
        masks = masks[1:]

        if image_id not in self.boxes:
            boxes = []
            for m in masks:
                coords = m.nonzero()
                x1 = coords[:, 1].min()
                x2 = coords[:, 1].max()
                y1 = coords[:, 0].min()
                y2 = coords[:, 0].max()
                boxes.append([x1, y1, x2 - x1, y2 - y1])
            self.boxes[image_id] = torch.tensor(boxes)
        boxes = torch.zeros([num_preds, 4])
        if self.boxes[image_id].shape[0] > 0:
            boxes[: self.boxes[image_id].shape[0], :] = self.boxes[image_id]

        tmp, counts = ann_types[1:].unique_consecutive(return_counts=True)
        counts = torch.cat([torch.tensor([0]), counts], dim=0)
        counts = counts.cumsum(dim=0)[:-1]
        counts = counts[tmp != 0]

        tmp = tmp[tmp != 0]
        ann_types = torch.zeros([self.cfg.TEST.MAX_NOUN_PHRASES]).long()
        if tmp.shape[0] > 0:
            ann_types[: tmp.shape[0]] = tmp

        tmp = annotations[1:][counts, :]
        annotations = torch.zeros([self.cfg.TEST.MAX_NOUN_PHRASES, num_preds]).long()
        if tmp.shape[0] > 0:
            annotations[: tmp.shape[0], :] = tmp

        noun_phrases = torch.zeros(
            [self.cfg.MODEL.MAX_SEQUENCE_LENGTH, self.cfg.TEST.MAX_NOUN_PHRASES]
        ).float()
        noun_indices = torch.tensor(localized_narrative["noun_vector"])
        noun_phrases[
            noun_indices.nonzero().squeeze() + 1, noun_indices[noun_indices != 0] - 1
        ] = 1

        if self.train:
            return caption, boxes, mask_features, annotations, ann_types, noun_phrases

        instances = torch.zeros([num_preds, 256, 256])
        if masks.shape[0] > 0:
            instances[: masks.shape[0], :] = self.mask_transform(masks)

        ann_categories = torch.zeros([self.cfg.MODEL.MAX_SEQUENCE_LENGTH]).long()
        panoptic_ann = self.panoptic_anns[image_id]
        panoptic_segm = io.imread(
            osp.join(
                self.ann_dir,
                "panoptic_segmentation",
                self.split,
                "{:012d}.png".format(image_id),
            )
        )
        panoptic_segm = (
            panoptic_segm[:, :, 0]
            + panoptic_segm[:, :, 1] * 256
            + panoptic_segm[:, :, 2] * 256 ** 2
        )
        grounding_instances = torch.zeros(
            [self.cfg.MODEL.MAX_SEQUENCE_LENGTH, 256, 256]
        )
        for i, bbox in enumerate(localized_narrative["boxes"]):
            for b in bbox:
                if b != [0] * 4:
                    segment_info = [
                        s for s in panoptic_ann["segments_info"] if s["bbox"] == b
                    ][0]
                    segment_cat = [
                        c
                        for c in self.panoptic["categories"]
                        if c["id"] == segment_info["category_id"]
                    ][0]
                    instance = torch.zeros([image_info["height"], image_info["width"]])
                    instance[panoptic_segm == segment_info["id"]] = 1
                    grounding_instances[i + 1, :] += self.mask_transform(
                        instance.unsqueeze(dim=0)
                    ).squeeze()
                    ann_categories[i + 1] = 1 if segment_cat["isthing"] else 2

        tmp = ann_categories[1:].unique_consecutive()
        tmp = tmp[tmp != 0]
        ann_categories = torch.zeros([self.cfg.TEST.MAX_NOUN_PHRASES]).long()
        if tmp.shape[0] > 0:
            ann_categories[: tmp.shape[0]] = tmp

        tmp = grounding_instances[1:][counts, :]
        grounding_instances = torch.zeros([self.cfg.TEST.MAX_NOUN_PHRASES, 256, 256])
        if tmp.shape[0] > 0:
            grounding_instances[: tmp.shape[0]] = tmp

        return (
            caption,
            boxes,
            instances,
            mask_features,
            annotations,
            ann_types,
            ann_categories,
            noun_phrases,
            grounding_instances,
            image_info,
        )

