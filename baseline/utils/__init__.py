# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c)
#
# Licensed under the terms of the MIT License
# (see LICENSE for details)
# -----------------------------------------------------------------------------

"""Misc data and other helping utillites."""

# from .transforms import ResizeImage, ResizeAnnotation, ToFloat, AddImageBorder

# ResizeImage
# ResizeAnnotation
# ToFloat
# AddImageBorder


def compute_mask_IoU(masks, target):
    assert target.shape[-2:] == masks.shape[-2:]
    temp = masks * target
    intersection = temp.sum()
    union = ((masks + target) - temp).sum()
    return intersection, union, intersection/union

def compute_box_IoU(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[0] + a[2], b[0] + b[2])
    y2 = min(a[1] + a[3], b[1] + b[3])
    intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    area_a = a[2] * a[3]
    area_b = b[2] * b[3]
    union = area_a + area_b - intersection
    return intersection, union, intersection/union


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
