# -*- coding: utf-8 -*-

"""
Generic Image Transform utillities.
"""

import cv2
import torch
import numpy as np
from collections import Iterable

import torch.nn.functional as F

cv2.setNumThreads(0)


def resize_factor(img, factor):
    im_h, im_w = img.shape[-2:]
    resized_h = int(np.round(im_h * factor))
    resized_w = int(np.round(im_w * factor))
    img = img.squeeze(dim=0)
    out = F.upsample(
        img.unsqueeze(0).unsqueeze(0),
        size=(resized_h, resized_w),
        mode="bilinear",
        align_corners=True,
    ).squeeze(dim=0)
    return out


class ResizePad:
    """
    Resize and pad an image to given size.
    """

    def __init__(self, size):
        if not isinstance(size, (int, Iterable)):
            raise TypeError("Got inappropriate size arg: {}".format(size))

        self.h, self.w = size

    def __call__(self, img):
        h, w = img.shape[:2]
        scale = min(self.h / h, self.w / w)
        resized_h = int(np.round(h * scale))
        resized_w = int(np.round(w * scale))
        pad_h = int(np.floor(self.h - resized_h) / 2)
        pad_w = int(np.floor(self.w - resized_w) / 2)

        resized_img = cv2.resize(img, (resized_w, resized_h))

        # if img.ndim > 2:
        if img.ndim > 2:
            new_img = np.zeros(
                (self.h, self.w, img.shape[-1]), dtype=resized_img.dtype
            )
        else:
            resized_img = np.expand_dims(resized_img, -1)
            new_img = np.zeros((self.h, self.w, 1), dtype=resized_img.dtype)
        new_img[
            pad_h : pad_h + resized_h, pad_w : pad_w + resized_w, ...
        ] = resized_img
        return new_img


class CropResize:
    """Remove padding and resize image to its original size."""

    def __call__(self, img, size):
        if not isinstance(size, (int, Iterable)):
            raise TypeError("Got inappropriate size arg: {}".format(size))
        im_h, im_w = img.data.shape[:2]
        input_h, input_w = size
        scale = max(input_h / im_h, input_w / im_w)
        # scale = torch.Tensor([[input_h / im_h, input_w / im_w]]).max()
        resized_h = int(np.round(im_h * scale))
        # resized_h = torch.round(im_h * scale)
        resized_w = int(np.round(im_w * scale))
        # resized_w = torch.round(im_w * scale)
        crop_h = int(np.floor(resized_h - input_h) / 2)
        # crop_h = torch.floor(resized_h - input_h) // 2
        crop_w = int(np.floor(resized_w - input_w) / 2)
        # crop_w = torch.floor(resized_w - input_w) // 2
        # resized_img = cv2.resize(img, (resized_w, resized_h))
        resized_img = F.upsample(
            img.unsqueeze(0).unsqueeze(0),
            size=(resized_h, resized_w),
            mode="bilinear",
            align_corners=True,
        )

        resized_img = resized_img.squeeze().unsqueeze(0)

        return resized_img[
            0, crop_h : crop_h + input_h, crop_w : crop_w + input_w
        ]


class ResizeImage:
    """Resize the largest of the sides of the image to a given size"""

    def __init__(self, size):
        if not isinstance(size, (int, Iterable)):
            raise TypeError("Got inappropriate size arg: {}".format(size))

        self.size = size

    def __call__(self, img):
        im_h, im_w = img.shape[-2:]
        scale = min(self.size / im_h, self.size / im_w)
        resized_h = int(np.round(im_h * scale))
        resized_w = int(np.round(im_w * scale))
        out = (
            F.upsample(
                img.unsqueeze(0),
                size=(resized_h, resized_w),
                mode="bilinear",
                align_corners=True,
            )
            .squeeze(dim=0)
            .data
        )
        return out


class ResizeAnnotation:
    """Resize the largest of the sides of the annotation to a given size"""

    def __init__(self, size):
        if not isinstance(size, (int, Iterable)):
            raise TypeError("Got inappropriate size arg: {}".format(size))

        self.size = size

    def __call__(self, img):
        im_h, im_w = img.shape[-2:]
        scale = min(self.size / im_h, self.size / im_w)
        resized_h = int(np.round(im_h * scale))
        resized_w = int(np.round(im_w * scale))
        out = F.upsample(
            img.unsqueeze(0),
            size=(resized_h, resized_w),
            mode="bilinear",
            align_corners=True,
        )
        out = out == 1
        out = out.squeeze(dim=0)
        return out


class ResizeFactor:
    def __init__(self, factor):
        if not isinstance(factor, (int, float)):
            raise TypeError(
                "Got inappropriate size arg: {}, "
                "it should be a number".format(factor)
            )
        self.factor = factor

    def __call__(self, img):
        return resize_factor(img, self.factor)


class ToNumpy:
    """Transform an torch.*Tensor to an numpy ndarray."""

    def __call__(self, x):
        return x.numpy()


class Identity:
    """Do nothing."""

    def __call__(self, x):
        return x


class ToFloat:
    """Do nothing."""

    def __call__(self, x):
        return x.float()


class AddImageBorder:
    def __init__(self, h_start=28, w_start=320, h=1080, w=1920):
        self.h_start = h_start
        self.w_start = w_start
        self.h = h
        self.w = w

    def __call__(self, img):
        if np.ndims(img) > 2:
            image = np.zeros([self.h, self.w, img.size(-1)])
        else:
            image = np.zeros([self.h, self.w])
        image[self.h_start : self.h_start + img.size(0),
              self.w_start : self.w_start + img.size(1)] = img
        return image


class RandomHorizontalFlip:
    def __init__(self, min_prob=0.5):
        self.min_prob = min_prob

    def __call__(self, img, ann, cands):
        flipped = False
        flip_prob = torch.rand(1)
        if flip_prob > self.min_prob:
            assert (((len(img.shape) == 3 and len(ann.shape) == 2)
                     and len(cands.shape) == 3))
            img = torch.flip(img, (2,))  # dim=2 flips left right 3d mat
            ann = torch.flip(ann, (1,))  # dim=1 flips left right 2d mat
            cands = torch.flip(cands, (2,))  # dim=2 flips left right 3d mat
            flipped = True
        return img, ann, cands, flipped
