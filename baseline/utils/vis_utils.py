import os
import re
import glob
import numpy as np
import os.path as osp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pylab
import torch

from tqdm import tqdm
from skimage import io
from skimage.color import rgb2grey
from skimage.transform import resize

import pdb

def uncrop_image(img, org_h=1080,
               org_w=1920, h_start=28,
               w_start=320, h=1024, w=1280):
    h_end, w_end = h + h_start, w + w_start
    large_img = torch.zeros(org_h, org_w, dtype=torch.uint8)
    large_img[h_start:h_end, w_start:w_end] = img
    return large_img

def crop_image(image, h_start=28, w_start=320, h=1024, w=1280):
    image = image[h_start : h_start + h, w_start : w_start + w]
    return image

class VisualizeInstruments(object):
    """class with visualization utils"""
    def __init__(self, save_dir):
        super(VisualizeInstruments, self).__init__()
        self.save_dir = save_dir
        if not osp.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.colors_mask = [[73, 218, 154], [235, 117, 50],
                            [247, 208, 56], [163, 224, 71],
                            [52, 187, 230], [67, 85, 219],
                            [209, 58, 231], [244, 58, 74],
                            [244, 58, 200]]
    def color_ann(self, ann, ax):
        these_cats = torch.unique(ann)
        these_cats = these_cats[1:] # remove background
        for cat in these_cats:
            mask = (ann == cat).cpu().float()
            colored_ann = np.ones((ann.shape[0], ann.shape[1], 3))
            color_ann = np.array(self.colors_mask[cat-1])/255
            for i in range(3):
                colored_ann[:,:,i] = color_ann[i]
            ax.imshow(np.dstack((colored_ann, mask.cpu()*0.4)))

    def visualize_cand(self, cand, image, ann,
                        pred, probs, target, 
                        old_preds, filename='name.png',
                       figsize=(7.5,2)):
        """
        cand: predicted candidate mask
        image: original image
        pred: predicted class for cand
        probs: probabilities for the topk classes
        target: target class for cand
        old_preds: topk classes predicted by mrcnn
        """
        plt.figure(figsize=figsize)
        # show image in grayscale with colored ann
        ax1 = plt.subplot(121)
        plt.imshow(image, cmap = "gray")
        plt.axis('off')
        ax1.set_title('target: {} old_preds: {}'.format(target, old_preds.cpu().numpy()))
        plt.gca()
        self.color_ann(ann, ax1)

        # show colored cand on top of image
        colored_pred = np.ones((cand.shape[0], cand.shape[1], 3))
        color_pred = np.array(self.colors_mask[pred-1])/255
        for i in range(3):
            colored_pred[:,:,i] = color_pred[i]
        #plt.imshow(np.dstack((colored_pred, mask.cpu()*0.4)))
        
        ax2 = plt.subplot(122)
        plt.imshow(image, cmap = "gray")
        plt.axis('off')
        ax2.set_title('pred: {} probs: {}'.format(pred, probs.cpu().numpy()))
        plt.gca()
        plt.imshow(np.dstack((colored_pred, cand.cpu()*0.4) ))
        #plt.subplots_adjust(left=0, right=1, bottom=0,
        #                    top=1, wspace=0, hspace=0)
        #plt.margins(0,0)
        #ax.annotate(s = 'pred: {}\n'
        #                'probs: {}\n'
        #                'target: {}\n'
        #                'old_preds: {}'.format(pred, probs, target, old_preds),
        #            xy=(0, 0), 
        #            xytext=(0, 0), 
        #            va='top',
        #            ha='left',
        #            fontsize = 15,
        #            bbox=dict(facecolor='white', alpha=1),)
        plt.savefig(osp.join(self.save_dir, filename), pad_inches=0)
        plt.close()

    def visualize_pred(self, ann, pred, image,
                       filename='name.png',
                       figsize=(7.5,2)):
        """
        pred: prediction with labels
        image: original image
        ann: original annotation with labels for image
        """
        plt.figure(figsize=figsize)
        # show image in grayscale with colored ann
        ax1 = plt.subplot(121)
        plt.imshow(image, cmap = "gray")
        plt.axis('off')
        ax1.set_title('Mask R-CNN')
        plt.gca()
        self.color_ann(ann, ax1)

        # show image in grayscale with colored pred
        ax2 = plt.subplot(122)
        plt.imshow(image, cmap = "gray")
        plt.axis('off')
        ax2.set_title('Our prediction')
        plt.gca()
        self.color_ann(pred,ax2)
        plt.savefig(osp.join(self.save_dir, filename), pad_inches=0)
        plt.close()

    def visualize_confusion_matrix(self, cm,cmap=plt.cm.viridis, 
                                   class_names=['bipolar forceps',
                                                'prograsp forceps',
                                                'large needle driver',
                                                'vessel sealer',
                                                'grasping retractor',
                                                'monopolar curved scissors',
                                                'ultrasound probe']):

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=class_names, yticklabels=class_names,
               title='Confusion matrix',
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        plt.savefig(osp.join(self.save_dir, 'confusion_matrix.png'), pad_inches=0)
        plt.close()
    