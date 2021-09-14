# -*- coding: utf-8 -*-

"""
Panoptic Narrative Grounding Baseline Network PyTorch implementation.
"""

import torch
import torch.nn as nn

from .encoder_bert import BertEncoder

class PanopticNarrativeGroundingBaseline(nn.Module):
    def __init__(self, cfg,
                 device="cpu"):
        super().__init__()
        self.cfg = cfg
        self.device = device

        # Define the network
        self.bert_encoder = BertEncoder(
            cfg,
        )
        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, sent, pos, feat, noun_phrases):
        """
        :param feat: b, 2, o, f
        :param pos:  b, 2, o, 4
        :param sent: b, (string)
        :param noun_phrases: b, l, np
        :return:
        """
        output_lang, output_img, _ = self.bert_encoder(sent, (feat, pos), noun_phrases)
        output_img = output_img.permute([0, 2, 1])
        output = torch.matmul(output_lang, output_img)

        return output
