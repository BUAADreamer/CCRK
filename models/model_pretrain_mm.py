# Cross-View Language Modeling: Towards Unified Cross-Lingual Cross-Modal Pre-training (https://arxiv.org/abs/2206.00621)
# Github: https://github.com/zengyan-97/CCLM
# Copyright (c) 2022, ByteDance Inc.
# All rights reserved.
import torch
from models import XVLMBase


class UniAlignLM(XVLMBase):
    def __init__(self, config):
        super().__init__(config, load_vision_params=True, load_text_params=True,
                         use_contrastive_loss=True, use_matching_loss=True, use_mlm_loss=True, use_bbox_loss=True,
                         config_text=None)
        self.cclm_easy = True if config['cclm_easy'] == 1 else False
        self.use_tlm = True if ('use_tlm' in config) and config['use_tlm'] else False

    def get_tlm_loss(self, text_ids_masked, text_atts, masked_pos, masked_ids):
        return self.text_encoder(text_ids_masked,
                                 attention_mask=text_atts,
                                 encoder_hidden_states=None,
                                 encoder_attention_mask=None,
                                 return_dict=True,
                                 labels=masked_ids,
                                 masked_pos=masked_pos).loss

    def forward_multimodal(self, image, text_ids, text_atts, text_ids_masked=None, masked_pos=None, masked_ids=None,
                           image_atts=None, idx_to_group_img=None, target_bbox=None, is_image=None,
                           ret_bbox_loss=False):
        image_embeds, image_atts = self.get_vision_embeds(image)
        text_embeds = self.get_text_embeds(text_ids, text_atts)
        # if torch.isnan(text_embeds).any():
        #     torch.set_printoptions(threshold=torch.inf)
        #     print(text_embeds)
        #     print(text_ids, text_atts)
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)
        image_feat, text_feat = self.get_features(image_embeds, text_embeds)
        if self.cclm_easy:
            loss_mitc = self.get_contrastive_loss(image_feat, text_feat)
            loss_hitm = self.get_matching_loss(
                image_embeds, image_atts, image_feat, text_embeds, text_atts, text_feat)
            loss_hmlm = self.get_mlm_loss(text_ids_masked, text_atts, image_embeds, image_atts, masked_pos, masked_ids)
        else:
            loss_mitc = self.get_unialign_contrastive_loss(image_feat, text_feat)
            loss_hitm, text_pos_idx_ls = self.get_unialign_matching_loss(
                image_embeds, image_atts, image_feat, text_embeds, text_atts, text_feat)
            loss_hmlm = self.get_unialign_mlm_loss(
                text_ids_masked, text_atts, image_embeds, image_atts, masked_pos, masked_ids, text_pos_idx_ls)
        if self.vicreg == 1:
            loss_invariance = self.get_invariance_loss(image_feat, text_feat)
            loss_variance = self.get_variance_loss(image_feat, text_feat)
            loss_covariance = self.get_covariance_loss(image_feat, text_feat)
        else:
            zero_t = torch.tensor(0)
            loss_invariance = zero_t
            loss_variance = zero_t
            loss_covariance = zero_t
        loss = {'loss_mitc': loss_mitc, 'loss_hitm': loss_hitm, 'loss_hmlm': loss_hmlm,
                'loss_invariance': loss_invariance, 'loss_variance': loss_variance,
                'loss_covariance': loss_covariance}
        return loss

    def forward(self, image=None, text_ids=None, text_atts=None,
                text_ids_masked=None, masked_pos=None, masked_ids=None, image_atts=None,
                idx_to_group_img=None, target_bbox=None,
                is_image=None, ret_bbox_loss=False, text_atts_masked=None):
        loss = self.forward_multimodal(image, text_ids, text_atts, text_ids_masked, masked_pos,
                                       masked_ids,
                                       image_atts, idx_to_group_img,
                                       target_bbox, is_image, ret_bbox_loss)
        return loss
