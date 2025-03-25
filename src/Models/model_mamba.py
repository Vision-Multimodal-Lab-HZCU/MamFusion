import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
from Models.model_components import (
    BertAttention,
    LinearLayer,
    TrainablePositionalEncoding,
    GMMBlock,
)

from mamba_ssm import Mamba
from torch import Tensor
from typing import Optional


class VisionLanguageFusionModule(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout
        )
        self.vid_embed_proj = nn.Conv2d(5, 15680, kernel_size=1)
        self.top_k = 50
        self.km = None

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        tgt,
        memory,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):

        if tgt[1] is not None and memory[1] is not None:
            min_dim = min(tgt.size(1), memory.size(1))
            tgt = tgt[:, :min_dim]
            memory = memory[:, :min_dim]
        tgt2, weight = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=None,
            key_padding_mask=memory_key_padding_mask,
        )
        tgt = tgt * tgt2
        return tgt


class GMMFormer_Mamba_Net(nn.Module):
    def __init__(self, config):
        super(GMMFormer_Mamba_Net, self).__init__()
        self.config = config

        self.query_pos_embed = TrainablePositionalEncoding(
            max_position_embeddings=config.max_desc_l,
            hidden_size=config.hidden_size,
            dropout=config.input_drop,
        )
        self.clip_pos_embed = TrainablePositionalEncoding(
            max_position_embeddings=config.max_ctx_l,
            hidden_size=config.hidden_size,
            dropout=config.input_drop,
        )
        self.frame_pos_embed = TrainablePositionalEncoding(
            max_position_embeddings=config.max_ctx_l,
            hidden_size=config.hidden_size,
            dropout=config.input_drop,
        )

        self.query_input_proj = LinearLayer(
            config.query_input_size,
            config.hidden_size,
            layer_norm=True,
            dropout=config.input_drop,
            relu=True,
        )
        self.query_encoder = BertAttention(
            edict(
                hidden_size=config.hidden_size,
                intermediate_size=config.hidden_size,
                hidden_dropout_prob=config.drop,
                num_attention_heads=config.n_heads,
                attention_probs_dropout_prob=config.drop,
            )
        )

        self.clip_input_proj = LinearLayer(
            config.visual_input_size,
            config.hidden_size,
            layer_norm=True,
            dropout=config.input_drop,
            relu=True,
        )
        self.clip_encoder = GMMBlock(
            edict(
                hidden_size=config.hidden_size,
                intermediate_size=config.hidden_size,
                hidden_dropout_prob=config.drop,
                num_attention_heads=config.n_heads,
                attention_probs_dropout_prob=config.drop,
            )
        )
        self.clip_encoder_mamba = nn.ModuleList(
            [
                Mamba(
                    d_model=config.hidden_size,  # Model dimension d_model
                    d_state=16,  # SSM state expansion factor # 64
                    d_conv=4,  # Local convolution width
                    expand=2,  # Block expansion factor
                    use_fast_path=False,
                ),
                nn.LayerNorm(config.hidden_size, eps=1e-5),
            ]
        )

        self.clip_encoder_2 = GMMBlock(
            edict(
                hidden_size=config.hidden_size,
                intermediate_size=config.hidden_size,
                hidden_dropout_prob=config.drop,
                num_attention_heads=config.n_heads,
                attention_probs_dropout_prob=config.drop,
            )
        )
        self.clip_encoder_2_mamba = nn.ModuleList(
            [
                Mamba(
                    d_model=config.hidden_size,  # Model dimension d_model
                    d_state=16,  # SSM state expansion factor # 64
                    d_conv=4,  # Local convolution width
                    expand=2,  # Block expansion factor
                    use_fast_path=False,
                ),
                nn.LayerNorm(config.hidden_size, eps=1e-5),
            ]
        )

        self.frame_input_proj = LinearLayer(
            config.visual_input_size,
            config.hidden_size,
            layer_norm=True,
            dropout=config.input_drop,
            relu=True,
        )
        self.frame_encoder_1 = GMMBlock(
            edict(
                hidden_size=config.hidden_size,
                intermediate_size=config.hidden_size,
                hidden_dropout_prob=config.drop,
                num_attention_heads=config.n_heads,
                attention_probs_dropout_prob=config.drop,
            )
        )
        self.frame_encoder_1_mamba = nn.ModuleList(
            [
                Mamba(
                    d_model=config.hidden_size,  # Model dimension d_model
                    d_state=16,  # SSM state expansion factor # 64
                    d_conv=4,  # Local convolution width
                    expand=2,  # Block expansion factor
                    use_fast_path=False,
                ),
                nn.LayerNorm(config.hidden_size, eps=1e-5),
            ]
        )

        self.frame_encoder_2 = GMMBlock(
            edict(
                hidden_size=config.hidden_size,
                intermediate_size=config.hidden_size,
                hidden_dropout_prob=config.drop,
                num_attention_heads=config.n_heads,
                attention_probs_dropout_prob=config.drop,
            )
        )
        self.frame_encoder_2_mamba = nn.ModuleList(
            [
                Mamba(
                    d_model=config.hidden_size,  # Model dimension d_model
                    d_state=16,  # SSM state expansion factor # 64
                    d_conv=4,  # Local convolution width
                    expand=2,  # Block expansion factor
                    use_fast_path=False,
                ),
                nn.LayerNorm(config.hidden_size, eps=1e-5),
            ]
        )

        self.modular_vector_mapping = nn.Linear(
            config.hidden_size, out_features=1, bias=False
        )
        self.modular_vector_mapping_2 = nn.Linear(
            config.hidden_size, out_features=1, bias=False
        )

        self.v2t_fusion = VisionLanguageFusionModule(
            d_model=config.hidden_size, nhead=config.n_heads
        )

        self.t2v_fusion = VisionLanguageFusionModule(
            d_model=config.hidden_size, nhead=config.n_heads
        )

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize the weights."""

        def re_init(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(
                    mean=0.0, std=self.config.initializer_range
                )
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            elif isinstance(module, nn.Conv1d):
                module.reset_parameters()
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

        self.apply(re_init)

    def set_hard_negative(self, use_hard_negative, hard_pool_size):
        """use_hard_negative: bool; hard_pool_size: int,"""
        self.config.use_hard_negative = use_hard_negative
        self.config.hard_pool_size = hard_pool_size

    def forward(self, batch):

        clip_video_feat = batch["clip_video_features"]
        query_feat = batch["text_feat"]
        query_mask = batch["text_mask"]
        query_labels = batch["text_labels"]
        frame_video_feat = batch["frame_video_features"]
        frame_video_mask = batch["videos_mask"]
        encoded_frame_feat, vid_proposal_feat = self.encode_context(
            clip_video_feat, frame_video_feat, frame_video_mask
        )

        (
            clip_scale_scores,
            clip_scale_scores_,
            frame_scale_scores,
            frame_scale_scores_,
        ) = self.get_pred_from_raw_query(
            query_feat,
            query_mask,
            query_labels,
            vid_proposal_feat,
            encoded_frame_feat,
            return_query_feats=True,
        )

        label_dict = {}
        for index, label in enumerate(query_labels):
            if label in label_dict:
                label_dict[label].append(index)
            else:
                label_dict[label] = []
                label_dict[label].append(index)

        video_query = self.encode_query(query_feat, query_mask)
        encoded_frame_feat = encoded_frame_feat.repeat(4, 1)
        encoded_frame_feat = encoded_frame_feat[:, : video_query.shape[1]]

        encoded_frame_feat = self.v2t_fusion(
            tgt=encoded_frame_feat,
            memory=video_query,
            memory_key_padding_mask=None,
            pos=None,
            query_pos=None,
        )

        video_query = self.t2v_fusion(
            tgt=video_query,
            memory=encoded_frame_feat,
            memory_key_padding_mask=None,
            pos=None,
            query_pos=None,
        )

        return [
            clip_scale_scores,
            clip_scale_scores_,
            label_dict,
            frame_scale_scores,
            frame_scale_scores_,
            video_query,
        ]

    def encode_query(self, query_feat, query_mask):
        encoded_query = self.encode_input(
            query_feat,
            query_mask,
            self.query_input_proj,
            self.query_encoder,
            self.query_pos_embed,
        )
        if query_mask is not None:
            mask = query_mask.unsqueeze(1)

        video_query = self.get_modularized_queries(encoded_query, query_mask)

        return video_query

    def encode_context(
        self, clip_video_feat, frame_video_feat, video_mask=None
    ):

        encoded_clip_feat = self.encode_input(
            clip_video_feat,
            None,
            self.clip_input_proj,
            self.clip_encoder,
            self.clip_pos_embed,
        )

        encoded_clip_feat_mamba = encoded_clip_feat
        for layer in self.clip_encoder_mamba:
            encoded_clip_feat_mamba = layer(encoded_clip_feat_mamba)
        encoded_clip_feat = encoded_clip_feat + encoded_clip_feat_mamba

        encoded_clip_feat = self.clip_encoder_2(encoded_clip_feat, None)

        encoded_clip_feat_mamba = encoded_clip_feat
        for layer in self.clip_encoder_2_mamba:
            encoded_clip_feat_mamba = layer(encoded_clip_feat_mamba)
        encoded_clip_feat = encoded_clip_feat + encoded_clip_feat_mamba

        encoded_frame_feat = self.encode_input(
            frame_video_feat,
            video_mask,
            self.frame_input_proj,
            self.frame_encoder_1,
            self.frame_pos_embed,
        )

        encoded_frame_feat_mamba = encoded_frame_feat
        for layer in self.frame_encoder_1_mamba:
            encoded_frame_feat_mamba = layer(encoded_frame_feat_mamba)
        encoded_frame_feat = encoded_frame_feat + encoded_frame_feat_mamba

        encoded_frame_feat = self.frame_encoder_2(
            encoded_frame_feat, video_mask.unsqueeze(1)
        )

        encoded_frame_feat_mamba = encoded_frame_feat
        for layer in self.frame_encoder_2_mamba:
            encoded_frame_feat_mamba = layer(encoded_frame_feat_mamba)
        encoded_frame_feat = encoded_frame_feat + encoded_frame_feat_mamba

        encoded_frame_feat = self.get_modularized_frames(
            encoded_frame_feat, video_mask
        )

        return encoded_frame_feat, encoded_clip_feat

    @staticmethod
    def encode_input(
        feat, mask, input_proj_layer, encoder_layer, pos_embed_layer
    ):
        feat = input_proj_layer(feat)
        feat = pos_embed_layer(feat)
        if mask is not None:
            mask = mask.unsqueeze(1)
        return encoder_layer(feat, mask)

    def get_modularized_queries(self, encoded_query, query_mask):
        modular_attention_scores = self.modular_vector_mapping(encoded_query)
        modular_attention_scores = F.softmax(
            mask_logits(modular_attention_scores, query_mask.unsqueeze(2)),
            dim=1,
        )
        modular_queries = torch.einsum(
            "blm,bld->bmd", modular_attention_scores, encoded_query
        )
        return modular_queries.squeeze()

    def get_modularized_frames(self, encoded_query, query_mask):

        modular_attention_scores = self.modular_vector_mapping_2(encoded_query)
        modular_attention_scores = F.softmax(
            mask_logits(modular_attention_scores, query_mask.unsqueeze(2)),
            dim=1,
        )
        modular_queries = torch.einsum(
            "blm,bld->bmd", modular_attention_scores, encoded_query
        )
        return modular_queries.squeeze()

    @staticmethod
    def get_clip_scale_scores(modularied_query, context_feat):

        modularied_query = F.normalize(modularied_query, dim=-1)
        context_feat = F.normalize(context_feat, dim=-1)

        clip_level_query_context_scores = torch.matmul(
            context_feat, modularied_query.t()
        ).permute(2, 1, 0)

        query_context_scores, indices = torch.max(
            clip_level_query_context_scores, dim=1
        )

        return query_context_scores

    @staticmethod
    def get_unnormalized_clip_scale_scores(modularied_query, context_feat):

        query_context_scores = torch.matmul(
            context_feat, modularied_query.t()
        ).permute(2, 1, 0)

        output_query_context_scores, indices = torch.max(
            query_context_scores, dim=1
        )

        return output_query_context_scores

    def get_pred_from_raw_query(
        self,
        query_feat,
        query_mask,
        query_labels=None,
        video_proposal_feat=None,
        encoded_frame_feat=None,
        return_query_feats=False,
    ):

        video_query = self.encode_query(query_feat, query_mask)

        clip_scale_scores = self.get_clip_scale_scores(
            video_query, video_proposal_feat
        )

        frame_scale_scores = torch.matmul(
            F.normalize(encoded_frame_feat, dim=-1),
            F.normalize(video_query, dim=-1).t(),
        ).permute(1, 0)

        if return_query_feats:
            clip_scale_scores_ = self.get_unnormalized_clip_scale_scores(
                video_query, video_proposal_feat
            )
            frame_scale_scores_ = torch.matmul(
                encoded_frame_feat, video_query.t()
            ).permute(1, 0)

            return (
                clip_scale_scores,
                clip_scale_scores_,
                frame_scale_scores,
                frame_scale_scores_,
            )
        else:

            return clip_scale_scores, frame_scale_scores


def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e10)
