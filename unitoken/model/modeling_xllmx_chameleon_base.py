import functools
import logging
import math
from typing import List

import torch
from torch import nn
from transformers import AutoProcessor, AutoModel

from .chameleon import ChameleonForConditionalGeneration
from .configuration_xllmx_chameleon import ChameleonXLLMXConfig

logger = logging.getLogger(__name__)

default_linear_init = functools.partial(nn.init.kaiming_uniform_, a=math.sqrt(5))

__all__ = ["ChameleonXLLMXForConditionalGenerationBase"]


class ChameleonXLLMXForConditionalGenerationBase(ChameleonForConditionalGeneration):
    config_class = ChameleonXLLMXConfig

    def __init__(self, config):
        super().__init__(config)
        self._init_proj()   # Initialize here to enable "from_pretrained" method to resume pre-saved 'adapter' weights
        self._init_vit()    # Initialize here to enable "from_pretrained" method to resume pre-saved 'vit' weights

    def _init_vit(self, vit_root="./ckpts/SigLIP"):
        self.vit = AutoModel.from_pretrained(vit_root).vision_model

    def _init_proj(self):
        self.adapter = nn.Sequential(
            nn.Linear(1152, self.model.config.hidden_size, bias=True),  # 1152 is the hidden_size of SigLIP
            nn.GELU(),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size, bias=True),
        )

    def forward(self, input_ids=None, labels=None, training=True, **kwargs):

        discrete_ids = [_[0] for _ in input_ids]
        images = [_[1] for _ in input_ids]
        image_paths = [_[2] for _ in input_ids]

        # with torch.no_grad():
        #     # Generate continuous visual tokens
        images = torch.cat(images, dim=0)
        vit_features = self.vit(images, interpolate_pos_encoding=True).last_hidden_state
        continuous_tokens = self.adapter(vit_features)
        continuous_tokens_num = continuous_tokens.shape[1]

        # Pad 'discrete_ids' and 'labels' to the same sequence length
        max_tokens = max([len(_) for _ in discrete_ids])
        max_tokens = min(max_tokens, self.config.max_position_embeddings)
        discrete_ids = [_[:max_tokens] for _ in discrete_ids]
        labels = [_[:max_tokens] for _ in labels]

        discrete_ids = [example + [0] * (max_tokens - len(example)) for example in discrete_ids]
        discrete_ids = torch.tensor(discrete_ids, dtype=torch.int64, device=self.device)

        labels = [label + [-100] * (max_tokens - len(label)) for label in labels]
        labels = torch.tensor(labels, dtype=torch.int64, device=self.device)

        # Fetch discrete embeddings from vocab
        discrete_tokens = self.model.embed_tokens(discrete_ids)
        
        # Insert continuous tokens into discrete tokens
        # Use format of "<soi>[discrete_tokens]<sep>[continuous_tokens]<eoi>"
        # <soi>="<racm3:break>"(8197)  <eoi>="<eoss>"(8196) are already set in pretokenization
        # TODO: set <sep>="<sentinel:0>"(8198)
        pad_token = self.model.embed_tokens(torch.tensor(0, dtype=torch.int64, device=self.device))
        sep_token = self.model.embed_tokens(torch.tensor(8198, dtype=torch.int64, device=self.device))
        sep_label = torch.tensor([8198], dtype=torch.int64, device=self.device)
        # images_end_pos = torch.where(discrete_ids==8196)[1]  # Omit batch dimension

        uni_tokens, uni_labels = [], []
        pad_num = continuous_tokens_num + 1
        for batch_id in range(len(discrete_ids)):
            image_end_pos = torch.where(discrete_ids[batch_id]==8196)[0]
            if len(image_end_pos) == 0:    # Text-only samples
                # Pad Text-only tokens and labels to the same length as Multimodal ones 
                # Number of paddings = 1(sep tokens)+729(continuous tokens)
                discrete_token = discrete_tokens[batch_id]
                label = labels[batch_id]

                uni_token = torch.cat([
                    discrete_token,
                    pad_token.unsqueeze(0).repeat(pad_num,1)
                ], dim=0)
                uni_label = torch.cat([
                    label,
                    torch.tensor([-100]*pad_num, device=label.device, dtype=label.dtype)
                ])
                uni_tokens.append(uni_token)
                uni_labels.append(uni_label)
            else:   # Multimodal samples
                assert len(image_end_pos) == 1
                discrete_token = discrete_tokens[batch_id]
                continuous_token = continuous_tokens[batch_id]
                label = labels[batch_id]

                uni_token = torch.cat([
                    discrete_token[:image_end_pos],
                    sep_token.unsqueeze(0),
                    continuous_token,
                    discrete_token[image_end_pos:]
                ], dim=0)
                uni_label = torch.cat([
                    label[:image_end_pos],
                    sep_label,
                    torch.ones_like(continuous_token[:,0], dtype=torch.int64) * -100,
                    label[image_end_pos:]
                ], dim=0)
                uni_tokens.append(uni_token)
                uni_labels.append(uni_label)

        uni_tokens = torch.stack(uni_tokens, dim=0)
        uni_labels = torch.stack(uni_labels, dim=0)

        result = ChameleonForConditionalGeneration.forward(
            self, inputs_embeds=uni_tokens, labels=uni_labels, use_cache=False, **kwargs
        )

        # explicit use_cache=False for the following
        # https://github.com/Lightning-AI/pytorch-lightning/issues/19267
        # result = ChameleonForConditionalGeneration.forward(
        #     self, input_ids=input_ids, labels=labels, use_cache=False, **kwargs
        # )

        c_loss = result[0]

        additional_loss_dict = {}
        if self.config.z_loss_weight > 0:
            logits: torch.Tensor = result[1]
            shift_logits = logits[..., :-1, :].contiguous()
            # shift_labels = labels[..., 1:].contiguous()
            shift_labels = uni_labels[..., 1:].contiguous()
            valid_mask = shift_labels >= 0
            z_loss = torch.logsumexp(shift_logits, dim=-1).pow(2)[valid_mask].mean()
            additional_loss_dict["z_loss"] = (z_loss, self.config.z_loss_weight)
        return c_loss, additional_loss_dict

    def get_fsdp_wrap_module_list(self) -> List:
        modules = [*list(self.model.layers), self.lm_head, self.model.embed_tokens]
        if hasattr(self.model, "vqmodel"):  # may be deleted
            modules.append(self.model.vqmodel)
        if hasattr(self, "vit"):
            modules.append(self.vit)
        return modules

    def get_checkpointing_wrap_module_list(self) -> List:
        modules = [
            *list(self.model.layers),
        ]
        if hasattr(self, "vit"):
            modules.append(self.vit)
        return modules

    def get_trainable_params(self, trainable_params):
        trainable_params = trainable_params.split(',')
        return trainable_params
    # # For 1-GPU debug usage
    # def get_trainable_params(self):
    #     return ['model.layers.0.self_attn.q_proj.weight']