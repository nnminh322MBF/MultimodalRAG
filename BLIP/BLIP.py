import torch
import torch.nn as nn
import torch.nn.functional as F
from ImageEncoder import ImageEncoder
from TextEncoder import TextEncoder
from ImageGroundedTextEncoder import ImageGroundedTextEncoder
from ImageGroundedTextDecoder import ImageGroundedTextDecoder


class BLIP(nn.Module):
    def __init__(
        self,
        image_encoder_config,
        text_encoder_config,
        grounded_encoder_config,
        grounded_decoder_config,
        embed_dim=768,
        proj_dim=256,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder(**image_encoder_config)
        self.text_encoder = TextEncoder(**text_encoder_config)
        self.ground_encoder = ImageGroundedTextEncoder(**grounded_encoder_config)
        self.ground_decoder = ImageGroundedTextDecoder(**grounded_decoder_config)

        self.vision_prj = nn.Linear(embed_dim, proj_dim)
        self.text_prj = nn.Linear(embed_dim, proj_dim)

        self.itm_head = nn.Linear(embed_dim, 2)

    def _forward_image_text_contrastive(self, image, text_input_ids):
        image_embeds = self.image_encoder(image)
        text_embeds = self.text_encoder(text_input_ids)
        image_feat = F.normalize(self.vision_prj(image_embeds[:, 0, :]), dim=-1)
        text_feat = F.normalize(self.vision_prj(text_embeds[:, 0, :]), dim=-1)

        return image_feat, text_feat

    def _forward_image_text_matching(self, text_input_ids, image):
        image_embeds = self.image_encoder(image)
        grounded_text_output = self.ground_encoder(text_input_ids, image_embeds)
        logits = self.itm_head(grounded_text_output)

        return logits

    def _forward_language_model(self, text_input_ids, image):
        image_embed = self.image_encoder(image)
        caption_logits = self.ground_decoder(text_input_ids, image)
        return caption_logits

    def forward(self, text_input_ids, image, mode="contrastive"):
        if mode == "contrastive":
            return self._forward_image_text_contrastive(text_input_ids, image)
        elif mode == "matching":
            return self._forward_image_text_contrastive(text_input_ids, image)
        elif mode == "generation":
            return self._forward_image_text_contrastive(text_input_ids, image)
        else:
            raise ValueError("mode must be 'contrastive', 'matching', or 'generation'")
