import torch
import torch.nn as nn
from transformers import HubertModel, RobertaModel, HubertConfig, RobertaConfig
from .pet_modules import AdaptFormerAT


class ATmodel(nn.Module):
    def __init__(self, num_classes, num_latents=16, dim=128, from_pretrained=None):
        super(ATmodel, self).__init__()

        self.audio_encoder = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        self.text_encoder = RobertaModel.from_pretrained("roberta-base")

        for param in self.audio_encoder.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        self.audio_hidden_size = self.audio_encoder.config.hidden_size  # 768
        self.text_hidden_size = self.text_encoder.config.hidden_size  # 768

        self.audio_text_blocks = nn.Sequential(
            *[
                AdaptFormerAT(
                    num_latents,
                    dim,
                    self.audio_encoder.encoder.layers[i],
                    self.text_encoder.encoder.layer[i],
                )
                for i in range(12)
            ]
        )

        self.norm_audio = nn.LayerNorm(768)
        self.norm_text = nn.LayerNorm(768)

        self.classifier = nn.Linear(768, num_classes)

        if from_pretrained:
            print(f"Loading weights from {from_pretrained}")
            ckpt = torch.load(from_pretrained, map_location="cpu")
            # print(ckpt.keys())
            self.load_state_dict(ckpt)

    def forward_audio_features(self, x):
        out = self.audio_encoder.feature_extractor(x)
        out = out.transpose(1, 2)
        out = self.audio_encoder.feature_projection(out)
        out = self.audio_encoder.encoder.pos_conv_embed(out)
        out = self.audio_encoder.encoder.layer_norm(out)
        out = self.audio_encoder.encoder.dropout(out)
        return out

    # def forward_audio_features(self, x):
    #     with torch.no_grad():
    #         out = self.audio_encoder(input_values=x).last_hidden_state
    #     return out

    def forward_text_features(self, x, attn_mask):
        # out = self.text_encoder(
        #     x, attention_mask=attn_mask
        # ).last_hidden_state  # (B, T, 768)
        out = self.text_encoder.embeddings(x)
        return out

    def forward_encoder(self, audio_tokens, text_tokens):
        for blk in self.audio_text_blocks:
            audio_tokens, text_tokens = blk(audio_tokens, text_tokens)

        audio_cls = audio_tokens[:, 0]
        text_cls = self.text_encoder.pooler(text_tokens)

        audio_cls = self.norm_audio(audio_cls)
        text_cls = self.norm_text(text_cls)

        fused = 0.5 * (audio_cls + text_cls)
        return fused

    def forward(self, audio_input, text_input, text_mask=None, return_features=False):
        with torch.no_grad():
            audio_tokens = self.forward_audio_features(audio_input)
            text_tokens = self.forward_text_features(text_input, text_mask)
            fused = self.forward_encoder(audio_tokens, text_tokens)
        if return_features:
            return fused
        logits = self.classifier(fused)
        return logits


if __name__ == "__main__":
    pass
