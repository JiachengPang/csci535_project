import torch
import torch.nn as nn
from transformers import HubertModel, RobertaModel, HubertConfig, RobertaConfig
from .pet_modules import AdaptFormer


class ATmodel(nn.Module):
    def __init__(self, num_classes, num_latents=16, dim=128):
        super(ATmodel, self).__init__()

        self.audio_encoder = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        self.text_encoder = RobertaModel.from_pretrained("roberta-base")

        for param in self.audio_encoder.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        self.audio_hidden_size = self.audio_encoder.config.hidden_size  # 768
        self.text_hidden_size = self.text_encoder.config.hidden_size  # 768

        # class DummyBlock:
        #     def __init__(self, norm1, attn, norm2, mlp):
        #         self.norm1 = norm1
        #         self.attn = attn
        #         self.norm2 = norm2
        #         self.mlp = mlp

        # dummy_layer = DummyBlock(
        #     norm1=nn.LayerNorm(768),
        #     attn=nn.MultiheadAttention(768, 12, batch_first=True),
        #     norm2=nn.LayerNorm(768),
        #     mlp=nn.Sequential(nn.Linear(768, 3072), nn.GELU(), nn.Linear(3072, 768)),
        # )

        self.audio_text_blocks = nn.Sequential(
            *[
                AdaptFormer(
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

    def forward_audio_features(self, x):
        out = self.audio_encoder.feature_extractor(x)
        out = self.audio_encoder.feature_projection(out)
        out = self.audio_encoder.encoder.pos_conv_embed(out)
        out = self.audio_encoder.encoder.layer_norm(out)
        out = self.audio_encoder.encoder.dropout(out)
        return out

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
        text_tokens = self.text_encoder.pooler(text_tokens)
        text_cls = text_tokens[:, 0]

        audio_cls = self.norm_audio(audio_cls)
        text_cls = self.norm_text(text_cls)

        fused = 0.5 * (audio_cls + text_cls)
        return fused

    def forward(self, audio_input, text_input, text_mask):
        audio_tokens = self.forward_audio_features(audio_input)
        text_tokens = self.forward_text_features(text_input, text_mask)
        fused = self.forward_encoder(audio_tokens, text_tokens)
        logits = self.classifier(fused)
        return logits


if __name__ == "__main__":
    # audio_encoder = HubertModel.from_pretrained("facebook/hubert-base-ls960")
    # print(audio_encoder.feature_extractor)
    # print(audio_encoder.feature_projection)
    # print(audio_encoder.encoder.pos_conv_embed)
    # print(audio_encoder.encoder.layer_norm)
    # print(audio_encoder.encoder.dropout)
    # print(audio_encoder.encoder.layers)

    # text_encoder = RobertaModel.from_pretrained("roberta-base")
    # print(text_encoder.embeddings)
    # print(text_encoder.encoder.layer)
    # print(text_encoder.pooler)
    pass
