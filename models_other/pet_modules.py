import torch
import torch.nn as nn


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class AdaptFormerAT(nn.Module):
    def __init__(self, num_latents, dim, a_enc, t_enc):
        super(AdaptFormerAT, self).__init__()

        self.a_enc = a_enc
        self.t_enc = t_enc

        # Adapter params
        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

        # audio
        self.a_down = nn.Linear(768, dim)
        self.a_up = nn.Linear(dim, 768)
        nn.init.xavier_uniform_(self.a_down.weight)
        nn.init.zeros_(self.a_down.bias)
        nn.init.zeros_(self.a_up.weight)
        nn.init.zeros_(self.a_up.bias)
        self.a_scale = nn.Parameter(torch.ones(1))

        # text
        self.t_down = nn.Linear(768, dim)
        self.t_up = nn.Linear(dim, 768)
        nn.init.xavier_uniform_(self.t_down.weight)
        nn.init.zeros_(self.t_down.bias)
        nn.init.zeros_(self.t_up.weight)
        nn.init.zeros_(self.t_up.bias)
        self.t_scale = nn.Parameter(torch.ones(1))

        # Latents
        self.num_latents = num_latents
        self.latents = nn.Parameter(torch.empty(1, num_latents, 768).normal_(std=0.02))
        self.scale_a = nn.Parameter(torch.zeros(1))
        self.scale_t = nn.Parameter(torch.zeros(1))

    def attention(self, q, k, v):  # requires q,k,v to have same dim
        B, N, C = q.shape
        attn = (q @ k.transpose(-2, -1)) * (C**-0.5)  # scaling
        attn = attn.softmax(dim=-1)
        x = (attn @ v).reshape(B, N, C)
        return x

    # Latent Fusion
    def fusion(self, audio_tokens, text_tokens):
        # shapes
        BS = audio_tokens.shape[0]
        # concat all the tokens
        concat_ = torch.cat((audio_tokens, text_tokens), dim=1)
        # cross attention (AT -->> latents)
        fused_latents = self.attention(
            q=self.latents.expand(BS, -1, -1), k=concat_, v=concat_
        )
        # cross attention (latents -->> AT)
        audio_tokens = audio_tokens + self.scale_a * self.attention(
            q=audio_tokens, k=fused_latents, v=fused_latents
        )
        text_tokens = text_tokens + self.scale_t * self.attention(
            q=text_tokens, k=fused_latents, v=fused_latents
        )
        return audio_tokens, text_tokens

    def forward_audio_AF(self, x):
        x_down = self.a_down(x)
        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.a_up(x_down)
        return x_up

    def forward_text_AF(self, x):
        x_down = self.t_down(x)
        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.t_up(x_down)
        return x_up

    def forward(self, x, y):
        # Bottleneck Fusion
        x, y = self.fusion(x, y)
        with torch.no_grad():
            x = x + self.a_enc.attention(x)[0]
            y = y + self.t_enc.attention(y)[0]

        x_ = self.a_enc.dropout(x)
        x_ = self.a_enc.layer_norm(x)
        x_ = self.a_enc.feed_forward(x)
        x_ = self.a_enc.final_layer_norm(x)

        y_ = self.t_enc.intermediate(y)
        y_ = self.t_enc.output(y_, y)

        # FFN + skip conections
        x = x + x_ + self.forward_audio_AF(x) * self.a_scale
        y = y + y_ + self.forward_text_AF(y) * self.t_scale
        return x, y
