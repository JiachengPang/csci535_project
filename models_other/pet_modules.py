import torch
import torch.nn as nn


class VanillaEncoder(nn.Module):
    def __init__(self, num_latents, spec_enc, rgb_enc):
        super(VanillaEncoder, self).__init__()

        # SPEC
        # Attention Layer
        self.spec_norm1 = spec_enc.norm1
        self.spec_attn = spec_enc.attn
        # Feed Forward Layers
        self.spec_norm2 = spec_enc.norm2
        self.spec_mlp = spec_enc.mlp

        # RGB
        # Attention Layer
        self.rgb_norm1 = rgb_enc.norm1
        self.rgb_attn = rgb_enc.attn
        # Feed Forward Layers
        self.rgb_norm2 = rgb_enc.norm2
        self.rgb_mlp = rgb_enc.mlp

        # Latents
        self.num_latents = num_latents
        self.latents = nn.Parameter(torch.empty(1, num_latents, 768).normal_(std=0.02))
        self.scale_a = nn.Parameter(torch.zeros(1))
        self.scale_v = nn.Parameter(torch.zeros(1))

    def attention(self, q, k, v):  # requires q,k,v to have same dim
        B, N, C = q.shape
        attn = (q @ k.transpose(-2, -1)) * (C**-0.5)  # scaling
        attn = attn.softmax(dim=-1)
        x = (attn @ v).reshape(B, N, C)
        return x

    # Latent Fusion
    def fusion(self, audio_tokens, visual_tokens):
        # shapes
        BS = audio_tokens.shape[0]
        # concat all the tokens
        concat_ = torch.cat((audio_tokens, visual_tokens), dim=1)
        # cross attention (AV -->> latents)
        fused_latents = self.attention(
            q=self.latents.expand(BS, -1, -1), k=concat_, v=concat_
        )
        # cross attention (latents -->> AV)
        audio_tokens = audio_tokens + self.scale_a * self.attention(
            q=audio_tokens, k=fused_latents, v=fused_latents
        )
        visual_tokens = visual_tokens + self.scale_v * self.attention(
            q=visual_tokens, k=fused_latents, v=fused_latents
        )
        return audio_tokens, visual_tokens

    def forward(self, x, y):
        # Bottleneck Fusion
        x, y = self.fusion(x, y)

        # Attn skip connections
        x = x + self.spec_attn(self.spec_norm1(x))
        y = y + self.rgb_attn(self.rgb_norm1(y))

        # FFN + skip conections
        x = x + self.spec_mlp(self.spec_norm2(x))
        y = y + self.rgb_mlp(self.rgb_norm2(y))
        return x, y


#####################################################################################################################################################
#####################################################################################################################################################
#####################################################################################################################################################


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class AdaptFormer(nn.Module):
    def __init__(self, num_latents, dim, spec_enc, rgb_enc):
        super(AdaptFormer, self).__init__()

        # SPEC
        # Attention Layer
        self.spec_norm1 = spec_enc.norm1
        self.spec_attn = spec_enc.attn
        # Feed Forward Layers
        self.spec_norm2 = spec_enc.norm2
        self.spec_mlp = spec_enc.mlp

        # RGB
        # Attention Layer
        self.rgb_norm1 = rgb_enc.norm1
        self.rgb_attn = rgb_enc.attn
        # Feed Forward Layers
        self.rgb_norm2 = rgb_enc.norm2
        self.rgb_mlp = rgb_enc.mlp

        # Adapter params
        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

        # Spectrogram
        self.spec_down = nn.Linear(768, dim)
        self.spec_up = nn.Linear(dim, 768)
        nn.init.xavier_uniform_(self.spec_down.weight)
        nn.init.zeros_(self.spec_down.bias)
        nn.init.zeros_(self.spec_up.weight)
        nn.init.zeros_(self.spec_up.bias)
        self.spec_scale = nn.Parameter(torch.ones(1))

        # RGB images
        self.rgb_down = nn.Linear(768, dim)
        self.rgb_up = nn.Linear(dim, 768)
        nn.init.xavier_uniform_(self.rgb_down.weight)
        nn.init.zeros_(self.rgb_down.bias)
        nn.init.zeros_(self.rgb_up.weight)
        nn.init.zeros_(self.rgb_up.bias)
        self.rgb_scale = nn.Parameter(torch.ones(1))

        # Latents
        self.num_latents = num_latents
        self.latents = nn.Parameter(torch.empty(1, num_latents, 768).normal_(std=0.02))
        self.scale_a = nn.Parameter(torch.zeros(1))
        self.scale_v = nn.Parameter(torch.zeros(1))

    def attention(self, q, k, v):  # requires q,k,v to have same dim
        B, N, C = q.shape
        attn = (q @ k.transpose(-2, -1)) * (C**-0.5)  # scaling
        attn = attn.softmax(dim=-1)
        x = (attn @ v).reshape(B, N, C)
        return x

    # Latent Fusion
    def fusion(self, audio_tokens, visual_tokens):
        # shapes
        BS = audio_tokens.shape[0]
        # concat all the tokens
        concat_ = torch.cat((audio_tokens, visual_tokens), dim=1)
        # cross attention (AV -->> latents)
        fused_latents = self.attention(
            q=self.latents.expand(BS, -1, -1), k=concat_, v=concat_
        )
        # cross attention (latents -->> AV)
        audio_tokens = audio_tokens + self.scale_a * self.attention(
            q=audio_tokens, k=fused_latents, v=fused_latents
        )
        visual_tokens = visual_tokens + self.scale_v * self.attention(
            q=visual_tokens, k=fused_latents, v=fused_latents
        )
        return audio_tokens, visual_tokens

    def forward_audio_AF(self, x):
        x_down = self.spec_down(x)
        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.spec_up(x_down)
        return x_up

    def forward_visual_AF(self, x):
        x_down = self.rgb_down(x)
        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.rgb_up(x_down)
        return x_up

    def forward(self, x, y):
        # Bottleneck Fusion
        x, y = self.fusion(x, y)

        # Attn skip connections
        x = (
            x
            + self.spec_attn(
                self.spec_norm1(x), self.spec_norm1(x), self.spec_norm1(x)
            )[0]
        )

        y = (
            y
            + self.rgb_attn(self.rgb_norm1(y), self.rgb_norm1(y), self.rgb_norm1(y))[0]
        )
        # FFN + skip conections
        x = (
            x
            + self.spec_mlp(self.spec_norm2(x))
            + self.forward_audio_AF(self.spec_norm2(x)) * self.spec_scale
        )
        y = (
            y
            + self.rgb_mlp(self.rgb_norm2(y))
            + self.forward_visual_AF(self.rgb_norm2(y)) * self.rgb_scale
        )
        return x, y


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
