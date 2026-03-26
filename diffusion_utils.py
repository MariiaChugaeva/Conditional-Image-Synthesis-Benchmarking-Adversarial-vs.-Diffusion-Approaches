import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from torch import einsum

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 1e-5, 0.999)


class ImprovedDiffusion:
    def __init__(self, num_timesteps, img_shape, device, min_snr_gamma=5.0, uniform_loss_weights=False):
        self.num_timesteps = num_timesteps
        self.img_shape = img_shape
        self.device = device
        self.min_snr_gamma = min_snr_gamma
        self.uniform_loss_weights = uniform_loss_weights

        betas = cosine_beta_schedule(num_timesteps)
        self.betas = betas.to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        ab = self.alphas_cumprod.clamp(min=1e-9)
        snr = ab / (1.0 - ab)
        self.snr = snr
        if uniform_loss_weights:
            self.loss_weight = torch.ones(num_timesteps, device=device)
        else:
            clip = torch.full_like(snr, min_snr_gamma)
            self.loss_weight = torch.minimum(snr, clip) / snr.clamp(min=1e-9)

        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod).clamp(min=1e-9)
        )

    def sample_timesteps(self, n):
        return torch.randint(0, self.num_timesteps, (n,), device=self.device, dtype=torch.long)

    def noise_images(self, x0, t):
        ab = self.alphas_cumprod[t][:, None, None, None]
        noise = torch.randn_like(x0)
        xt = ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise
        return xt, noise

    def predict_x0_from_eps(self, x, t, eps):
        ab = self.alphas_cumprod[t][:, None, None, None]
        return (x - (1.0 - ab).sqrt() * eps) / ab.sqrt().clamp(min=1e-8)

    def _predict_eps_cfg(self, model, x, t, labels, guidance_scale, null_label):
        if guidance_scale == 1.0:
            return model(x, t, labels)
        uncond = torch.full_like(labels, null_label)
        eps_c = model(x, t, labels)
        eps_u = model(x, t, uncond)
        return eps_u + guidance_scale * (eps_c - eps_u)

    @torch.no_grad()
    def sample_ddim(self, model, n, labels, steps=100, guidance_scale=2.0, null_label=10, eta=0.0, x_start=None):
        model.eval()
        if x_start is not None:
            x = x_start.to(self.device)
        else:
            x = torch.randn(n, *self.img_shape, device=self.device)
        ts = torch.linspace(0, self.num_timesteps - 1, steps, device=self.device).long()
        ts = torch.unique(ts).sort(descending=True).values
        ts_next = torch.cat([ts[1:], torch.tensor([-1], device=self.device)])

        for t_now, t_next in zip(ts.tolist(), ts_next.tolist()):
            t_b = torch.full((n,), t_now, device=self.device, dtype=torch.long)
            ab = self.alphas_cumprod[t_now]
            ab_next = self.alphas_cumprod[t_next] if t_next >= 0 else torch.ones((), device=self.device)

            eps = self._predict_eps_cfg(model, x, t_b, labels, guidance_scale, null_label)
            pred_x0 = self.predict_x0_from_eps(x, t_b, eps).clamp(-1, 1)

            if t_next < 0:
                x = pred_x0
                break

            sigma = (
                eta
                * (
                    (1.0 - ab_next) / (1.0 - ab).clamp(min=1e-9)
                    * (1.0 - ab / ab_next.clamp(min=1e-9))
                ).sqrt()
            )
            c1 = ab_next.sqrt()
            c2 = ((1.0 - ab_next) - sigma ** 2).clamp(min=0).sqrt()
            noise = torch.randn_like(x) if eta > 0 else 0.0
            x = c1 * pred_x0 + c2 * eps + sigma * noise

        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        return (x * 255).type(torch.uint8)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, kern_size, padd, groups, downsample=True, use_act=True):
        super().__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_channel,
                out_channel,
                kern_size,
                stride=2 if downsample else 1,
                padding=padd,
                groups=groups,
            ),
        )
        self.add_module("norm", nn.GroupNorm(groups, out_channel))
        self.add_module("act", nn.SiLU() if use_act else nn.Identity())


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, *, time_emb_dim=None, groups=8, downsample=False):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels))
            if time_emb_dim is not None
            else None
        )

        self.block1 = ConvBlock(in_channels, out_channels, 3, 1, groups, downsample=downsample)
        self.block2 = ConvBlock(out_channels, out_channels, 3, 1, groups, downsample=False, use_act=False)

        if in_channels != out_channels or downsample:
            self.res_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=2 if downsample else 1
            )
        else:
            self.res_conv = nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x, time_emb=None):
        res = self.res_conv(x)

        h = self.block1(x)
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            h = h + time_emb[:, :, None, None]
        h = self.act(h)
        h = self.block2(h)

        return h + res


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv)
        sim = einsum("b h d i, b h d j -> b h i j", q, k) * self.scale
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), LayerNorm(dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class Unet(nn.Module):
    def __init__(self, num_classes=10, null_label=10):
        super().__init__()
        self.null_label = null_label

        time_dim = 32 * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(32),
            nn.Linear(32, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.init_conv = nn.Conv2d(1, 32, kernel_size=1)

        self.down1 = ResnetBlock(32, 64, time_emb_dim=time_dim, downsample=True)
        self.attn1 = LinearAttention(64)
        self.down2 = ResnetBlock(64, 128, time_emb_dim=time_dim, downsample=True)
        self.attn2 = LinearAttention(128)
        self.down3 = ResnetBlock(128, 256, time_emb_dim=time_dim, downsample=True)
        self.attn3 = LinearAttention(256)

        self.bottleneck = ResnetBlock(256, 512, time_emb_dim=time_dim, downsample=False)
        self.attn_bottle = Attention(512)

        self.upsample1_op = Upsample(512)
        self.up1 = ResnetBlock(512 + 128, 256, time_emb_dim=time_dim, downsample=False)

        self.upsample2_op = Upsample(256)
        self.up2 = ResnetBlock(256 + 64, 128, time_emb_dim=time_dim, downsample=False)

        self.upsample3_op = Upsample(128)
        self.up3 = ResnetBlock(128 + 32, 64, time_emb_dim=time_dim, downsample=False)

        self.final_res = ResnetBlock(64, 32, time_emb_dim=time_dim, downsample=False)
        self.final_conv = nn.Conv2d(32, 1, 1)

        self.label_emb = nn.Embedding(num_classes + 1, time_dim)

    def forward(self, x, t, y):
        t_emb = self.time_mlp(t.float())
        t_emb = t_emb + self.label_emb(y)

        x_initial_conv_out = self.init_conv(x)
        h = x_initial_conv_out

        h_skip_14x14 = self.down1(h, t_emb)
        h = self.attn1(h_skip_14x14)

        h_skip_7x7 = self.down2(h, t_emb)
        h = self.attn2(h_skip_7x7)

        h_skip_4x4 = self.down3(h, t_emb)
        h = self.attn3(h_skip_4x4)

        h = self.bottleneck(h, t_emb)
        h = self.attn_bottle(h)

        h = self.upsample1_op(h)
        h = F.interpolate(h, size=h_skip_7x7.shape[-2:], mode="nearest")
        h = torch.cat([h, h_skip_7x7], dim=1)
        h = self.up1(h, t_emb)

        h = self.upsample2_op(h)
        h = F.interpolate(h, size=h_skip_14x14.shape[-2:], mode="nearest")
        h = torch.cat([h, h_skip_14x14], dim=1)
        h = self.up2(h, t_emb)

        h = self.upsample3_op(h)
        h = F.interpolate(h, size=x_initial_conv_out.shape[-2:], mode="nearest")
        h = torch.cat([h, x_initial_conv_out], dim=1)
        h = self.up3(h, t_emb)

        h = self.final_res(h, t_emb)
        return self.final_conv(h)