Diffusion Transformers (DiT)

## 🎯 핵심 질문

UNet의 inductive bias가 scaling의 병목이라면, ViT 스타일의 pure transformer architecture는 어떻게 더 나은 scaling law를 달성하는가?

## 🔍 왜 Transformer인가?

**문제 (Peebles & Xie, 2023)**:
- UNet의 skip connection, hierarchical structure는 **특정 image 구조를 가정** (작은 이미지에 최적)
- Scaling up하면, 이 inductive bias가 오히려 병목
- 다른 도메인(비디오, 3D, multimodal)으로 확장 어려움

**해결 (DiT)**:
- ViT처럼 image를 patch로 tokenize
- N개의 identical transformer block으로 diffusion 수행
- Timestep과 conditioning을 per-token adaptive normalization으로 주입

**결과**: FID가 **GFLOPs에 monotone** → compute-optimal scaling law 따름.

## 📐 수학적 선행 조건

- Vision Transformer (ViT, Dosovitskiy et al., 2021)
- Layer Normalization 및 affine transformation
- Patch embedding (Ch3)
- Scaling laws in deep learning (Kaplan et al., 2020)

## 📖 직관적 이해

**ViT-style tokenization**:
- Image $x \in \mathbb{R}^{H \times W \times 3}$ → patches $p \in \mathbb{R}^{(HW/P^2) \times (3P^2)}$
- Linear projection: $t_i = W_p \cdot p_i + b_p$ (patch embedding)
- Positional encoding (learnable)

**Transformer blocks** (N개, identical):
$$x_{\ell+1} = \text{Transformer}(x_\ell, \text{timestep}, \text{conditioning})$$

**AdaLN-Zero** (Adaptive Layer Norm with Zero initialization):
- Timestep & condition → $(γ, β, α)$ 생성
- Layer norm 이후: $y = (1 - \alpha) \cdot x + \alpha \cdot (\gamma \cdot \text{LN}(x) + \beta)$
- $\alpha = 0$ 초기화 → 초반 training에서 skip (identity에 가까움)

이는 "warm-up" 효과 + 안정적 gradient flow를 동시에 달성.

## ✏️ 엄밀한 정의

**DiT Forward Pass**:

**1. Patch Embedding & Positional Encoding**:
$$T = \text{patchify}(x, P) \in \mathbb{R}^{(HW/P^2) \times (3P^2)}$$
$$t_i = W_p T_i + b_p + \text{pos\_enc}(i), \quad i = 1, \ldots, HW/P^2$$

Add learnable [CLS] or [MASK] token (optional).

**2. Diffusion Timestep Embedding**:
$$t_{\text{emb}} = \text{sinusoidal}(t) \in \mathbb{R}^{d_{\text{hidden}}}$$
$$t_{\text{proj}} = \text{MLP}(t_{\text{emb}}) \in \mathbb{R}^{d_{\text{hidden}}}$$

**3. Conditioning Embedding** (e.g., class label $c$ or text):
$$c_{\text{proj}} = \text{Embedding}(c) \text{ or } \text{Text Encoder}(c) \in \mathbb{R}^{d_{\text{hidden}}}$$

**4. DiT Block** ($\ell$번째, $1 \le \ell \le L$):
$$h_\ell = t_{\text{proj}} + c_{\text{proj}} \in \mathbb{R}^{d_{\text{hidden}}}$$

**Adaptive Layer Norm**:
$$\gamma_\ell, \beta_\ell, \alpha_\ell = W_{\text{lin}}(h_\ell) \in \mathbb{R}^{3d_{\text{hidden}}}$$

$$\text{AdaLN}(x_\ell) = (1 - \alpha_\ell) x_\ell + \alpha_\ell [\gamma_\ell \cdot \text{LN}(x_\ell) + \beta_\ell]$$

**Self-Attention** (standard):
$$\text{Attn}(x_\ell) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

**Output Projection**:
$$x_{\ell+1} = \text{AdaLN}(\text{Attn}(x_\ell)) + x_\ell \quad \text{(residual)}$$

**5. Unpatch & Output**:
$$\hat{z}_0 = W_{\text{out}} \cdot \text{unpatchify}(x_L)$$

## 🔬 정리와 증명

**정리 (AdaLN-Zero의 Identity 초기화)**:

AdaLN-Zero에서 $\alpha = 0$로 초기화하면, 초반 forward pass는 대부분 identity에 가까우며, 이는 안정적 학습과 효율적 gradient flow를 보장한다.

*증명*:
1. AdaLN 정의:
   $$y = (1 - \alpha) x + \alpha [\gamma \cdot \text{LN}(x) + \beta]$$

2. $\alpha = 0$ 초기화:
   $$y = x + 0 \cdot [\gamma \cdot \text{LN}(x) + \beta] = x$$
   
   즉, skip identity.

3. Residual block이므로:
   $$x_{\ell+1} = x_\ell + \text{AdaLN}(\text{Attn}(x_\ell))$$
   
   Training 초반, $\text{AdaLN}$ 출력이 ~0이면, $x_{\ell+1} \approx x_\ell$ (stable gradient path).

4. Parameter update로 $\alpha$가 0에서 벗어나면서, 네트워크는 점진적으로 modulation 학습.

5. Gradient flow:
   $$\frac{\partial x_{\ell+1}}{\partial x_\ell} = I + \frac{\partial \text{AdaLN}(\text{Attn}(x_\ell))}{\partial x_\ell}$$
   
   Identity $I$가 항상 존재 → deep network에서도 vanishing 방지.

6. 동치로, 표준 transformer의 warm-up learning rate schedule처럼 작동하지만, **parameter 레벨**에서 automatic.

$\square$

## 💻 NumPy / PyTorch 구현 검증

### 실험 1: DiT Block 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TimestepEmbedding(nn.Module):
    """Sinusoidal + MLP"""
    def __init__(self, d_emb):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_emb, 4 * d_emb),
            nn.SiLU(),
            nn.Linear(4 * d_emb, d_emb)
        )
    
    def forward(self, t):
        # t: (batch,)
        device = t.device
        half_dim = t.shape[-1] if t.dim() > 0 else 1
        d_emb = self.mlp[0].in_features
        
        # Sinusoidal encoding
        emb = self._get_sinusoidal(t, d_emb, device)
        return self.mlp(emb)
    
    def _get_sinusoidal(self, t, d_emb, device):
        half_dim = d_emb // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        
        if t.dim() == 0:
            t = t.unsqueeze(0)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class AdaLNZero(nn.Module):
    """Adaptive Layer Norm with Zero initialization"""
    def __init__(self, d_hidden):
        super().__init__()
        self.ln = nn.LayerNorm(d_hidden)
        self.linear = nn.Linear(d_hidden, 3 * d_hidden)
        
        # Zero initialization for stability
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x, condition):
        # condition: (batch, d_hidden) = timestep embedding + class embedding
        gamma, beta, alpha = torch.chunk(
            self.linear(condition), 3, dim=-1
        )
        
        # alpha: (batch, d_hidden) → (batch, 1) for broadcasting
        alpha = alpha.mean(dim=-1, keepdim=True)  # aggregate
        
        # AdaLN formula
        ln_x = self.ln(x)  # (batch, seq_len, d_hidden)
        y = (1 - alpha) * x + alpha * (gamma * ln_x + beta)
        return y

class DiTBlock(nn.Module):
    """Transformer block with AdaLN-Zero"""
    def __init__(self, d_hidden, n_heads, ff_dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_hidden, n_heads, batch_first=True)
        self.norm1 = AdaLNZero(d_hidden)
        
        self.ff = nn.Sequential(
            nn.Linear(d_hidden, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_hidden)
        )
        self.norm2 = AdaLNZero(d_hidden)
    
    def forward(self, x, condition):
        # x: (batch, seq_len, d_hidden)
        # condition: (batch, d_hidden)
        
        # Self-attention with AdaLN-Zero
        x_norm = self.norm1(x, condition)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # FFN with AdaLN-Zero
        x_norm = self.norm2(x, condition)
        ff_out = self.ff(x_norm)
        x = x + ff_out
        
        return x

class MinimalDiT(nn.Module):
    """Minimal DiT for 64x64 latent diffusion"""
    def __init__(self, latent_channels=4, patch_size=2, d_hidden=256, n_heads=8, n_layers=4):
        super().__init__()
        
        # Patch embedding
        self.patch_embed = nn.Linear(
            (patch_size ** 2) * latent_channels,
            d_hidden
        )
        self.pos_embed = nn.Parameter(
            torch.randn(1, (64 // patch_size) ** 2, d_hidden) * 0.02
        )
        
        # Timestep embedding
        self.t_emb = TimestepEmbedding(d_hidden)
        
        # Class/conditioning embedding (optional)
        self.c_emb = nn.Embedding(10, d_hidden)  # 10 classes
        
        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(d_hidden, n_heads, 4 * d_hidden)
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.norm_out = nn.LayerNorm(d_hidden)
        self.linear_out = nn.Linear(d_hidden, (patch_size ** 2) * latent_channels)
        
        self.patch_size = patch_size
        self.d_hidden = d_hidden
    
    def forward(self, z, t, c=None):
        # z: (batch, channels, 64, 64) latent
        # t: (batch,) timestep
        # c: (batch,) class label (optional)
        
        batch, channels, h, w = z.shape
        
        # Patch embedding
        z_patches = z.reshape(batch, channels, h // self.patch_size, self.patch_size,
                              w // self.patch_size, self.patch_size)
        z_patches = z_patches.permute(0, 2, 4, 1, 3, 5).contiguous()
        z_patches = z_patches.reshape(batch, (h // self.patch_size) * (w // self.patch_size), -1)
        
        x = self.patch_embed(z_patches) + self.pos_embed
        
        # Conditioning
        t_emb = self.t_emb(t)  # (batch, d_hidden)
        if c is not None:
            c_emb = self.c_emb(c)
        else:
            c_emb = torch.zeros_like(t_emb)
        
        condition = t_emb + c_emb
        
        # DiT blocks
        for block in self.blocks:
            x = block(x, condition)
        
        # Output
        x = self.norm_out(x)
        x = self.linear_out(x)
        
        # Unpatch
        x = x.reshape(batch, h // self.patch_size, w // self.patch_size, -1)
        x = x.permute(0, 3, 1, 2).contiguous()
        
        return x

# Test
dit = MinimalDiT(latent_channels=4, patch_size=2, d_hidden=256, n_heads=8, n_layers=4)
z = torch.randn(2, 4, 64, 64)
t = torch.tensor([100, 500])
c = torch.tensor([3, 7])

output = dit(z, t, c)
print(f"Input: {z.shape}, Output: {output.shape}")
print(f"DiT Parameters: {sum(p.numel() for p in dit.parameters()) / 1e6:.1f}M")
```

### 실험 2: AdaLN-Zero의 초기화 효과

```python
# AdaLN-Zero 초기화가 identity에 가까운지 확인

dit_block = DiTBlock(d_hidden=256, n_heads=8, ff_dim=1024)

x = torch.randn(2, 32, 256)  # batch=2, seq_len=32, d_hidden=256
condition = torch.randn(2, 256)

# Forward with zero-init
output = dit_block(x, condition)

# 초기화 직후, output - x (residual) 의 norm이 작아야 함
residual = output - x
residual_norm = residual.norm().item()
input_norm = x.norm().item()

print(f"Input norm: {input_norm:.4f}")
print(f"Residual norm (w/ AdaLN-Zero init): {residual_norm:.4f}")
print(f"Ratio: {residual_norm / input_norm:.6f}")
print("→ Close to 0 means near-identity at initialization")
```

### 실험 3: DiT vs UNet 파라미터 효율성

```python
# 동일한 computational budget에서 모델 크기 비교

def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6

# DiT with varying depth
dit_small = MinimalDiT(d_hidden=192, n_heads=6, n_layers=4)
dit_medium = MinimalDiT(d_hidden=384, n_heads=8, n_layers=12)
dit_large = MinimalDiT(d_hidden=768, n_heads=16, n_layers=24)

print(f"DiT-S: {count_params(dit_small):.1f}M")
print(f"DiT-M: {count_params(dit_medium):.1f}M")
print(f"DiT-L: {count_params(dit_large):.1f}M")

# UNet for comparison (from 02-unet-architecture.md)
from diffusers import UNet2DModel
unet = UNet2DModel(
    sample_size=64,
    in_channels=4,
    out_channels=4,
    layers_per_block=2,
    block_out_channels=(128, 256, 512, 1024)
)

print(f"UNet (comparable): {count_params(unet):.1f}M")
print("\n→ DiT scales more predictably; no architectural bottleneck")
```

## 🔗 실전 활용

**Sora (OpenAI, 2024)**: DiT backbone (video diffusion)
**SD3 (Stability, 2024)**: MM-DiT (text + image tokens, Ch6-04)

**코드 예시**:
```python
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium",
    torch_dtype=torch.float16
).to("cuda")

image = pipe(
    "a cat wearing sunglasses",
    num_inference_steps=50,
    guidance_scale=7.5
).images[0]
```

## ⚖️ 가정과 한계

1. **Patch 크기 고정**: patch size = 2, 4, 8 중 선택만 가능 (architecture-dependent)
2. **Positional encoding**: learnable positional embedding은 해상도 변화에 약함 (absolute position 가정)
3. **Computation vs memory**: Self-attention $O(N^2)$ → 매우 고해상도에서는 병목 (여전히)
4. **Scaling law 가정**: FID가 GFLOPs에 monotone이라는 가정은 특정 범위에서만 유효

## 📌 핵심 정리

1. **DiT = patch tokenization + N identical transformer blocks + AdaLN-Zero**
2. AdaLN-Zero의 $\alpha=0$ 초기화 → identity skip → 안정적 학습
3. **Scaling law**: FID $\propto$ GFLOPs$^{-\beta}$ (polynomial), UNet의 병목 회피
4. **산업 채택**: Sora (video), SD3 (multimodal) 기본 backbone

## 🤔 생각해볼 문제

1. **왜 AdaLN-Zero에서 $\alpha$를 채널별이 아닌 시퀀스 전체에 대해 스칼라로 사용하는가?**
   <details>
   <summary>힌트</summary>
   채널별 $\alpha$는 더 많은 파라미터 (overhead). 스칼라 $\alpha$는 모든 토큰에 동일한 skip ratio → simpler, 여전히 효과적. 논문 실험에서 스칼라가 충분함을 보였음.
   </details>

2. **Patch size를 작게 하면 (예: 1×1) 더 많은 토큰이 생기는데, 이것이 long-range dependency 학습에 도움이 될까?**
   <details>
   <summary>힌트</summary>
   토큰 수 증가 → attention complexity $O(N^2)$ 폭증. 반면, 더 세밀한 정보 보존. Trade-off: patch_size=2, 4 (64×64 latent) is sweet spot for $O(N^2)$ tractability.
   </details>

3. **DiT의 scaling law가 왜 UNet의 hierarchical structure보다 더 "clean"한가?**
   <details>
   <summary>힌트</summary>
   UNet은 encoder-decoder 깊이, skip connection 구조가 image resolution에 dependent → architecture choice가 이미 bias. DiT는 identical blocks → no structural bias in depth → compute 효율이 전적으로 parameter 수와 flops에만 dependent → scaling law가 pure하게 emerge.
   </details>

---

<div align="center">

[◀ 이전](./02-unet-architecture.md) | [📚 README](../README.md) | [다음 ▶](./04-mm-dit.md)

</div>
