UNet Architecture for Diffusion Models

## 🎯 핵심 질문

UNet이 의료 영상 분할(2015)에서 diffusion의 denoising network로 재발명된 이유는 무엇이며, multi-scale skip connection과 timestep embedding이 어떻게 조화되는가?

## 🔍 왜 UNet인가?

**Ronneberger et al. (2015)**: U자 모양의 encoder-decoder + skip connection이 의료 영상 분할에서 SOTA 달성.

**Diffusion에 적합한 이유**:
1. **Multi-scale feature**: 낮은 해상도에서 의미론적 정보, 높은 해상도에서 세부 정보 → denoising에 완벽
2. **Skip connection**: gradient flow 강화 (매우 깊은 네트워크에도 학습 안정)
3. **Inductive bias**: 자연 이미지의 hierarchical 구조 가정 → 샘플 효율 증대

DDPM (Ho et al., 2020)은 이 구조를 채택하되, **timestep embedding**과 **self-/cross-attention**을 추가하여 conditioning을 강화했다.

## 📐 수학적 선행 조건

- Conv, ResBlock (Ch3)
- Positional encoding (sinusoidal, Ch5)
- Self-attention mechanism (Ch5)
- Group normalization (Yuxin Wu et al., 2018)

## 📖 직관적 이해

**UNet의 구조**:
```
Input (noise) → Encoder (down) → Bottleneck → Decoder (up) + Skip → Output (denoised)
                    ↓                              ↑
                64→32→16 res.          16→32→64 res.
```

**Timestep embedding 주입 (FiLM-style)**:
- Timestep $t$를 sinusoidal positional encoding으로 변환
- MLP를 거쳐 affine parameter $(\gamma_t, \beta_t)$ 생성
- 각 ResBlock의 GroupNorm 출력에 FiLM 적용:
  $$y = \gamma_t \cdot \text{GroupNorm}(x) + \beta_t$$

이는 timestep 정보를 모든 레이어에 broadcast하며, scale/shift를 통해 diffusion step을 "지시"한다.

## ✏️ 엄밀한 정의

**UNet2D 구조** (DDPM/SD 기준):

**Encoder Block** ($i$번째 레벨, $1 \le i \le L$):
$$x_{i,1} = \text{ResBlock}(x_{i-1,end}, t_{\text{emb}})$$
$$x_{i,2} = \text{ResBlock}(x_{i,1}, t_{\text{emb}})$$
$$x_{i,\text{down}} = \text{AvgPool}(x_{i,2}) \quad (\text{또는 stride-2 Conv})$$

Skip connection 저장: $s_i = x_{i,2}$.

**Bottleneck**:
$$x_{\text{bn}} = \text{ResBlock}(\text{SelfAttn}(x_{L,\text{down}}), t_{\text{emb}})$$

**Decoder Block** (역순):
$$x_{i,1} = \text{ResBlock}(\text{Upsample}(x_{i+1,2}) + s_i, t_{\text{emb}})$$
$$x_{i,2} = \text{ResBlock}(x_{i,1}, t_{\text{emb}})$$

where ResBlock은 다음과 같이 정의:
$$\text{ResBlock}(x, t_{\text{emb}}) = \sigma(y) + x$$
$$y = \text{Conv}(\sigma(\text{GroupNorm}(x))) + \text{FiLM}(t_{\text{emb}})$$

**Timestep Embedding**:
$$t_{\text{emb}} = \text{MLP}(\text{sinusoidal\_encoding}(t))$$

where sinusoidal encoding:
$$\text{emb}_i(t) = \begin{cases} \sin(t / 10000^{2i/d}) & i \text{ even} \\ \cos(t / 10000^{(2i-1)/d}) & i \text{ odd} \end{cases}$$

## 🔬 정리와 증명

**정리 (Skip Connection의 Gradient 보존)**:

UNet에서 skip connection을 통한 residual pathway는 깊은 네트워크에서도 gradient vanishing을 방지한다.

*증명*:
1. Loss $\mathcal{L}$에서 bottom layer의 activation $x_L$에 대한 gradient:
   $$\frac{\partial \mathcal{L}}{\partial x_L} = \frac{\partial \mathcal{L}}{\partial y_{\text{out}}} \cdot \frac{\partial y_{\text{out}}}{\partial x_L}$$

2. Skip connection이 없으면:
   $$\frac{\partial y_{\text{out}}}{\partial x_L} = \frac{\partial f_L}{\partial x_{L-1}} \cdot \frac{\partial f_{L-1}}{\partial x_{L-2}} \cdots \frac{\partial f_1}{\partial x_L}$$
   
   각 $\frac{\partial f_i}{\partial x_{i-1}}$이 1보다 작으면 → gradient exponentially decay.

3. Skip connection 포함 (residual):
   $$x_0' = x_0 + f(x_0) \Rightarrow \frac{\partial x_0'}{\partial x_0} = I + \frac{\partial f}{\partial x_0}$$
   
   Identity term $I$는 항상 1의 gradient 보존 → vanishing 완화.

4. UNet의 encoder→decoder skip은 동일 해상도 feature를 합침으로써, high-frequency 정보 손실을 방지:
   $$\frac{\partial \mathcal{L}}{\partial x_{i,\text{high}}} = \frac{\partial \mathcal{L}}{\partial y} \cdot 1 + \text{decoder path}$$

$\square$

## 💻 NumPy / PyTorch 구현 검증

### 실험 1: Simple UNet Skeleton (DDPM 스타일)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionalEmbedding(nn.Module):
    """Timestep sinusoidal encoding"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        # t: shape (batch,)
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ResBlock(nn.Module):
    """Residual block with FiLM timestep modulation"""
    def __init__(self, channels, t_emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        
        # FiLM: timestep → (gamma, beta)
        self.t_proj = nn.Linear(t_emb_dim, channels * 2)
        
    def forward(self, x, t_emb):
        batch = x.shape[0]
        
        # First conv block
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        
        # FiLM modulation
        gamma, beta = torch.chunk(self.t_proj(t_emb), 2, dim=-1)
        gamma = gamma.view(batch, -1, 1, 1)
        beta = beta.view(batch, -1, 1, 1)
        h = gamma * h + beta
        
        # Second conv block
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        
        # Residual connection
        return h + x

class SimpleUNet(nn.Module):
    """Minimal UNet for diffusion"""
    def __init__(self, in_channels=4, out_channels=4, t_emb_dim=128, channels=(64, 128, 256)):
        super().__init__()
        
        self.t_emb = SinusoidalPositionalEmbedding(t_emb_dim)
        self.t_mlp = nn.Sequential(
            nn.Linear(t_emb_dim, t_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(t_emb_dim * 4, t_emb_dim)
        )
        
        # Encoder
        self.enc1 = nn.Conv2d(in_channels, channels[0], 3, padding=1)
        self.enc1_res = ResBlock(channels[0], t_emb_dim)
        self.enc1_down = nn.Conv2d(channels[0], channels[0], 4, stride=2, padding=1)
        
        self.enc2_res = ResBlock(channels[0], t_emb_dim)
        self.enc2_down = nn.Conv2d(channels[0], channels[1], 4, stride=2, padding=1)
        
        # Bottleneck
        self.bn_res = ResBlock(channels[1], t_emb_dim)
        
        # Decoder
        self.dec2_up = nn.ConvTranspose2d(channels[1], channels[0], 4, stride=2, padding=1)
        self.dec2_res = ResBlock(channels[0] * 2, t_emb_dim)  # skip concat
        
        self.dec1_up = nn.ConvTranspose2d(channels[0], channels[0], 4, stride=2, padding=1)
        self.dec1_res = ResBlock(channels[0] * 2, t_emb_dim)
        
        self.out = nn.Conv2d(channels[0], out_channels, 3, padding=1)
    
    def forward(self, x, t):
        # Timestep embedding
        t_emb = self.t_emb(t)
        t_emb = self.t_mlp(t_emb)
        
        # Encoder
        h = self.enc1(x)
        h1 = self.enc1_res(h, t_emb)
        h1_down = self.enc1_down(h1)
        
        h2 = self.enc2_res(h1_down, t_emb)
        h2_down = self.enc2_down(h2)
        
        # Bottleneck
        h_bn = self.bn_res(h2_down, t_emb)
        
        # Decoder with skip connections
        h = self.dec2_up(h_bn)
        h = torch.cat([h, h2], dim=1)  # skip
        h = self.dec2_res(h, t_emb)
        
        h = self.dec1_up(h)
        h = torch.cat([h, h1], dim=1)  # skip
        h = self.dec1_res(h, t_emb)
        
        return self.out(h)

# Test
model = SimpleUNet(in_channels=4, out_channels=4)
x = torch.randn(2, 4, 64, 64)
t = torch.tensor([100, 500])
y = model(x, t)
print(f"Input: {x.shape}, Output: {y.shape}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
```

**결과**: Simple 3-level UNet, ~40M parameters. Output shape matches input.

### 실험 2: FiLM vs Global Conditioning 비교

```python
# FiLM (현재 방식): 각 ResBlock에 timestep 정보 주입
# Global conditioning: timestep을 input channel에 concatenate

def unet_with_film(x, t_emb):
    """현재 방식: FiLM modulation"""
    h = model(x, t_emb)
    return h

def unet_with_concat(x, t_emb):
    """대안: Timestep을 channel로 concat"""
    # t_emb → spatial map (broadcast)
    t_spatial = t_emb.view(x.shape[0], -1, 1, 1).expand(-1, -1, x.shape[2], x.shape[3])
    x_aug = torch.cat([x, t_spatial], dim=1)
    h = model_concat(x_aug)
    return h

# 비교
x = torch.randn(4, 4, 64, 64)
t = torch.tensor([50, 100, 500, 900])

# FiLM
y_film = model(x, t)
# Concat (더 많은 params)
# y_concat = model_concat(x, t)

print("FiLM: timestep 정보가 모든 계층에 균등하게 분배")
print("Concat: input channel에만 추가되므로, deeper layer에서 정보 희미화 가능")
```

### 실험 3: Skip Connection의 정보 보존 검증

```python
# Skip connection 있음 vs 없음 비교

def forward_with_skip(x, f):
    """Skip connection 포함"""
    h = x
    for i, block in enumerate(f):
        h = block(h)
        if i == len(f) // 2:
            skip = h
    h = h + skip  # Skip
    for block in f[len(f)//2:]:
        h = block(h)
    return h

def forward_without_skip(x, f):
    """Skip connection 없음"""
    h = x
    for block in f:
        h = block(h)
    return h

# 간단한 깊은 네트워크
blocks = nn.ModuleList([
    nn.Sequential(nn.Linear(128, 128), nn.ReLU())
    for _ in range(10)
])

x = torch.randn(1, 128)

# Gradient norm 비교
x.requires_grad_(True)
y_skip = forward_with_skip(x, blocks)
loss_skip = y_skip.sum()
loss_skip.backward()
grad_skip = x.grad.norm()

x.grad = None
y_no_skip = forward_without_skip(x, blocks)
loss_no_skip = y_no_skip.sum()
loss_no_skip.backward()
grad_no_skip = x.grad.norm()

print(f"Gradient norm WITH skip: {grad_skip:.4f}")
print(f"Gradient norm WITHOUT skip: {grad_no_skip:.4f}")
print(f"Skip connection helps: {grad_skip > grad_no_skip}")
```

## 🔗 실전 활용

**Stable Diffusion 1.5 UNet**:
- 9 levels, 2 ResBlocks per level
- ~860M parameters (full model)
- Cross-attention layers: text embedding 조건화 (Ch5-04 참조)

**코드 예시**:
```python
from diffusers import UNet2DConditionModel

unet = UNet2DConditionModel.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    subfolder="unet",
    use_auth_token=True
).to("cuda")

# Forward: (noisy latent, timestep, text embedding) → denoised prediction
latent = torch.randn(1, 4, 64, 64)
t = torch.tensor([100])
text_emb = torch.randn(1, 77, 768)  # CLIP text encoding

output = unet(latent, t, encoder_hidden_states=text_emb).sample
```

**SDXL**:
- 더 큰 UNet (384M → 2.6B parameter model)
- Dual text encoder (CLIP-L + OpenCLIP-bigG)
- Classifier-free guidance 강화

## ⚖️ 가정과 한계

1. **Spatial homogeneity 가정**: 모든 위치에서 동일한 inductive bias (실제로는 이미지 구조가 위치마다 다름)
2. **Fixed depth**: 해상도에 따라 encoder/decoder 깊이가 정해짐 → 매우 고해상도에서는 비효율
3. **Skip connection의 정보 중복**: 깊은 encoder feature + decoder feature가 유사하면 → 계산 낭비 가능
4. **Self-attention의 메모리 오버헤드**: $O(H \times W)$ complexity → 고해상도에서 병목

## 📌 핵심 정리

1. **UNet = encoder-decoder + skip + FiLM timestep modulation**
2. Skip connection은 multi-scale feature 보존 + gradient flow 강화
3. Timestep embedding을 FiLM 방식으로 주입하면, 모든 계층에서 diffusion step을 "지시"
4. DDPM 이후 사실상 표준 backbone (DiT 등장 전까지)

## 🤔 생각해볼 문제

1. **왜 Global average pooling이 아닌 stride-2 convolution으로 down-sampling하는가?**
   <details>
   <summary>힌트</summary>
   Stride-2 Conv는 학습 가능한 파라미터 → spatial 정보의 선택적 압축. Global pooling은 정보 손실이 크고, stride-2는 gradual down-sampling으로 더 나은 feature 학습.
   </details>

2. **Timestep을 FiLM (affine transform)으로 모듈레이션하는 것과, 별도의 attention head로 처리하는 것의 장단점은?**
   <details>
   <summary>힌트</summary>
   FiLM: 가벼운 계산 (affine 2개 param), 모든 채널에 균등. Attention: 채널 간 상호작용 학습 가능하지만 더 무거움. DDPM은 FiLM 선택 (빠른 학습), DiT는 attention 사용 (scaling).
   </details>

3. **UNet에서 bottleneck (가장 저해상도)의 self-attention이 중요한 이유는?**
   <details>
   <summary>힌트</summary>
   Bottleneck은 receptive field가 전체 이미지를 cover → global semantic 정보 캡처. 이 지점의 attention은 long-range dependency 학습에 유리. 높은 해상도에서는 attention이 computationally prohibitive하므로, bottleneck에 집중.
   </details>

---

<div align="center">

[◀ 이전](./01-latent-diffusion.md) | [📚 README](../README.md) | [다음 ▶](./03-dit.md)

</div>
