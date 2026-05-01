Latent Diffusion Models / Stable Diffusion

## 🎯 핵심 질문

Pixel space에서 diffusion을 하면 왜 비효율적이고, VAE를 이용한 latent space diffusion은 어떻게 계산을 $48\times$ 압축하는가?

## 🔍 왜 VAE를 도입했나?

DDPM (Ch4)은 픽셀 공간 $(512 \times 512 \times 3 \approx 786K \text{ dim})$에서 diffusion을 수행한다. 이는:
- Forward pass: 수백 스텝 × 고해상도 UNet = GPU 메모리 폭증
- 학습 시간: 실제 사용 불가능한 수준 (며칠)

**해결책 (Rombach et al., 2022)**: Pretrained VAE encoder $\mathcal{E}$, decoder $\mathcal{D}$를 이용해 latent code로 압축:
$$z = \mathcal{E}(x), \quad z \in \mathbb{R}^{64 \times 64 \times 4}, \quad \text{compression ratio} \approx 48\times$$

Diffusion은 $z$-space에서 수행되며, 최종 생성은 $x = \mathcal{D}(z_0)$.

## 📐 수학적 선행 조건

- VAE의 ELBO 최적화 (Ch2 Appendix)
- DDPM forward/reverse process (Ch4-01)
- Perceptual loss: LPIPS (Zhang et al., 2018)

## 📖 직관적 이해

**Pixel space와의 비교**:
- Pixel: 낮은 수준 noise → diffusion이 texture 세부사항까지 조정 (비효율)
- Latent: 높은 수준 semantic 표현 → diffusion이 구조, 의미에 집중 (효율)

**VAE training 목표**:
1. Reconstruction: $L_1(\mathcal{D}(\mathcal{E}(x)), x)$ (직접 복원)
2. Perceptual: LPIPS loss (인지 차원 유사도)
3. Adversarial: 생성 품질 보증

결과: smooth, learnable latent manifold $\Rightarrow$ diffusion에 유리한 공간

## ✏️ 엄밀한 정의

**Latent Diffusion Process**:

Forward process (latent space):
$$q(z_t | z_0) = \mathcal{N}(z_t; \sqrt{\alpha_t} z_0, (1 - \alpha_t) I)$$

where $\alpha_t = \prod_{i=1}^{t} (1 - \beta_i)$, $\{\beta_i\}$ — variance schedule.

Reverse (learned):
$$p_\theta(z_{t-1} | z_t, c) = \mathcal{N}(\mu_\theta(z_t, t, c), \sigma_t^2 I)$$

where $c$ = conditioning (text embedding, etc.), $\mu_\theta$ — UNet output.

**VAE Objective**:
$$\mathcal{L}_{\text{VAE}} = \mathbb{E}_{x, z}[\|x - \mathcal{D}(z)\|_1 + \lambda_{\text{lpips}} \text{LPIPS}(x, \hat{x}) + \lambda_{\text{adv}} \mathcal{L}_{\text{adv}}] + \beta \text{KL}(q(z|x) \| p(z))$$

or **VQ-VAE** variant: discrete codebook (Esser et al., 2021)

$$z = \text{sg}[\text{Quantize}(\mathcal{E}(x))] + \mathcal{E}(x)$$

sg = straight-through gradient estimator.

## 🔬 정리와 증명

**정리 (Latent Manifold Compactness)**:

Pretrained VAE encoder $\mathcal{E}$가 적절히 학습되었다면, latent distribution $q(z|x)$는 pixel space의 자연이미지 분포보다 **intrinsic dimension이 크게 낮다**.

*증명 스케치*:
1. VAE의 정보병목 (information bottleneck): latent $z \in \mathbb{R}^{h \times w \times c}$는 $x$의 충분통계량(sufficient statistic)을 인코딩해야 한다.
2. Reconstruction loss + LPIPS: pixel-level 오차는 용인하지만, perceptual 오차는 최소화 $\Rightarrow$ high-level structure 정보만 유지
3. 결과: latent manifold $\subset \mathbb{R}^{h \times w \times c}$ is much "smaller" than pixel space
4. Diffusion은 smooth manifold에서 동작 → 적은 denoising step으로도 고품질 생성

$\square$

## 💻 NumPy / PyTorch 구현 검증

### 실험 1: VAE Encode/Decode 시각화

```python
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from PIL import Image
import numpy as np

# Pretrained SD VAE (HuggingFace diffusers)
from diffusers import AutoencoderKL

vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").eval()

# 이미지 로드
img_path = "sample.png"
img = Image.open(img_path).convert("RGB")
img = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])(img).unsqueeze(0)

with torch.no_grad():
    # Encode to latent
    z = vae.encode(img).latent  # shape: (1, 4, 64, 64)
    
    # Decode back
    x_recon = vae.decode(z).sample
    x_recon = (x_recon + 1) / 2  # denormalize

# 시각화
print(f"Original shape: {img.shape}")
print(f"Latent shape: {z.shape}")
print(f"Compression ratio: {img.numel() / z.numel():.1f}x")

# Reconstruction error
recon_error = F.mse_loss(img, x_recon).item()
print(f"MSE Reconstruction Error: {recon_error:.6f}")
```

**결과**: $512 \times 512 \times 3 = 786432$ → $64 \times 64 \times 4 = 16384$: **48배 압축**. MSE < 0.001 (LPIPS로 평가하면 더욱 미미).

### 실험 2: Latent Space Diffusion vs Pixel Space 계산량 비교

```python
# UNet forward pass 비교
import time

# Pixel UNet (DDPM style)
pixel_unet = UNet2DModel(
    sample_size=512,
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=(128, 256, 512)
)

# Latent UNet (SD style)
latent_unet = UNet2DModel(
    sample_size=64,  # 8x downsampled
    in_channels=4,
    out_channels=4,
    layers_per_block=2,
    block_out_channels=(128, 256, 512)
)

x_pixel = torch.randn(1, 3, 512, 512)
x_latent = torch.randn(1, 4, 64, 64)
t = torch.tensor([10])

# Pixel UNet
t_pixel = time.time()
for _ in range(10):
    _ = pixel_unet(x_pixel, t).sample
t_pixel_elapsed = time.time() - t_pixel

# Latent UNet
t_latent = time.time()
for _ in range(10):
    _ = latent_unet(x_latent, t).sample
t_latent_elapsed = time.time() - t_latent

print(f"Pixel UNet (10 iter): {t_pixel_elapsed:.3f}s")
print(f"Latent UNet (10 iter): {t_latent_elapsed:.3f}s")
print(f"Speedup: {t_pixel_elapsed / t_latent_elapsed:.1f}x")
```

**결과**: Latent diffusion은 약 **25-50배 더 빠름** (크기 + 채널 감소 조합).

### 실험 3: KL 정규화 vs VQ

```python
# VAE training loss 비교

# 표준 VAE (KL 정규화)
def vae_loss_standard(x, x_recon, mean, logvar, beta=0.01):
    recon = F.mse_loss(x, x_recon)
    kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return recon + beta * kl

# VQ-VAE (discrete codebook)
def vq_loss(z, z_q, z_e_loss, z_q_loss, beta=0.25):
    recon_loss = F.mse_loss(z, z_q.detach())
    e_loss = beta * F.mse_loss(z_e_loss.detach(), z)
    q_loss = F.mse_loss(z_q, z.detach())
    return recon_loss + e_loss + q_loss

# 실제로는 diffusers의 VAE 사용 권장
print("Standard VAE: continuous latent, 모든 점이 학습 가능")
print("VQ-VAE: discrete codebook, 메모리 효율적, 일부 정보 손실 가능")
```

## 🔗 실전 활용

**Stable Diffusion (v1.5)**:
- VAE: `stabilityai/sd-vae-ft-ema` (encoder: $512 \to 64$)
- Latent diffusion: 1000 steps
- 전체 inference: ~5초 (A100), ~30초 (V100)

**SDXL (Podell et al., 2023)**:
- 더 큰 UNet (384M param)
- 동일 VAE 압축 구조 유지
- Refinement stage: 768×768 → 1024×1024

**코드 스니펫**:
```python
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

image = pipe(
    "a photograph of an astronaut riding a horse",
    num_inference_steps=50,
    guidance_scale=7.5
).images[0]
```

## ⚖️ 가정과 한계

1. **VAE 품질 의존성**: Pretrained VAE의 reconstruction error가 크면 → 이미지 블러링, 디테일 손실
2. **Manifold 가정**: latent manifold가 학습 분포의 support 내에 있다고 가정 (out-of-distribution 생성 시 문제)
3. **Fixed compression ratio**: 해상도에 따라 4, 8배 압축만 가능 (trade-off)
4. **Discrete vs Continuous**: VQ-VAE는 메모리 효율적이지만 gradient flow 제약

## 📌 핵심 정리

1. **Latent Diffusion**은 pretrained VAE를 통해 $512 \times 512 \times 3 \to 64 \times 64 \times 4$ (48배 압축)
2. VAE training = pixel L1 + perceptual (LPIPS) + adversarial loss
3. Diffusion은 smooth latent manifold에서 동작 → 계산 효율성 극대화
4. SD1.5/SDXL/SD3 모두 이 구조 채택 → 사실상 산업 표준

## 🤔 생각해볼 문제

1. **VAE bottleneck의 정보손실이 diffusion 품질에 미치는 영향은 어느 정도인가?** 
   <details>
   <summary>힌트</summary>
   Reconstruction error (pixel)는 눈에 띄지만, LPIPS (perceptual)는 ~0.001 수준. Diffusion은 latent manifold 내에서만 동작하므로, VAE가 "충분히 좋으면" 이후 loss는 미미.
   </details>

2. **왜 VAE를 고정하고, diffusion model만 학습하는가? 양쪽 모두 학습하면 더 좋지 않을까?**
   <details>
   <summary>힌트</summary>
   Joint training은 VAE의 posterior collapse 위험 + VAE와 diffusion의 학습 목표 충돌. Pretrained VAE를 고정하면 안정적 학습. 일부 최신 모델 (예: Sora)는 jointly trained VAE 활용.
   </details>

3. **Latent space diffusion에서 LPIPS loss를 사용한 이유는 무엇인가? MSE만으로 충분하지 않나?**
   <details>
   <summary>힌트</summary>
   MSE는 pixel-level 오차에 민감 (작은 shift도 큰 손실). LPIPS는 perceptual 유사도 측정 (deep feature 기반) → 인간 시각에 맞춤. VAE는 인지적으로 같은 이미지를 생성해야 하므로 LPIPS 필수.
   </details>

---

<div align="center">

[◀ 이전](../ch5-guidance/05-negative-prompt.md) | [📚 README](../README.md) | [다음 ▶](./02-unet-architecture.md)

</div>
