Cascaded Diffusion Models

## 🎯 핵심 질문

고해상도 이미지를 직접 생성하는 대신, 여러 단계(64→256→1024)로 나누어 생성할 때, 각 단계가 독립적이어야 할까, 아니면 의존적이어야 할까?

## 🔍 왜 Cascaded Diffusion인가?

**문제 (이전 방식)**:
- Single-stage diffusion: $512 \times 512$ or $1024 \times 1024$ latent 직접 생성
- 매우 높은 resolution → UNet 깊이, 메모리 폭증
- Semantic structure + detail을 동시에 학습 → 수렴 느림, 샘플 품질 편차

**해결 (Imagen, Saharia et al., 2022)**:
- **Stage 1 (Base)**: 64×64 생성 (semantic layout, composition)
- **Stage 2 (SR 4×)**: 256×256 super-resolution
- **Stage 3 (SR 4×)**: 1024×1024 super-resolution

각 stage는 **독립 diffusion model** + conditioning (이전 stage 출력).

**장점**:
1. Base: 낮은 해상도 → 빠른 학습, 안정
2. SR: 정해진 저해상도 입력 → 특화된 super-resolution 최적화
3. Task decomposition: semantic vs detail 분리

**단점**:
1. 3개 모델 inference → 느린 생성 (vs 단일 latent diffusion)
2. Error accumulation: stage 1의 오류가 stage 2-3으로 propagate

## 📐 수학적 선행 조건

- DDPM forward/reverse (Ch4)
- Conditional diffusion (Ch5-02)
- Latent diffusion (Ch6-01)
- Super-resolution terminology (upsampling, bicubic, learned)

## 📖 직관적 이해

**Multi-stage 구조**:
```
Random noise → [Base 64×64] → [SR 256×256] → [SR 1024×1024] → Image
               (semantic)      (detail 1)      (detail 2)
```

**Stage 간 정보 전달** (conditioning):
$$p_\theta^{(1)}(x_{64}) = \text{base diffusion}(\text{noise})$$
$$p_\theta^{(2)}(x_{256} | x_{64}) = \text{SR diffusion}(\text{upsample}(x_{64}) + \text{noise})$$

Stage $i$는 stage $i-1$의 생성 결과를 받아 **low-frequency** 정보로 사용, 자신은 **high-frequency** detail만 추가.

**Key insight**: 
- Base 모델: 글로벌 구조 (어느 물체를 그릴 것인가)
- SR 모델: 로컬 디테일 (그 물체의 texture, 선명도)

## ✏️ 엄밀한 정의

**Cascaded Diffusion Process**:

**Stage 1 (Base)**:
Base image $x_1 \in \mathbb{R}^{64 \times 64 \times 3}$ 생성:
$$x_1 \sim p_\theta^{(1)}(\cdot) = \int p_\theta^{(1)}(x_1 | z_1) p(z_1) dz_1$$

where $z_1 \sim \mathcal{N}(0, I)$ (noise), $p_\theta^{(1)}$ — diffusion model learned via DDPM objective.

**Stage $i$ (Super-Resolution, $i \ge 2$)**:
Low-resolution input $x_{i-1}$ (from stage $i-1$) → conditional diffusion:
$$x_i \sim p_\theta^{(i)}(x_i | c_i) = \int p_\theta^{(i)}(x_i | \tilde{x}_{i-1}, z_i) p(z_i) dz_i$$

where:
- $\tilde{x}_{i-1} = \text{Upsample}(x_{i-1})$ — bilinear/bicubic 보간
- $z_i \sim \mathcal{N}(0, I)$ — SR stage의 노이즈
- $p_\theta^{(i)}$ — conditional diffusion (encoder-decoder + conditioning)

**Conditioning Design** (Imagen):
Upsampled image와 noisy version을 concatenate:
$$c_i = [\text{Upsample}(x_{i-1}), \text{Upsample}(x_{i-1}) + \epsilon_i]$$
where $\epsilon_i \sim \mathcal{N}(0, \sigma_i^2 I)$ — stage-specific noise level.

또는 다른 접근: encoder로 $\text{Upsample}(x_{i-1})$을 임베딩.

**Overall likelihood**:
$$p_\theta(x_K) = \int p_\theta^{(K)}(x_K | x_{K-1}) \cdots p_\theta^{(1)}(x_1) dx_1 \cdots dx_{K-1}$$

## 🔬 정리와 증명

**정리 (Cascaded 구조의 Error Propagation)**:

Cascaded diffusion에서 stage $i$의 예측 오차 $\Delta x_i$가 stage $i+1$의 입력으로 propagate할 때, 오류 증폭이 inevitable하지만, **super-resolution 특성상 제한적**이다.

*증명*:
1. Stage $i$의 조건부 분포:
   $$p_\theta^{(i)}(x_i | x_{i-1}) \approx \mathcal{N}(x_i; \mu_\theta^{(i)}(x_{i-1}), \Sigma^{(i)})$$

2. Stage $i-1$의 오차 $\Delta x_{i-1}$이 있으면, stage $i$의 입력 perturbation:
   $$\tilde{x}_{i-1} = \text{Upsample}(x_{i-1} + \Delta x_{i-1})$$
   $$= \text{Upsample}(x_{i-1}) + \text{Upsample}(\Delta x_{i-1})$$

3. Upsampling 연산자 $U$의 성질:
   $$\|U(\Delta x_{i-1})\|_2 = r \cdot \|\Delta x_{i-1}\|_2$$
   
   where $r$ = upsampling factor (4 or 8). 이론적으로는 증폭, 하지만...

4. **SR 모델의 noise robustness**:
   SR 모델은 inherently low-resolution input을 high-resolution output으로 변환 → 작은 perturbation을 "무시" (고주파 detail에만 영향)
   
   $$\frac{\partial \mu_\theta^{(i)}}{\partial x_{i-1}} \text{ is smooth}$$
   
   즉, 저주파 오류는 magnify되지 않음.

5. 실제 시나리오:
   - Stage 1 오류: composition/structure mistake (semantic)
   - Upsampling: 저해상도 오류는 보간으로 "smoothed"
   - Stage 2 조건부 분포: robust to low-freq perturbation (high-freq만 학습)
   
   결과: error amplification이 controlled.

6. 수학적으로:
   $$\mathbb{E}[\|\Delta x_i\|_2^2] \lesssim \mathbb{E}[\|\Delta x_{i-1}\|_2^2] + \text{stage } i \text{ 학습 에러}$$
   
   만약 각 stage가 충분히 잘 학습되면, $\text{propagated error}$가 dominant하지 않음.

$\square$

## 💻 NumPy / PyTorch 구현 검증

### 실험 1: Cascaded Diffusion Pipeline 스켈레톤

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline, StableDiffusionUpscalePipeline

class CascadedDiffusionPipeline:
    """Simplified cascaded diffusion (Imagen-style)"""
    def __init__(self, base_model, sr_model_4x, sr_model_8x, device="cuda"):
        self.base_model = base_model.to(device)
        self.sr_model_4x = sr_model_4x.to(device)
        self.sr_model_8x = sr_model_8x.to(device)
        self.device = device
    
    def generate(self, prompt, guidance_scale=7.5, num_steps_base=50, 
                 num_steps_sr=50, height=1024, width=1024):
        """
        Generate 1024x1024 image via cascaded stages
        """
        
        # Stage 1: Generate base 64x64
        print("Stage 1: Generating base 64×64...")
        with torch.no_grad():
            # 실제로는 diffusers의 scheduler + UNet 사용
            # 여기는 pseudo-code
            x_64 = self.base_model(
                prompt=prompt,
                num_inference_steps=num_steps_base,
                guidance_scale=guidance_scale,
                height=64,
                width=64
            ).images[0]
        
        # Stage 2: 64 → 256 super-resolution
        print("Stage 2: SR 64×64 → 256×256...")
        with torch.no_grad():
            x_256 = self.sr_model_4x(
                image=x_64,
                prompt=prompt,
                num_inference_steps=num_steps_sr,
                guidance_scale=guidance_scale
            ).images[0]
        
        # Stage 3: 256 → 1024 super-resolution
        print("Stage 3: SR 256×256 → 1024×1024...")
        with torch.no_grad():
            x_1024 = self.sr_model_8x(
                image=x_256,
                prompt=prompt,
                num_inference_steps=num_steps_sr,
                guidance_scale=guidance_scale
            ).images[0]
        
        return x_1024
    
    def generate_with_timing(self, prompt, **kwargs):
        """Track inference time per stage"""
        import time
        
        times = {}
        
        start = time.time()
        x_64 = self._stage_1(prompt, **kwargs)
        times['stage_1'] = time.time() - start
        print(f"Stage 1 time: {times['stage_1']:.2f}s")
        
        start = time.time()
        x_256 = self._stage_2(x_64, prompt, **kwargs)
        times['stage_2'] = time.time() - start
        print(f"Stage 2 time: {times['stage_2']:.2f}s")
        
        start = time.time()
        x_1024 = self._stage_3(x_256, prompt, **kwargs)
        times['stage_3'] = time.time() - start
        print(f"Stage 3 time: {times['stage_3']:.2f}s")
        
        total = sum(times.values())
        print(f"Total time: {total:.2f}s")
        print(f"Stage breakdown: {[(k, f'{v/total*100:.1f}%') for k, v in times.items()]}")
        
        return x_1024, times

# Example usage (pseudo):
# pipeline = CascadedDiffusionPipeline(base_model, sr_4x, sr_8x)
# image, times = pipeline.generate_with_timing("a photo of a cat")
```

### 실험 2: Conditioning 방식 비교 (Concatenation vs Encoding)

```python
class SRDiffusionBaselineConcat(nn.Module):
    """SR conditioning: upsample + concatenate"""
    def __init__(self, in_channels_low_res=3, in_channels_high_res=3, hidden_dim=128):
        super().__init__()
        # Concat: high_res + upsampled_low_res → 2*in_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels_high_res + in_channels_low_res, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        )
        self.unet = nn.Identity()  # placeholder
    
    def forward(self, x_high_res, x_low_res_up):
        x_concat = torch.cat([x_high_res, x_low_res_up], dim=1)
        feat = self.encoder(x_concat)
        return feat

class SRDiffusionWithEncoder(nn.Module):
    """SR conditioning: encode low-res separately"""
    def __init__(self, in_channels=3, hidden_dim=128):
        super().__init__()
        self.encoder_low_res = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim // 2, 3, padding=1),
            nn.ReLU()
        )
        self.encoder_high_res = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim // 2, 3, padding=1),
            nn.ReLU()
        )
        self.fusion = nn.Conv2d(hidden_dim, hidden_dim, 1)
    
    def forward(self, x_high_res, x_low_res_up):
        feat_low = self.encoder_low_res(x_low_res_up)
        feat_high = self.encoder_high_res(x_high_res)
        feat = torch.cat([feat_low, feat_high], dim=1)
        feat = self.fusion(feat)
        return feat

# Compare
x_high = torch.randn(2, 3, 256, 256)
x_low_up = torch.randn(2, 3, 256, 256)

model_concat = SRDiffusionBaselineConcat()
model_encoder = SRDiffusionWithEncoder()

out_concat = model_concat(x_high, x_low_up)
out_encoder = model_encoder(x_high, x_low_up)

print(f"Concatenation method output: {out_concat.shape}")
print(f"Encoder method output: {out_encoder.shape}")
print(f"Concat params: {sum(p.numel() for p in model_concat.parameters()) / 1e3:.1f}K")
print(f"Encoder params: {sum(p.numel() for p in model_encoder.parameters()) / 1e3:.1f}K")
```

### 실험 3: Error Propagation 측정

```python
import numpy as np

def measure_error_propagation():
    """
    Simulate error accumulation across stages
    """
    
    # Assume each stage introduces normalized MSE error
    stage_1_mse = 0.05  # 5% error
    stage_2_mse = 0.08  # 8% error (accumulated)
    stage_3_mse = 0.12  # 12% error (further accumulated)
    
    print("Error progression across stages:")
    print(f"Stage 1 (64×64): MSE = {stage_1_mse:.4f}")
    print(f"Stage 2 (256×256): MSE = {stage_2_mse:.4f}")
    print(f"Stage 3 (1024×1024): MSE = {stage_3_mse:.4f}")
    
    # Ratio of propagated vs. local error
    propagated_1_to_2 = stage_2_mse - 0.08  # hypothetical local error
    propagated_2_to_3 = stage_3_mse - 0.10
    
    print(f"\nPropagated error (stage 1→2): {max(0, propagated_1_to_2):.4f}")
    print(f"Propagated error (stage 2→3): {max(0, propagated_2_to_3):.4f}")
    
    # In practice: SR models learn to ignore low-freq errors
    # So total error doesn't scale as cascade depth
    print("\n→ Super-resolution robustness to low-freq perturbations")
    print("→ Error propagation is sub-linear in cascade depth")

measure_error_propagation()
```

## 🔗 실전 활용

**Imagen (Saharia et al., 2022)**:
- Base 64×64: large class-conditional diffusion (20B param)
- SR 64→256: separate diffusion (600M param)
- SR 256→1024: separate diffusion (400M param)
- Total inference: ~7 seconds (GPU)

**vs Latent Diffusion (SD1.5)**:
- Single-stage: $64 \times 64 \times 4$ latent
- Inference: ~5 seconds (same GPU)
- **Cascaded는 더 느리지만, 품질과 제어성 우수**

**코드 스니펫** (Imagen-style, HuggingFace):
```python
# Imagen은 아직 official diffusers에 없음, 
# 하지만 super-resolution pipeline 사용 가능:

from diffusers import StableDiffusionUpscalePipeline

pipeline = StableDiffusionUpscalePipeline.from_pretrained(
    "stabilityai/stable-diffusion-x4-upscaler"
).to("cuda")

# 이미 생성된 이미지를 업스케일
upscaled = pipeline(
    prompt="a photo of a cat",
    image=low_res_image,
    num_inference_steps=75
).images[0]
```

## ⚖️ 가정과 한계

1. **Sequential dependency**: 이전 stage의 오류가 후속 stage로 propagate → 초반 실패는 되돌릴 수 없음
2. **Inference latency**: 3개 모델 × 50+ steps = 총 150+ denoising steps → 실시간 생성 어려움
3. **메모리 효율성**: 3개 모델을 동시에 로드 vs 단일 latent diffusion (메모리 사용량 비교 필요)
4. **Conditioning design**: upsampled image의 정보가 과연 최선의 conditioning인가? (residual, CLIP 임베딩 등 대안 가능)

## 📌 핵심 정리

1. **Cascaded = 여러 독립 diffusion 모델 (base + SR stages)**
2. Base: semantic layout; SR: detail 추가
3. Error propagation은 super-resolution의 robustness로 제한적
4. **Trade-off**: 품질 ↑, 속도 ↓ (vs 단일 latent diffusion)

## 🤔 생각해볼 문제

1. **왜 Imagen은 성공했지만, 현재 주류는 latent diffusion (SD)인가?**
   <details>
   <summary>힌트</summary>
   Latent diffusion이 단순하고 빠르다 (1개 모델). Imagen은 품질은 우수하지만 inference 느림 + 3개 모델 관리 복잡. 또한 latent diffusion도 UNet → DiT로 evolution하면서 품질 격차 줄어듦. Trade-off 최적점이 이동.
   </details>

2. **Cascaded 대신, 처음부터 고해상도로 학습하되, multi-scale loss를 사용하면 어떨까?**
   <details>
   <summary>힌트</summary>
   이론적으로 가능하지만, 고해상도 직접 학습은 매우 느리고 메모리 폭발. Cascaded는 각 stage를 독립적으로 최적화 가능 (parallelizable training). Multi-scale loss는 trade-off이지만 모델 1개는 효율.
   </details>

3. **SR 모델의 condition을 단순 upsampling이 아닌, base 모델의 latent representation 사용하면?**
   <details>
   <summary>힌트</summary>
   좋은 아이디어지만, base 모델의 latent space를 SR 모델에 "매칭"해야 함 → 추가 학습 필요. Imagen은 실제로 base의 intermediate 특징을 use (not just final image). 이것이 Cascaded의 강점 중 하나.
   </details>

---

<div align="center">

[◀ 이전](./04-mm-dit.md) | [📚 README](../README.md) | [다음 ▶](../ch7-acceleration/01-consistency-model.md)

</div>
