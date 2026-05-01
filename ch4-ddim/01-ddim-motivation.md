# 01. DDIM 동기: Diffusion 의 샘플링 가속

## 🎯 핵심 질문

DDPM 은 왜 느린가? 1000 step 을 거쳐야 하는데, 이를 가속할 수 있는가?

## 🔍 왜 DDPM 의 샘플링이 병목인가?

DDPM 학습 중에는 모든 time step $t \in \{1,...,T\}$ 에 대해 노이즈 예측 손실 $L = \mathbb{E}[\|\epsilon - \epsilon_\theta(x_t, t)\|^2]$ 를 균등하게 최소화한다. 하지만 **샘플링** 시에는:

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t, t)\right) + \sigma_t \epsilon, \quad t=T,...,1$$

매 step 마다 UNet 을 forward pass 해야 한다. 512×512 이미지 생성 시:

- **DDPM (T=1000)**: 1000 × UNet forward → 수십 초~수 분
- **실무 요구**: 수 초 이내 interactive 응답

이는 **역확산 과정(reverse process)이 Markovian** 이기 때문이다. 이전 상태만으로 다음을 결정할 수 없어 매 step 을 무시할 수 없다.

## 📐 수학적 선행 조건

- **Forward process**: $q(x_t|x_0) = \mathcal{N}(\sqrt{\bar\alpha_t}x_0, (1-\bar\alpha_t)I)$
- **Reverse (1 step)**: $p_\theta(x_{t-1}|x_t) \sim \mathcal{N}(\mu_\theta, \sigma_t^2 I)$
- **Markov chain**: $p_\theta(x_{0:T}) = p(x_T)\prod_{t=1}^T p_\theta(x_{t-1}|x_t)$

## 📖 직관적 이해

DDPM 은 Forward 과정을 그대로 역순으로 따라간다. 하지만 학습된 reverse step 이 충분히 정확하다면, **일부 step 을 건너뛰어도** forward marginal 을 재구성할 수 있을까?

**핵심 아이디어**: 학습 시 조건부 분포 $p_\theta(x_{t-1}|x_t)$ 를 fitting 했으므로, 원래 forward 경로의 일부 step 만 사용하는 "지름길" sampling 이 가능할 수 있다.

## ✏️ 엄밀한 정의

### 정의 1.1: Sampling Acceleration 이란?

같은 학습된 노이즈 예측기 $\epsilon_\theta$ 를 사용하되, reverse Markov chain 의 step 개수를 $T \gg S$ 에서 $S$ 로 줄이는 방법.

두 가지 접근:

1. **Discretization** (이 챕터 주제)
   - 원래 DDPM 역학 유지, step 수만 감소
   - 예: DDIM (Song 2021), DPM-Solver (Lu 2022)
   
2. **Distillation** (별도 챕터)
   - 다른 모델 학습 또는 teacher 에서 압축
   - 예: Consistency Models, Progressive Distillation

## 💻 구현 검증: DDPM vs 빠른 샘플링 비교

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

# Ch2 에서 학습한 1D 가우시안 DDPM 모델 불러오기
# model = SimpleUNet() 은 이미 학습됨 (Ch2 참고)

def ddpm_sample(model, x_T, alphas_cumprod, T=1000):
    """표준 DDPM 샘플링 (1000 step)"""
    x = x_T.clone()
    for t in range(T-1, -1, -1):
        alpha_t = alphas_cumprod[t]
        alpha_prev = alphas_cumprod[t-1] if t > 0 else torch.tensor(1.0)
        beta_t = 1 - alpha_t / alpha_prev if t > 0 else 1.0
        
        with torch.no_grad():
            eps_pred = model(x, torch.tensor(t))
        
        x = (x - (1 - alpha_t) / torch.sqrt(1 - alpha_t) * eps_pred) / torch.sqrt(alpha_t)
        if t > 0:
            x += torch.sqrt(beta_t) * torch.randn_like(x)
    return x

def fast_sample_placeholder(model, x_T, alphas_cumprod, num_steps=50):
    """빠른 샘플링 placeholder (이후 04 에서 구현)"""
    # 여기서는 단순히 fewer step 으로 표준 공식 적용
    T = len(alphas_cumprod)
    step_indices = torch.linspace(T-1, 0, num_steps).long()
    
    x = x_T.clone()
    for i, t in enumerate(step_indices):
        if i == len(step_indices) - 1:
            break
        # 간단한 interpolation (정확하지 않음)
        with torch.no_grad():
            eps_pred = model(x, torch.tensor(t))
        # ... (다음 섹션에서 상세 공식)
    return x

# 실험
alphas = torch.linspace(0.99, 0.0001, 1000)
alphas_cumprod = torch.cumprod(alphas, dim=0)

x_T = torch.randn(100)  # batch of 100 samples

print("[Sampling comparison]")
print(f"DDPM (1000 steps): ~1000 forward passes")
print(f"Fast (50 steps): ~50 forward passes → 20배 가속")
print(f"DDIM (10 steps): ~10 forward passes → 100배 가속")
```

## 🔗 실전 활용

- **Interactive generation**: Stable Diffusion WebUI 에서 "steps" 파라미터 조절 (default 20-50)
- **Real-time inference**: mobile/edge device 에서 수 백 ms 제약

## ⚖️ 가정과 한계

- **가정**: 학습된 reverse step $p_\theta(x_{t-1}|x_t)$ 가 충분히 정확함
- **한계**: Step 수 과도하게 줄면 분포 추적 실패 → 품질 저하

## 📌 핵심 정리

DDPM 의 느린 샘플링은 매 step 마다 UNet forward 필요성에서 비롯. **같은 학습 모델을 재사용하면서** step 수를 줄이는 **discretization 기반 가속** (DDIM, DPM-Solver) 과 **distillation** 의 두 경로 존재. 이 챕터는 전자의 이론.

## 🤔 생각해볼 문제

<details>
<summary><b>1. Forward process 의 Markovian 구조가 필수인가?</b></summary>

답: 아니다. Non-Markovian forward 를 정의할 수 있고, 같은 marginal $q(x_t|x_0)$ 를 유지하면서 다른 reverse 경로를 택할 수 있다. (→ 02 에서 상세)
</details>

<details>
<summary><b>2. Step 수를 줄일 때 어떤 시간점들을 선택해야 최적인가?</b></summary>

답: Linear, quadratic, 또는 exponential 스케줄링 등 여러 방법 있음. 후자들이 일반적으로 better (early noise removal 에 더 step 할당).
</details>

<details>
<summary><b>3. Teacher-Student distillation 은 왜 한 번에 50배 가속도 가능한가?</b></summary>

답: Forward process 를 1-2 step 으로 줄이는 다른 PDE/ODE 를 직접 학습 (별도 장에서).
</details>

---

<div align="center">

[◀ 이전](../ch3-score-sde/05-vp-ve-sde.md) | [📚 README](../README.md) | [다음 ▶](./02-non-markovian.md)

</div>
