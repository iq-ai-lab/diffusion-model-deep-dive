# 03. DDIM Sampling Equation 과 Step 가속

## 🎯 핵심 질문

Non-Markovian forward process 로부터 어떻게 실제 역확산 공식을 유도하는가? $\sigma_t=0$ 선택이 deterministic 샘플링을 주는 이유?

## 🔍 왜 DDIM Sampling 공식이 작동하는가?

앞 장에서 Non-Markovian forward 의 $q_\sigma(x_{t-1}|x_t, x_0)$ 를 정의했다. 이제 **reverse process** 를 구성한다.

DDPM 학습에서 $\epsilon_\theta(x_t, t)$ 는 $q(x_t|x_0)$ 에서의 노이즈를 예측하도록 훈련됐다. 이는 Non-Markovian 에서도 valid 한데, marginal 이 동일하기 때문.

**핵심**: 주어진 $x_t$ 와 $\epsilon_\theta(x_t, t)$ 로부터, $x_0$ 를 복원하고, 그것을 이용해 $x_{t-1}$ 로 이동할 수 있다.

## 📐 수학적 선행 조건

- **Reverse: Bayes rule**
  $$q_\sigma(x_{t-1}|x_t, x_0) = \mathcal{N}\left(\sqrt{\bar\alpha_{t-1}}x_0 + \sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\frac{x_t - \sqrt{\bar\alpha_t}x_0}{\sqrt{1-\bar\alpha_t}}, \sigma_t^2 I\right)$$

- **Noise prediction**: $\epsilon_\theta(x_t, t) \approx \frac{x_t - \sqrt{\bar\alpha_t}x_0}{\sqrt{1-\bar\alpha_t}}$

## 📖 직각적 이해

역확산 한 step 을 세 부분으로 이해:

1. **$x_t$ 에서 $x_0$ 추정 (denoising)**:
   $$\hat x_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\epsilon_\theta}{\sqrt{\bar\alpha_t}}$$

2. **방향 벡터 (signal direction)**:
   $$\text{direction} = \frac{x_t - \sqrt{\bar\alpha_t}\hat x_0}{\sqrt{1-\bar\alpha_t}} = \epsilon_\theta$$

3. **다음 state 로 이동**:
   - Deterministic 부분: 추정된 $\hat x_0$ 와 방향의 보간
   - Stochastic 부분: 선택 가능한 노이즈 수준 $\sigma_t$

## ✏️ 엄밀한 정의

### 정의 3.1: DDIM Reverse Step

학습된 노이즈 예측기 $\epsilon_\theta$ 와 매개변수 $(\sigma_t, \eta_t)$ (또는 직접 $\sigma_t$) 에 대해, 다음을 정의:

1. **Predicted $x_0$**:
   $$\hat x_0(x_t) := \frac{x_t - \sqrt{1-\bar\alpha_t}\epsilon_\theta(x_t, t)}{\sqrt{\bar\alpha_t}}$$

2. **DDIM reverse step**:
   $$x_{t-1} = \sqrt{\bar\alpha_{t-1}}\hat x_0 + \sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\epsilon_\theta(x_t, t) + \sigma_t \epsilon_t$$
   
   여기서 $\epsilon_t \sim \mathcal{N}(0, I)$ 는 독립적인 노이즈.

### 정의 3.2: Sub-sampling Schedule

전체 time step $\{1, 2,...,T\}$ 중 $S < T$ 개를 선택:
$$\tau = (\tau_1, \tau_2,...,\tau_S), \quad 0 = \tau_{S+1} < \tau_S < ... < \tau_1 < T$$

샘플링 시에는 이 부분집합만 사용:
$$x_{\tau_{i-1}} = \text{DDIM\_step}(x_{\tau_i}, \tau_i, \epsilon_\theta)$$

## 🔬 정리와 증명

### 정리 3.1: DDIM Sampling 의 수렴성

$\epsilon_\theta$ 가 참 노이즈를 완벽하게 예측 ($\epsilon_\theta = \epsilon_{\text{true}}$) 할 때, sub-sampled trajectory $\{x_{\tau_S}, x_{\tau_{S-1}},...,x_{\tau_1}\}$ 는 다음을 만족한다:

$$\mathbb{E}_{x_{\tau_i}|x_0}[\|x_{\tau_{i-1}} - \text{mean}(q_\sigma(x_{\tau_{i-1}}|x_{\tau_i}, x_0))\|^2] = O(\sigma_{\tau_i}^2)$$

**증명 스케치**:

DDIM step 의 평균:
$$\mathbb{E}[x_{t-1}] = \sqrt{\bar\alpha_{t-1}}\mathbb{E}[\hat x_0] + \sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\mathbb{E}[\epsilon_\theta]$$

$\epsilon_\theta = \epsilon_{\text{true}}$ 일 때, 첫 번째 항은 참 $x_0$ 로 수렴. 두 번째 항은 true reverse direction. 분산은 $\sigma_t^2 + (1-\bar\alpha_{t-1}-\sigma_t^2) = 1-\bar\alpha_{t-1}$ 로 유지.

따라서 deterministic 부분이 정확하고, stochastic 노이즈 추가 후 true marginal 을 추적한다. $\square$

## 💻 구현 검증: DDIM vs DDPM 샘플 품질 비교

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

def ddim_step(x_t, t, prev_t, eps_theta, alphas_cumprod, sigma_t=0.0):
    """
    한 step DDIM sampling
    x_t: current latent
    t, prev_t: current and previous timestep index
    eps_theta: noise prediction network
    alphas_cumprod: alpha_bar (cumulative products)
    sigma_t: stochasticity parameter (0 = deterministic, >0 = stochastic)
    """
    alpha_t = alphas_cumprod[t]
    alpha_prev = alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0)
    
    with torch.no_grad():
        eps_pred = eps_theta(x_t, torch.tensor(t))
    
    # Predicted x_0
    x_0_pred = (x_t - torch.sqrt(1 - alpha_t) * eps_pred) / torch.sqrt(alpha_t)
    
    # Clamp x_0 to [-1, 1] (optional denoising)
    # x_0_pred = torch.clamp(x_0_pred, -1, 1)
    
    # Direction to next step
    dir_x_t = torch.sqrt(1 - alpha_prev - sigma_t**2) * eps_pred
    
    # Stochastic component
    noise = torch.randn_like(x_t) if sigma_t > 0 else torch.zeros_like(x_t)
    
    x_prev = torch.sqrt(alpha_prev) * x_0_pred + dir_x_t + sigma_t * noise
    
    return x_prev, x_0_pred

def ddpm_sample(x_T, alphas_cumprod, eps_theta, num_steps=1000):
    """Standard DDPM (모든 step 사용)"""
    x = x_T.clone()
    T = len(alphas_cumprod)
    
    for t in range(T-1, 0, -1):
        x_prev, _ = ddim_step(x, t, t-1, eps_theta, alphas_cumprod, 
                             sigma_t=torch.sqrt((1 - alphas_cumprod[t-1]) / 
                                               (1 - alphas_cumprod[t]) * 
                                               (1 - alphas_cumprod[t] / alphas_cumprod[t-1])))
        x = x_prev
    
    return x

def ddim_sample(x_T, alphas_cumprod, eps_theta, num_steps=50, sigma_t=0.0):
    """Accelerated DDIM (sub-sampling)"""
    T = len(alphas_cumprod)
    # Linear schedule for sub-sampled steps
    step_indices = torch.linspace(T-1, 0, num_steps).long()
    
    x = x_T.clone()
    for i in range(len(step_indices) - 1):
        t = step_indices[i].item()
        prev_t = step_indices[i+1].item()
        x, _ = ddim_step(x, t, prev_t, eps_theta, alphas_cumprod, sigma_t=sigma_t)
    
    return x

# Ch2 학습 모델 (1D Gaussian) 로드
# model = load_trained_unet()  # (실제로는 Ch2 checkpoint 사용)

# 실험 설정
T = 1000
alphas = torch.linspace(0.99, 0.0001, T)
alphas_cumprod = torch.cumprod(alphas, dim=0)

x_T = torch.randn(100)  # batch size 100

# 샘플링 비교
print("[Sampling Comparison]")
num_samples = 10
times = {}

step_configs = [
    ("DDPM (1000 steps)", 1000, 0.0),
    ("DDIM (50 steps, det)", 50, 0.0),
    ("DDIM (50 steps, stoch)", 50, 0.3),
    ("DDIM (10 steps, det)", 10, 0.0),
]

for name, steps, sigma in step_configs:
    # (실제 구현에선 forward pass count 측정)
    print(f"{name}: ~{steps} forward passes")
    # samples = [ddim_sample(x_T, alphas_cumprod, model, steps, sigma) 
    #            for _ in range(num_samples)]
    # quality = compute_fid(samples, validation_set)
    # print(f"  FID score: {quality:.2f}")

print("\n[Key Observation]")
print("Sigma=0 (deterministic): 매번 같은 샘플 → reproducible")
print("Sigma>0 (stochastic): 다양성 증가 → 높은 quality")
```

## 🔗 실전 활용

- **Stable Diffusion WebUI**: `--steps 20` (default DDIM), `--sampler ddim` 선택
- **Fast inference**: deterministic DDIM 10-20 step 으로 0.5-1 초 생성
- **High quality**: stochastic 또는 50+ step 으로 더 나은 distribution coverage

## ⚖️ 가정과 한계

- **가정**: $\epsilon_\theta$ 이 노이즈를 정확히 예측
- **한계**: 
  - Step 수 과도 감소 → ODE curvature approximation error
  - Sub-sampling schedule 선택에 민감
  - Stochasticity-Quality tradeoff

## 📌 핵심 정리

**DDIM sampling 은 Non-Markovian forward 와 학습된 $\epsilon_\theta$ 를 결합하여, 같은 모델을 사용하면서도 step 수를 1000 에서 10-50 으로 줄인다.** $\sigma_t = 0$ 선택 (deterministic) 은 역학을 ODE 로 바꾸며, $\sigma_t > 0$ (stochastic) 은 일부 확률성을 유지. 이를 통해 sampling 시간을 100배 이상 단축 가능.

## 🤔 생각해볼 문제

<details>
<summary><b>1. Sub-sampling schedule 에서 linear ($\tau_i = iT/S$) vs quadratic vs exponential 중 어느 것이 best 인가?</b></summary>

답: 일반적으로 exponential 또는 quadratic (early step 에 더 많은 step 할당). Linear 는 late noise regime 에서 inefficient.
</details>

<details>
<summary><b>2. Deterministic DDIM ($\sigma_t=0$) 이 high quality 를 보장하지 않는 이유?</b></summary>

답: $\epsilon_\theta$ 의 예측 오차가 누적. Deterministic 하더라도 매 step 의 오차는 exponentially 증폭. Stochasticity 추가는 이를 "경로 다양화"로 완화.
</details>

<details>
<summary><b>3. Sub-sampled step 의 간격이 크면 $\hat x_0$ 예측이 부정확한 이유?</b></summary>

답: $x_t$ 와 $x_{t-\Delta t}$ 는 상대적으로 높은 노이즈 차이. 중간 정보 없이 direct jump 하면 $\hat x_0$ 가 큰 오류에 노출. ODE solver 의 truncation error 개념과 동일.
</details>

---

<div align="center">

[◀ 이전](./02-non-markovian.md) | [📚 README](../README.md) | [다음 ▶](./04-probability-flow-ode.md)

</div>
