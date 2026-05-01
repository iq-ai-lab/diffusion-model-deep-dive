# 03. Forward Closed-Form: $q(x_t \mid x_0)$

## 🎯 핵심 질문

- Forward process 를 한 번에 계산할 수 있는가? (여러 스텝 거치지 않고)
- Closed-form 을 유도하려면 어떤 수학이 필요한가?
- $\alpha_t := 1-\beta_t, \bar\alpha_t := \prod_{s=1}^t \alpha_s$ 정의가 왜 이렇게 특별한가?
- Reparameterization 을 통해 sampling 을 어떻게 간단히 할 수 있는가?

## 🔍 왜 이 Closed-Form 이 DDPM 의 핵심인가

실제로 reverse process 를 학습할 때, 우리는:
$$L = \mathbb{E}_{x_0, t, \epsilon} \left[ \|\epsilon - \epsilon_\theta(x_t, t)\|^2 \right]$$

를 최소화한다. 이를 위해서는 **임의의 $t$ 에 대해 $x_t$ 를 빠르게 계산** 할 수 있어야 한다.

Markov chain 을 따라가면서 $t-1$ 번 계산하는 것은 너무 느리다.

**Closed-form** 을 알면, **시간 $t$ 를 샘플링하고 직접 $x_t$ 를 생성** 할 수 있다.
이것이 DDPM 학습을 실용적으로 만드는 핵심이다.

## 📐 수학적 선행 조건

- **Gaussian Algebra** : 두 Gaussian 의 합은 Gaussian
- **Variance of Sum** : $\text{Var}(aX + bY) = a^2\text{Var}(X) + b^2\text{Var}(Y)$ (독립일 때)
- **Induction** : 귀납법으로 일반화
- **Reparameterization Trick** : $\epsilon \sim \mathcal{N}(0,I)$ 로 parameterize
- **Ch1-02** : Forward Markov chain 정의

## 📖 직관적 이해

```
직관: 여러 번의 노이즈 누적

x₀ = 신호 (예: 사진)
x₁ = √(1-β₁) x₀ + √β₁ ε₁
x₂ = √(1-β₂) x₁ + √β₂ ε₂
   = √(1-β₂)[√(1-β₁) x₀ + √β₁ ε₁] + √β₂ ε₂
   = √(1-β₁)(1-β₂) x₀ + √(1-β₁)β₁ ε₁ + √β₂ ε₂
   
   ← 노이즈 항들 (ε₁, ε₂) 합치기
     두 독립 Gaussian 의 합 = 더 큰 분산의 단일 Gaussian

x₂ = √(1-β₁)(1-β₂) x₀ + √[(1-β₁)β₁ + β₂] ε'
     (where ε' 는 새로운 표준 정규분포)

일반화:
x_t = √[∏(1-β_s)] x₀ + √[1 - ∏(1-β_s)] ε
    = √(ᾱ_t) x₀ + √(1-ᾱ_t) ε
```

**핵심 아이디어**: 노이즈를 재조합하면, 결국 하나의 큰 노이즈가 되고,
그 분산은 누적된 $\beta$ 값들의 함수다.

## ✏️ 엄밀한 정의

### 정의 3.1: Cumulative Product Notation

- $\alpha_t := 1 - \beta_t$ (단일 스텝의 신호 보존 비율)
- $\bar{\alpha}_t := \prod_{s=1}^t \alpha_s = \alpha_1 \alpha_2 \cdots \alpha_t$ (누적)
- $\bar{\alpha}_0 := 1$ (정의상)

예: $T=3$, $\beta = [0.001, 0.002, 0.005]$ 이면
$$\alpha = [0.999, 0.998, 0.995]$$
$$\bar{\alpha} = [0.999, 0.999 \times 0.998, 0.999 \times 0.998 \times 0.995] = [0.999, 0.997, 0.992]$$

### 정의 3.2: Forward Closed-Form

**정리 3.1 (Sohl-Dickstein et al. 2015)**:

$$q(x_t \mid x_0) = \mathcal{N}\left( \sqrt{\bar{\alpha}_t} x_0, \, (1 - \bar{\alpha}_t) I \right)$$

**동치 표현** (Reparameterization):

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

이 하나의 식으로 Markov chain 의 모든 스텝을 건너뛸 수 있다.

## 🔬 정리와 증명

### 정리 3.1: Closed-Form Derivation (귀납법)

**명제**: Forward Markov chain
$$x_t = \sqrt{1-\beta_t} x_{t-1} + \sqrt{\beta_t} \epsilon_t$$

을 만족할 때, 다음이 성립한다:
$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$$

여기서 $\epsilon \sim \mathcal{N}(0, I)$ 는 단일 표준 정규분포.

**증명** (강한 귀납법):

*Base case* ($t=1$):
$$x_1 = \sqrt{1-\beta_1} x_0 + \sqrt{\beta_1} \epsilon_1$$

$\alpha_1 = 1 - \beta_1, \bar{\alpha}_1 = \alpha_1$ 이므로:
$$x_1 = \sqrt{\bar{\alpha}_1} x_0 + \sqrt{1 - \bar{\alpha}_1} \epsilon_1 \quad \checkmark$$

*Induction step*: $t-1$ 에서 성립하면 $t$ 도 성립함을 보이자.

**Induction Hypothesis** (IH): 
$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} x_0 + \sqrt{1 - \bar{\alpha}_{t-1}} \epsilon'$$

여기서 $\epsilon' \sim \mathcal{N}(0, I)$.

**목표**: $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon''$ 를 얻기.

**계산**:
$$x_t = \sqrt{1-\beta_t} x_{t-1} + \sqrt{\beta_t} \epsilon_t$$

IH 대입:
$$x_t = \sqrt{1-\beta_t} \left[ \sqrt{\bar{\alpha}_{t-1}} x_0 + \sqrt{1 - \bar{\alpha}_{t-1}} \epsilon' \right] + \sqrt{\beta_t} \epsilon_t$$

$$= \sqrt{(1-\beta_t) \bar{\alpha}_{t-1}} x_0 + \sqrt{(1-\beta_t)(1 - \bar{\alpha}_{t-1})} \epsilon' + \sqrt{\beta_t} \epsilon_t$$

이제 $\alpha_t = 1 - \beta_t$ 를 사용:
$$= \sqrt{\alpha_t \bar{\alpha}_{t-1}} x_0 + \sqrt{\alpha_t(1 - \bar{\alpha}_{t-1}) + \beta_t} \epsilon_{\text{new}}$$

여기서 $\epsilon_{\text{new}}$ 는 두 독립 Gaussian 의 합을 하나로 정규화한 것:
$$\epsilon_{\text{new}} = \frac{\sqrt{\alpha_t(1 - \bar{\alpha}_{t-1})} \epsilon' + \sqrt{\beta_t} \epsilon_t}{\sqrt{\alpha_t(1 - \bar{\alpha}_{t-1}) + \beta_t}} \sim \mathcal{N}(0, I)$$

variance 계산:
$$\alpha_t(1 - \bar{\alpha}_{t-1}) + \beta_t = (1-\beta_t)(1 - \bar{\alpha}_{t-1}) + \beta_t$$

$$= 1 - \bar{\alpha}_{t-1} - \beta_t(1 - \bar{\alpha}_{t-1}) + \beta_t$$

$$= 1 - \bar{\alpha}_{t-1} - \beta_t + \beta_t \bar{\alpha}_{t-1} + \beta_t$$

$$= 1 - \bar{\alpha}_{t-1} + \beta_t \bar{\alpha}_{t-1}$$

$$= 1 - \bar{\alpha}_{t-1}(1 - \beta_t)$$

$$= 1 - \bar{\alpha}_{t-1} \alpha_t$$

$$= 1 - \bar{\alpha}_t$$

왜냐하면 $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s = \bar{\alpha}_{t-1} \cdot \alpha_t$.

따라서:
$$x_t = \sqrt{\alpha_t \bar{\alpha}_{t-1}} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon_{\text{new}}$$

$$= \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon_{\text{new}}$$

따라서 IH 가 $t$ 에서도 성립. 수학적 귀납법으로 증명 완료.

$$\square$$

### 정리 3.2: Sampling Complexity Reduction

**명제**: Closed-form 을 이용하면 sampling 시간이 $O(T)$ 에서 $O(1)$ 로 감소.

*증명 스케치*:
- Markov chain: $x_0 → x_1 → \cdots → x_T$ (반복 계산, $O(T)$ 시간)
- Closed-form: $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$ (직접, $O(1)$ 시간)

## 💻 NumPy / PyTorch 구현 검증

### 실험 1: Closed-Form vs Markov Chain (동등성)

```python
import torch
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

# Setup
T = 100
beta = torch.linspace(0.0001, 0.02, T)
alpha = 1 - beta
alpha_bar = torch.cumprod(alpha, dim=0)

# 초기 신호
x0 = torch.randn(10)  # Batch of 10

# Method 1: Markov chain (sequential)
x_markov = x0.clone()
for t in range(T):
    epsilon = torch.randn(10)
    x_markov = torch.sqrt(alpha[t]) * x_markov + torch.sqrt(beta[t]) * epsilon

# Method 2: Closed-form (direct)
epsilon_global = torch.randn(10)
x_closed = torch.sqrt(alpha_bar[T-1]) * x0 + torch.sqrt(1 - alpha_bar[T-1]) * epsilon_global

# 둘 다 같은 분포를 따라야 함 (한 번 샘플링이므로 값은 다르지만, variance/mean 같음)
print("Markov chain result:")
print(f"  mean: {x_markov.mean():.4f}, var: {x_markov.var():.4f}")
print("\nClosed-form result:")
print(f"  mean: {x_closed.mean():.4f}, var: {x_closed.var():.4f}")
print(f"\nTheoretical (N(0, 1)): mean ≈ 0, var ≈ 1")
```

### 실험 2: Closed-Form Sampling at Arbitrary t

```python
import torch
import matplotlib.pyplot as plt

torch.manual_seed(42)

# Setup
T = 1000
t_values = [0, 250, 500, 750, 999]  # 관찰할 시간 포인트

beta = torch.linspace(0.0001, 0.02, T)
alpha = 1 - beta
alpha_bar = torch.cumprod(alpha, dim=0)

# 초기 신호: 간단한 값 x0 = [2.0]
x0 = torch.tensor([2.0])

# 각 t 에서 sampling
samples = []
for t in t_values:
    x_t = torch.sqrt(alpha_bar[t]) * x0 + torch.sqrt(1 - alpha_bar[t]) * torch.randn(1)
    samples.append(x_t.item())

# 시각화
print("Closed-form sampling at different timesteps:")
print(f"x0 = {x0.item()}")
for t, sample in zip(t_values, samples):
    alpha_bar_t = alpha_bar[t]
    signal_ratio = alpha_bar_t.item()
    noise_ratio = (1 - alpha_bar_t).item()
    print(f"t={t:4d}: x_t={sample:7.4f} (signal: {signal_ratio:.1%}, noise: {noise_ratio:.1%})")
```

### 실험 3: Reparameterization Equivalence

```python
# Reparameterization: 
# x_t = sqrt(alpha_bar_t) x0 + sqrt(1 - alpha_bar_t) epsilon
# 는
# x_t = sqrt(alpha_bar_t) x0 + sqrt(1 - alpha_bar_t) epsilon
# 와 동일 (당연하지만, 코드로 확인)

import torch

torch.manual_seed(42)

T = 100
beta = torch.linspace(0.0001, 0.02, T)
alpha = 1 - beta
alpha_bar = torch.cumprod(alpha, dim=0)

x0 = torch.ones(1)
t = 50

# 여러 번 sampling (같은 x0, 같은 t, 다른 epsilon)
n_samples = 100000
samples = []

for _ in range(n_samples):
    epsilon = torch.randn(1)
    x_t = torch.sqrt(alpha_bar[t]) * x0 + torch.sqrt(1 - alpha_bar[t]) * epsilon
    samples.append(x_t.item())

samples = torch.tensor(samples)

# 통계
empirical_mean = samples.mean()
empirical_var = samples.var()

theoretical_mean = torch.sqrt(alpha_bar[t]) * x0[0]
theoretical_var = 1 - alpha_bar[t]

print(f"t={t}:")
print(f"  Empirical: mean={empirical_mean:.6f}, var={empirical_var:.6f}")
print(f"  Theoretical: mean={theoretical_mean:.6f}, var={theoretical_var:.6f}")
print(f"  Match: {torch.allclose(empirical_mean, theoretical_mean, atol=0.01) and torch.allclose(empirical_var, theoretical_var, atol=0.01)}")
```

## 🔗 실전 활용

**DDPM Loss Function**:
```python
def ddpm_loss(model, x0, t_batch, noise):
    # x0: 진짜 이미지
    # t_batch: 랜덤 시간 (1 to T)
    # noise: 랜덤 노이즈
    
    # Closed-form 으로 x_t 생성
    alpha_bar_t = get_alpha_bar(t_batch)
    x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise
    
    # 신경망으로 노이즈 예측
    noise_pred = model(x_t, t_batch)
    
    # Loss: MSE
    loss = (noise - noise_pred).pow(2).mean()
    return loss
```

**DDIM (Denoising Diffusion Implicit Models)**:
- Closed-form 을 이용해 큰 스텝 크기로 점프 가능
- $t$ 에서 바로 $t-k$ 로 갈 수 있음 (reverse 수정)

**Latent Diffusion Models (Stable Diffusion)**:
- Autoencoder VAE 로 이미지 → latent vector
- Latent space 에서 diffusion 적용 (closed-form 동일)
- 더 빠른 학습

## ⚖️ 가정과 한계

| 항목 | 설명 | 주의사항 |
|------|------|---------|
| **Gaussian 노이즈** | 누적된 노이즈도 Gaussian | 정규분포 가정; 일반화 어려움 |
| **Additivity** | 노이즈를 단순 더하기 | 곱하기, 비선형 변환 불가 |
| **Independent $\epsilon$** | 각 노이즈 단계가 독립 | Correlate noise 는 분석 어려움 |
| **Fixed schedule** | $\beta$ 를 미리 정함 | 적응형 $\beta$ 는 closed-form 파괴 |

## 📌 핵심 정리

$$\boxed{q(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t)I)}$$

$$\boxed{x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0,I)}$$

| 표기 | 정의 | 의미 |
|------|------|------|
| $\alpha_t$ | $1 - \beta_t$ | 신호 보존 비율 (단일 스텝) |
| $\bar{\alpha}_t$ | $\prod_{s=1}^t \alpha_s$ | 누적 신호 보존 비율 |
| $1 - \bar{\alpha}_t$ | 누적 노이즈 비율 | 노이즈의 상대적 강도 |
| Reparameterization | $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$ | 효율적 샘플링 |

## 🤔 생각해볼 문제

### 문제 1 (기초): $\alpha_t, \bar{\alpha}_t$ 계산

$T=3, \beta=[0.001, 0.002, 0.003]$ 일 때,
$\alpha_t$ 와 $\bar{\alpha}_t$ 를 계산하시오.

그리고 $x_3$ 를 $x_0$ 와 $\epsilon$ (단일 표준정규분포) 로 표현하시오.

<details>
<summary>해설</summary>

$\alpha = [0.999, 0.998, 0.997]$

$\bar{\alpha} = [0.999, 0.999 \times 0.998, 0.999 \times 0.998 \times 0.997]$
$= [0.999, 0.997002, 0.994008994]$

따라서:
$$x_3 = \sqrt{0.994008994} \cdot x_0 + \sqrt{1 - 0.994008994} \cdot \epsilon$$
$$= \sqrt{0.994008994} \cdot x_0 + \sqrt{0.005991006} \cdot \epsilon$$
$$\approx 0.997 \cdot x_0 + 0.0774 \cdot \epsilon$$

</details>

### 문제 2 (심화): Gaussian 의 합 (분산 계산)

두 독립 Gaussian 을 더할 때, 분산이 어떻게 더해지는가?

$Y = aX_1 + bX_2, X_1 \sim \mathcal{N}(0, \sigma_1^2), X_2 \sim \mathcal{N}(0, \sigma_2^2)$ (독립) 일 때,
$\text{Var}(Y)$ 를 유도하시오.

<details>
<summary>해설</summary>

$$\text{Var}(Y) = \text{Var}(aX_1 + bX_2)$$
$$= a^2 \text{Var}(X_1) + b^2 \text{Var}(X_2)$$
(independence 로 교차항 0)
$$= a^2 \sigma_1^2 + b^2 \sigma_2^2$$

따라서 표준편차는 $\sqrt{a^2\sigma_1^2 + b^2\sigma_2^2}$.

증명 3.1 에서 우리는 정확히 이를 사용했고,
$a^2\sigma_1^2 + b^2\sigma_2^2 = 1 - \bar{\alpha}_t$ 임을 보였다.

</details>

### 문제 3 (논문 비평): Closed-Form 의 DDPM 에서의 중요성

Ho et al. (2020) DDPM 논문에서 closed-form 이 없었다면,
학습 시간과 실용성이 어떻게 달라졌을까?

또한, score-based diffusion (Song et al. 2021) 에서는
continuous time 이므로 closed-form 이 다르다. 그것을 추론해보시오.

<details>
<summary>해설</summary>

**Closed-form 없다면**:
- 각 배치마다 forward pass 시 Markov chain 을 $T$ 번 반복 → $O(T)$ overhead
- 학습 시간 10-100배 증가
- 메모리도 누적 증가
- DDPM 의 실용성 급격히 떨어짐

**Continuous time (Song et al. 2021)**:
- SDE: $dx = f(t)x dt + g(t) dW$
- Closed-form: $x_t = x(t)$ 를 ODE/SDE solver 로 계산
- Exact closed-form 은 없지만, numerical integration 으로 대체
- Probability flow ODE 로 deterministic reverse 가능

결론: Closed-form 은 DDPM 의 실용화에 필수적!

</details>

---

<div align="center">

[◀ 이전](./02-forward-markov-chain.md) | [📚 README](../README.md) | [다음 ▶](./04-reverse-process.md)

</div>
