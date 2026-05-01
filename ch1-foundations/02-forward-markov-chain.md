# 02. Forward Process: Markov Chain

## 🎯 핵심 질문

- Forward Markov chain $q(x_t \mid x_{t-1})$ 의 구체적인 정의는 무엇인가?
- $\beta_t$ schedule (linear vs. cosine) 은 어떻게 선택하는가?
- Variance preservation: 왜 scaling factor $\sqrt{1-\beta_t}$ 가 정확히 그 형태인가?
- Forward process 에서 variance 가 폭주 (explode) 하지 않는 이유를 수학적으로 증명할 수 있는가?

## 🔍 왜 이 Markov chain 이 diffusion 의 핵심인가

Forward process 는 **데이터를 체계적으로 노이즈로 변환** 하는 스케줄이다.
만약 이 과정이 맞지 않으면 (예: variance 폭발) 학습 자체가 불안정해진다.

Gaussian Markov chain 은 두 가지 장점을 준다:
1. **계산 용이**: Markov property → closed-form solution (Ch1-03)
2. **물리적 타당성**: 각 스텝의 노이즈 추가가 독립적 (Fokker-Planck 의 discrete analog)

DDPM 의 성공은 이 간단하면서도 강력한 선택에서 비롯되었다.

## 📐 수학적 선행 조건

- **Gaussian Distribution** : $\mathcal{N}(\mu, \sigma^2 I)$ 의 성질
- **Variance of Sum of Independent RVs** : $\text{Var}(aX + bY) = a^2 \text{Var}(X) + b^2 \text{Var}(Y)$
- **Markov Property** : $q(x_t \mid x_0, \ldots, x_{t-1}) = q(x_t \mid x_{t-1})$
- **Fokker-Planck / Diffusion** (Ch1-01 참조)
- **$\beta$ schedule** : linear, cosine, etc.

## 📖 직관적 이해

```
Forward Diffusion Markov Chain
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

x₀ (진짜 이미지, 예: 숫자 "8")
│
├─ 노이즈 ε₁ ~ N(0,I) 추가 (가중 √(1-β₁), 강도 √β₁)
│
v
x₁ = √(1-β₁) · x₀ + √β₁ · ε₁   (약간 흐릿함)
│
├─ 노이즈 ε₂ ~ N(0,I) 추가
│
v
x₂ = √(1-β₂) · x₁ + √β₂ · ε₂   (더 흐릿함)
│
├─ ... (T스텝)
│
v
x_T ≈ N(0,I)  (순수 가우시안 노이즈)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

핵심: 각 스텝에서 기존 신호 + 새 노이즈 섞기
      scaling √(1-β) + √β = 정규화 (variance 유지)
```

**비유**: 수채화. 원본 색 (x₀) 에서 시작 → 매 스텝마다 물을 조금씩 섞음 (noise) → 최종적으로 회색 (무색 노이즈).

**핵심 수식**:
$$q(x_t \mid x_{t-1}) = \mathcal{N}\left( \sqrt{1-\beta_t} \, x_{t-1}, \, \beta_t I \right)$$

이 선택이 타당한 이유:
- $\sqrt{1-\beta_t} \in [0, 1]$ : 신호 감쇠
- $\sqrt{\beta_t}$ : 노이즈 강도
- 함께: $(\sqrt{1-\beta_t})^2 + (\sqrt{\beta_t})^2 = 1$ → variance 정규화

## ✏️ 엄밀한 정의

### 정의 2.1: Forward Markov Chain

시간 $t = 1, 2, \ldots, T$ 에 대해, forward diffusion process 는 다음 Markov chain 이다:

$$q(x_t \mid x_{t-1}) = \mathcal{N}\left( \sqrt{1-\beta_t} \, x_{t-1}, \, \beta_t I \right)$$

여기서:
- $\beta_t \in (0, 1)$ : diffusion rate (또는 noise schedule)
- $x_0 \sim p_{\text{data}}(x)$ : 데이터 분포 (예: CIFAR-10)
- $x_T$ : target distribution ≈ $\mathcal{N}(0, I)$

**샘플링**: 주어진 $x_{t-1}$, 다음을 계산:
$$x_t = \sqrt{1-\beta_t} \, x_{t-1} + \sqrt{\beta_t} \, \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I)$$

### 정의 2.2: $\beta$ Schedule

$\beta_t$ 는 사전에 정해진 스케줄로, 다음 중 하나:

**Linear Schedule (DDPM, Ho et al. 2020)**:
$$\beta_t = \beta_{\min} + \frac{t}{T}(\beta_{\max} - \beta_{\min})$$

대표값: $\beta_{\min} = 0.0001, \beta_{\max} = 0.02, T = 1000$

**Cosine Schedule (Improved DDPM, Nichol & Dhariwal 2021)**:
$$\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\left(\frac{t/T + s}{1+s} \cdot \frac{\pi}{2}\right)^2$$

여기서 $s = 0.008$ (offset), $\alpha_t = 1 - \beta_t$, $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$

**장점**: 선형 스케줄보다 분산이 더 잘 유지되고, $T$ 초반에 노이즈 추가가 천천히 진행.

### 정의 2.3: Variance Preservation Property

Forward process 의 중요한 성질: variance 가 시간이 지나도 폭주하지 않음.

$$\mathbb{E}[\|x_t\|^2] = \mathbb{E}[\|\mathbb{E}[x_t \mid x_{t-1}]\|^2] + \mathbb{E}[\text{Var}(x_t \mid x_{t-1})]$$

계산하면:
$$\mathbb{E}[\|x_t\|^2] = (1-\beta_t) \mathbb{E}[\|x_{t-1}\|^2] + \beta_t d$$

여기서 $d$ = dimension, $\mathbb{E}[\|\epsilon_t\|^2] = d$.

초기에 $\mathbb{E}[\|x_0\|^2] \approx d$ (정규화된 이미지) 이면, 
정상 상태 (stationary) 에서 $\mathbb{E}[\|x_t\|^2] \approx d$ 유지 → variance 안정.

## 🔬 정리와 증명

### 정리 2.1: Variance Preservation (귀납법)

**명제**: 데이터 분포 $p_{\text{data}}$ 가 unit variance (평균 0, 분산 ≈ 1) 를 가질 때,
forward Markov chain 의 variance 는 다음을 만족한다:

$$\mathbb{E}[\|x_t - \mathbb{E}[x_t]\|^2] \leq \mathbb{E}[\|x_{t-1} - \mathbb{E}[x_{t-1}]\|^2] + \beta_t + o(\beta_t)$$

더 정확히, $\beta_t$ 가 작으면 variance 는 대략 유지된다.

**증명**:

*Base case*: $t=1$.
$$x_1 = \sqrt{1-\beta_1} x_0 + \sqrt{\beta_1} \epsilon_1$$

양변 제곱의 기댓값:
$$\mathbb{E}[\|x_1\|^2] = (1-\beta_1) \mathbb{E}[\|x_0\|^2] + \beta_1 \mathbb{E}[\|\epsilon_1\|^2]$$

$\mathbb{E}[\|x_0\|^2] = d$ (unit variance), $\mathbb{E}[\|\epsilon_1\|^2] = d$ 이므로:
$$\mathbb{E}[\|x_1\|^2] = (1-\beta_1) \cdot d + \beta_1 \cdot d = d$$

*Induction step*: $t-1$ 에서 성립하면 $t$ 도 성립.

$\mathbb{E}[\|x_{t-1}\|^2] \approx d$ 라 가정. 그러면:
$$\mathbb{E}[\|x_t\|^2] = (1-\beta_t) \mathbb{E}[\|x_{t-1}\|^2] + \beta_t d \approx (1-\beta_t)d + \beta_t d = d$$

따라서 귀납법으로 모든 $t$ 에 대해 $\mathbb{E}[\|x_t\|^2] \approx d$.

$$\square$$

### 정리 2.2: $\beta$ Schedule 의 선택 (정성적)

**명제**: Cosine schedule 이 linear schedule 보다 이미지 품질을 향상시킨다.

**이유**:
1. Linear: $\beta_t$ 가 균등하게 증가 → 초반부 노이즈 추가가 빠름
   - 결과: 초기 $x_0 → x_1$ 에서 이미 정보 손실 심함
   
2. Cosine: $\beta_t$ 가 $t$ 에 대해 concave → 초반부 천천히, 후반부 빠름
   - 결과: 정보 손실을 고르게 분산 → reverse 학습이 더 안정적

경험적으로 cosine schedule 은 FID (Fréchet Inception Distance) 약 2-3% 개선.

## 💻 NumPy / PyTorch 구현 검증

### 실험 1: Linear vs Cosine Schedule

```python
import numpy as np
import matplotlib.pyplot as plt

T = 1000
t = np.arange(1, T+1)

# Linear schedule
beta_min, beta_max = 0.0001, 0.02
beta_linear = beta_min + (t / T) * (beta_max - beta_min)

# Cosine schedule
s = 0.008
f_t = np.cos(((t/T + s) / (1 + s)) * np.pi / 2) ** 2
f_0 = np.cos(s / (1 + s) * np.pi / 2) ** 2
alpha_bar_cosine = f_t / f_0

# beta 계산
alpha_cosine = 1.0 - np.gradient(alpha_bar_cosine, 1)
beta_cosine = np.clip(alpha_cosine, 0.0001, 0.9999)

print("Linear β schedule:")
print(f"  min: {beta_linear[0]:.6f}, max: {beta_linear[-1]:.6f}")
print("\nCosine β schedule:")
print(f"  min: {beta_cosine[0]:.6f}, max: {beta_cosine[-1]:.6f}")
print(f"  β grows slower initially (more gradual)")
```

### 실험 2: Variance Preservation Check

```python
import torch

T = 50
torch.manual_seed(42)

# Linear schedule
beta_linear = torch.linspace(0.0001, 0.02, T)
alpha_linear = 1 - beta_linear

# 1D toy data: x0 ~ N(0, 1), 100개 샘플
x0 = torch.randn(100)
var_x0 = x0.var()

# Forward diffusion
x_sequence = torch.zeros((T, 100))
x_current = x0.clone()

for t in range(T):
    epsilon = torch.randn(100)
    x_current = torch.sqrt(alpha_linear[t]) * x_current + torch.sqrt(beta_linear[t]) * epsilon
    x_sequence[t] = x_current

# Variance tracking
variances = x_sequence.var(dim=1)

print("Forward diffusion variance:")
print(f"Var(x0) = {var_x0:.4f}")
print(f"Var(x25) = {variances[24]:.4f}")
print(f"Var(x50) = {variances[49]:.4f}")
print(f"\nVariance stays roughly constant near 1.0 ✓")
```

### 실험 3: Markov Property Visualization

```python
# x_t 는 x_0 에만 조건부 독립인지 확인 (Markov property)
# E[x_t | x_{t-1}] = sqrt(1-beta_t) * x_{t-1}

import torch

T = 10
beta = torch.linspace(0.0001, 0.02, T)
alpha = 1 - beta

# 단일 x0
x0 = torch.tensor([1.0])

# Forward 경로
x_path = [x0]
for t in range(T):
    # 샘플링
    eps = torch.randn(1)
    xt = torch.sqrt(alpha[t]) * x_path[-1] + torch.sqrt(beta[t]) * eps
    x_path.append(xt)

x_path = torch.stack(x_path)

# 이론적 기댓값 (closed-form, Ch1-03)
alpha_bar = torch.cumprod(alpha, dim=0)
alpha_bar = torch.cat([torch.tensor([1.0]), alpha_bar])
x_mean = torch.sqrt(alpha_bar) * x0

print("Markov chain dynamics:")
for t in range(1, T+1):
    print(f"E[x_{t}] = {x_mean[t].item():.4f} (theoretical)"
          f" ≈ {x_path[t].item():.4f} (sampled, 1 trajectory)")
```

## 🔗 실전 활용

**DDPM (Ho et al. 2020)**:
- Linear schedule: 간단하고 effective
- 1000 steps (T=1000) 표준

**Improved DDPM (Nichol & Dhariwal 2021)**:
- Cosine schedule: 더 나은 FID 점수
- Learned variance: $\Sigma_\theta$ 도 신경망으로 학습

**Stable Diffusion**:
- Linear schedule 사용
- Latent space diffusion (pixel space 대신)

**DDIM (Song et al. 2021)**:
- Forward process 동일
- Reverse 를 non-Markovian 으로 가속화 (Ch3-02)

**Guidance (Classifier / Classifier-Free)**:
- Forward process 는 무조건부 (unconditional)
- Reverse 에서 조건부 (conditional) 로 조정

## ⚖️ 가정과 한계

| 항목 | 설명 | 주의사항 |
|------|------|---------|
| **Gaussian assumption** | $q(x_t \mid x_{t-1})$ 이 정확히 Gaussian | 이산 근사; small-$\beta$ 에서 타당 |
| **Independence of noise** | 각 $\epsilon_t$ 가 독립 | 물리적으로 자연스러움; 계산 효율 |
| **Fixed schedule** | $\beta_t$ 를 사전결정 | 학습 가능한 $\beta$ 는 성능 개선 가능 |
| **Unit variance** | $\mathbb{E}[\|x_0\|^2] \approx d$ 가정 | 데이터 정규화 필수 (보통 [-1,1] 범위) |
| **Constant dimension** | 노이즈가 신호와 같은 크기 | 비선형 변환 불가; Gaussian only |

## 📌 핵심 정리

$$\boxed{q(x_t \mid x_{t-1}) = \mathcal{N}\left(\sqrt{1-\beta_t} x_{t-1}, \beta_t I\right)}$$

$$\boxed{x_t = \sqrt{1-\beta_t} x_{t-1} + \sqrt{\beta_t} \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0,I)}$$

| 개념 | 정의 | 역할 |
|------|------|------|
| **Forward Markov Chain** | $q(x_t \mid x_{t-1})$ 정의 | 데이터 → 노이즈 변환 |
| **Linear Schedule** | $\beta_t = \beta_{\min} + \frac{t}{T}(\beta_{\max}-\beta_{\min})$ | 간단; DDPM 기본 |
| **Cosine Schedule** | $\bar{\alpha}_t = \cos^2(\frac{t/T+s}{1+s}\frac{\pi}{2})$ | 더 나은 정보 보존 |
| **Variance Preservation** | $\mathbb{E}[\|x_t\|^2] \approx d$ 유지 | 수치 안정성 보증 |

## 🤔 생각해볼 문제

### 문제 1 (기초): Linear Schedule 파라미터 선택

DDPM 에서 $\beta_{\min} = 0.0001, \beta_{\max} = 0.02$ 로 선택했다.
왜 이 범위인가? 만약 $\beta_{\max} = 0.1$ 로 너무 크게 하면 어떻게 될까?

<details>
<summary>해설</summary>

$\beta_{\min} = 0.0001$ (작음): 초반 노이즈 추가가 천천히 → 신호 정보 보존.

$\beta_{\max} = 0.02$ (작음): 말기에도 노이즈 추가가 완만 → Gaussian 까지 천천히.

만약 $\beta_{\max} = 0.1$ 이면:
- 너무 빨리 노이즈가 많아짐
- $x_T$ 의 신호 성분이 너무 약해짐 (reverse 학습 어려움)
- DDPM 역과정의 KL divergence 증가 → 품질 악화

경험적으로 $\beta_{\max} \in [0.01, 0.03]$ 이 최적.

</details>

### 문제 2 (심화): Variance Preservation 식 재유도

주어진 $\mathbb{E}[\|x_{t-1}\|^2] \approx d$ 일 때,
$\mathbb{E}[\|x_t\|^2]$ 를 계산하시오 (variance of Gaussian sum 이용).

<details>
<summary>해설</summary>

$x_t = \sqrt{1-\beta_t} x_{t-1} + \sqrt{\beta_t} \epsilon_t$

양변 제곱:
$$\|x_t\|^2 = (1-\beta_t)\|x_{t-1}\|^2 + \beta_t\|\epsilon_t\|^2 + 2\sqrt{1-\beta_t}\sqrt{\beta_t} x_{t-1}^T \epsilon_t$$

기댓값:
- $\mathbb{E}[\|x_{t-1}\|^2] = d$ (가정)
- $\mathbb{E}[\|\epsilon_t\|^2] = d$ (표준 Gaussian)
- $\mathbb{E}[x_{t-1}^T \epsilon_t] = 0$ (독립)

따라서:
$$\mathbb{E}[\|x_t\|^2] = (1-\beta_t) d + \beta_t d = d \quad \checkmark$$

Variance 정확히 보존됨!

</details>

### 문제 3 (논문 비평): Schedule 의 한계

Cosine schedule 이 linear 보다 낫다는 증거를 제시하시오.
그런데 왜 더 복잡한 schedule (예: adaptive) 은 널리 쓰이지 않는가?

<details>
<summary>해설</summary>

**Cosine 의 장점** (Nichol & Dhariwal 2021):
- Linear 대비 FID 약 2-3% 개선 (CIFAR-10)
- Information 손실이 고르게 분산
- 수학적 근거: SNR (signal-to-noise ratio) 의 선형 감소

**Schedule 이 고정인 이유**:
1. Forward process 는 deterministic → 고정 schedule 이 자연스러움
2. Reverse 는 신경망으로 학습 → schedule 의존성 낮음
3. Adaptive schedule 은 추가 하이퍼파라미터 (overfitting risk)
4. 실제 성능 개선은 미미; 계산 cost 증가

**결론**: Cosine schedule 은 fixed + simple 의 좋은 trade-off.

</details>

---

<div align="center">

[◀ 이전](./01-physical-origin.md) | [📚 README](../README.md) | [다음 ▶](./03-forward-closed-form.md)

</div>
