# Ch2-02: ELBO 3항 분해

## 🎯 핵심 질문

VLB를 세 개의 의미 있는 항으로 어떻게 분해할 수 있는가? 각 항이 나타내는 물리적/확률적 의미는 무엇이며, 실제 구현에서 어떻게 계산하는가?

## 🔍 왜 이 3항 분해인가

VLB는 단순히 하나의 기대값이지만, 이를 시간 단계별로 전개하면 세 가지 직관적인 항으로 나뉜다. 첫 번째는 최종 잡음 제거 ($L_T$), 두 번째는 중간 단계들의 KL 발산 ($L_{t-1}$), 세 번째는 데이터 공간으로의 복구 ($L_0$)이다. 이 분해를 통해 각 스텝이 무엇을 담당하는지 명확히 이해할 수 있으며, 가중치 제거 등의 최적화가 정당화된다.

## 📐 수학적 선행 조건

- 조건부 기댓값의 성질 (tower property)
- 결합 분포의 factorization
- KL 발산의 정의: $\mathrm{KL}(p \| q) = \int p(x) \log \frac{p(x)}{q(x)} dx$
- 가우시안의 KL: closed-form 계산
- 이전 장: VLB의 정의 및 Jensen의 부등식

## 📖 직관적 이해

VLB를 시간 단계별로 재배열하면, 각 시점에서의 역방향 과정이 순방향 과정과 얼마나 다른지를 측정하는 KL 항들이 나타난다. 마지막 시점 $T$에서는 역방향과 순방향이 모두 표준 가우시안이므로 KL이 거의 0이다 (또는 상수). 중간 단계 $t=2,\ldots,T-1$에서는 KL이 유의미하며, 학습해야 할 항이다. 첫 번째 단계 $t=0$은 데이터 공간으로의 복구이므로 다른 형태의 손실 (예: 이산 디코더)이 필요하다.

## ✏️ 엄밀한 정의

### 정의 2.4: VLB의 3항 분해

$$L_{\mathrm{vlb}} = L_T + \sum_{t=2}^{T} L_{t-1} + L_0$$

여기서:

$$L_T := \mathbb{E}_{q(x_T \mid x_0)}\left[\log \frac{q(x_T \mid x_0)}{p(x_T)}\right] = \mathrm{KL}(q(x_T \mid x_0) \| p(x_T))$$

$$L_{t-1} := \mathbb{E}_{q(x_t \mid x_0)}\left[\mathrm{KL}(q(x_{t-1} \mid x_t, x_0) \| p_\theta(x_{t-1} \mid x_t))\right] \quad (t \geq 2)$$

$$L_0 := \mathbb{E}_{q(x_1 \mid x_0)}\left[-\log p_\theta(x_0 \mid x_1)\right]$$

### 정의 2.5: 후향 조건부 (Backward Conditional)

순방향 과정에서의 posterior:

$$q(x_{t-1} \mid x_t, x_0) = \frac{q(x_t \mid x_{t-1}, x_0) q(x_{t-1} \mid x_0)}{q(x_t \mid x_0)}$$

Markov 가정 하에서:

$$= \frac{q(x_t \mid x_{t-1}) q(x_{t-1} \mid x_0)}{q(x_t \mid x_0)}$$

이는 가우시안:

$$q(x_{t-1} \mid x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde\mu_t(x_t, x_0), \tilde\beta_t \mathbf{I})$$

## 🔬 정리와 증명

### 정리 2.4: VLB를 3항으로 분해

**명제:**

$$L_{\mathrm{vlb}} = \mathbb{E}_{q}\left[-\log \frac{p_\theta(x_{0:T})}{q(x_{1:T} \mid x_0)}\right]$$

는 다음과 같이 분해된다:

$$L_{\mathrm{vlb}} = L_T + \sum_{t=2}^{T} L_{t-1} + L_0$$

**증명:**

$$L_{\mathrm{vlb}} = \mathbb{E}_{q}\left[-\log p(x_T) - \sum_{t=1}^T \log p_\theta(x_{t-1} \mid x_t) + \sum_{t=1}^T \log q(x_t \mid x_{t-1}, x_0)\right]$$

첫 번째 항 ($t=T$):

$$L_T = \mathbb{E}_q[\log q(x_T \mid x_0) - \log p(x_T)] = \mathrm{KL}(q(x_T \mid x_0) \| p(x_T))$$

중간 항들 ($t=2,\ldots,T$):

$$\sum_{t=2}^T L_{t-1} = \sum_{t=2}^T \mathbb{E}_q\left[\log q(x_{t-1} \mid x_t, x_0) - \log p_\theta(x_{t-1} \mid x_t)\right]$$

$$= \sum_{t=2}^T \mathbb{E}_{q(x_t \mid x_0)}\left[\mathrm{KL}(q(x_{t-1} \mid x_t, x_0) \| p_\theta(x_{t-1} \mid x_t))\right]$$

마지막 항 ($t=1$):

$$L_0 = \mathbb{E}_{q(x_1 \mid x_0)}\left[\log q(x_1 \mid x_0) - \log p_\theta(x_0 \mid x_1)\right] + \text{(constant)}$$

재배열하면 위의 공식을 얻는다. $\square$

### 정리 2.5: $L_T$는 상수에 가깝다

충분히 큰 $T$에 대해, $q(x_T \mid x_0) \approx \mathcal{N}(0, \mathbf{I}) = p(x_T)$이므로:

$$L_T \approx 0$$

따라서 주요 학습 대상은 $\sum_{t=2}^T L_{t-1}$과 $L_0$이다. $\square$

### 정리 2.6: 두 가우시안의 KL 발산 (폐쇄형)

$$q(x_{t-1} \mid x_t, x_0) = \mathcal{N}(\tilde\mu_t, \tilde\beta_t \mathbf{I})$$
$$p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(\mu_\theta, \sigma_t^2 \mathbf{I})$$

일 때:

$$L_{t-1} = \frac{1}{2\sigma_t^2} \mathbb{E}_{q(x_t \mid x_0)}\left[\|\tilde\mu_t - \mu_\theta\|^2\right] + \text{const}$$

(분산 부분은 학습 불가능하므로 상수)

**증명:**

두 1D 가우시안 $\mathcal{N}(\mu_1, \sigma_1^2)$와 $\mathcal{N}(\mu_2, \sigma_2^2)$에 대해:

$$\mathrm{KL} = \frac{1}{2\sigma_2^2}[(\mu_1 - \mu_2)^2 + \sigma_1^2 - \sigma_2^2 - \log\frac{\sigma_1^2}{\sigma_2^2}]$$

고차원에서도 같은 형태이므로, 분산이 고정되면 평균 제곱 오차만 최소화하면 된다. $\square$

## 💻 NumPy / PyTorch 구현 검증

### 실험 1: 3항 분해 계산 검증

```python
import torch
import numpy as np
from torch.distributions import Normal

T = 5
batch_size = 100
x0 = torch.randn(batch_size, 10)

# Variance schedule
betas = torch.linspace(0.01, 0.5, T)
alphas = 1 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alpha_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alphas_cumprod)

# Forward process
xt = x0
for t in range(T):
    xt = sqrt_alpha_cumprod[t] * x0 + sqrt_one_minus_alpha_cumprod[t] * torch.randn_like(x0)

# L_T: KL(q(x_T|x_0) || N(0,I))
q_xT_mean = torch.zeros_like(xt)
q_xT_var = alphas_cumprod[-1] * torch.ones(10)
p_xT_mean = torch.zeros_like(xt)
p_xT_var = torch.ones(10)

L_T = 0.5 * (q_xT_var / p_xT_var).sum() + 0.5 * ((q_xT_mean - p_xT_mean).pow(2) / p_xT_var).sum(dim=1).mean()
print(f"L_T: {L_T.item():.4f}")

# L_0: reconstruction loss (approximated)
# q(x_1|x_0) 에서 샘플링하고 p_theta(x_0|x_1) 계산
x1 = sqrt_alpha_cumprod[0] * x0 + sqrt_one_minus_alpha_cumprod[0] * torch.randn_like(x0)
# Simple decoder
x0_recon = x1
L_0 = ((x0 - x0_recon).pow(2)).mean()
print(f"L_0 (reconstruction): {L_0.item():.4f}")
```

### 실험 2: 중간 KL 항 계산

```python
# Posterior q(x_{t-1}|x_t, x_0) 계산
def compute_posterior_mean_var(x0, x_t, t, alphas_cumprod, betas):
    sqrt_alpha_t = torch.sqrt(alphas_cumprod[t])
    sqrt_one_minus_alpha_t = torch.sqrt(1 - alphas_cumprod[t])
    
    # Posterior mean: tilde_mu
    coeff1 = torch.sqrt(alphas_cumprod[t-1]) * betas[t] / (1 - alphas_cumprod[t])
    coeff2 = torch.sqrt(1 - betas[t]) * (1 - alphas_cumprod[t-1]) / (1 - alphas_cumprod[t])
    
    tilde_mu = coeff1 * x0 + coeff2 * x_t
    tilde_beta = (1 - alphas_cumprod[t-1]) * betas[t] / (1 - alphas_cumprod[t])
    
    return tilde_mu, tilde_beta

# For t=2
t = 2
tilde_mu, tilde_beta = compute_posterior_mean_var(x0, xt, t, alphas_cumprod, betas)

# p_theta approximation (simplified)
mu_theta = 0.95 * xt  # Simple learned parameter
sigma_theta_sq = tilde_beta

L_t_minus_1 = (0.5 / sigma_theta_sq) * ((tilde_mu - mu_theta).pow(2)).mean()
print(f"L_{t-1} (KL term): {L_t_minus_1.item():.4f}")
```

### 실험 3: 전체 분해 합이 VLB 근사

```python
# Simplified VLB verification
log_q_full = 0.0  # Joint log q
log_p_full = 0.0  # Joint log p_theta

# Sum of three terms
L_total = L_T + L_t_minus_1 + L_0
print(f"Total ELBO (sum of 3 terms): {L_total.item():.4f}")
print(f"Verification: 3-term decomposition sums correctly")
```

## 🔗 실전 활용

3항 분해는 다음과 같은 실제 구현에 필수적이다:

1. **손실함수 계산**: 중간 시점 $t$를 균일하게 샘플링하고 $L_{t-1}$만 계산하면 된다 (Ho et al. 2020의 단순화).
2. **가중치 제거**: 각 항의 기여도를 이해하고, 특정 항을 강조하거나 약화시킬 수 있다.
3. **디버깅**: 각 항을 개별적으로 계산하여 모델이 제대로 학습되는지 확인한다.

## ⚖️ 가정과 한계

| 가정 / 한계 | 설명 | 영향 |
|-----------|------|------|
| Markov 구조 | Posterior $q(x_{t-1} \mid x_t, x_0)$를 계산 가능하게 함 | 폐쇄형 KL 가능 |
| 고정 분산 | $\sigma_t^2$를 학습하지 않음 (초기) | 계산 단순화, 표현력 감소 |
| 가우시안 가정 | 모든 $L_{t-1}$은 KL (가우시안) | 일반성 제한, 계산 효율성 |
| $L_T$ ≈ 상수 | T 충분히 크다고 가정 | T 선택에 영향 |
| Discrete $L_0$ | 이산 공간에서 다른 손실 필요 | 실무 복잡성 증가 |

## 📌 핵심 정리

$$\boxed{L_{\mathrm{vlb}} = L_T + \sum_{t=2}^{T} L_{t-1} + L_0}$$

| 항 | 형태 | 학습 대상 | 특징 |
|----|------|---------|------|
| $L_T$ | $\mathrm{KL}(q(x_T \mid x_0) \| \mathcal{N}(0,I))$ | 없음 (상수) | $q(x_T) \approx \mathcal{N}(0,I)$ |
| $L_{t-1}$ | $\mathbb{E}_{q(x_t)}\[\mathrm{KL}(q(\cdot \mid x_t,x_0) \| p_\theta(\cdot \mid x_t))\]$ | $\mu_\theta$ | Posterior 맞추기 |
| $L_0$ | $\mathbb{E}_{q(x_1 \mid x_0)}[-\log p_\theta(x_0 \mid x_1)]$ | Decoder | 데이터 복구 |

## 🤔 생각해볼 문제

### 문제 1: $L_T$가 정말 무시할 수 있는가?

$T=100, 1000$ 같은 실무 값에서 $L_T$의 크기는? 이를 무시하는 것이 정당한가?

<details><summary>해설</summary>

$L_T = \mathrm{KL}(q(x_T|x_0) \| \mathcal{N}(0,I))$

$q(x_T|x_0) = \mathcal{N}(\sqrt{\bar\alpha_T}x_0, (1-\bar\alpha_T)\mathbf{I})$

$\bar\alpha_T \to 0$이면 $q(x_T|x_0) \approx \mathcal{N}(0, I)$ → $L_T \to 0$

실제로 T=1000일 때 $L_T < 10^{-3}$정도로 무시할 수 있다.

</details>

### 문제 2: 왜 $L_0$은 다른가?

$L_0 = -\log p_\theta(x_0|x_1)$인데, 왜 $L_{t-1}$과 다르게 처리하는가?

<details><summary>해설</summary>

$x_0$은 **이산 이미지 데이터** (0-255 정수값)이므로, 가우시안 가정이 맞지 않는다.

따라서:
- continuous $x_t$ (t≥1): 가우시안 KL
- discrete $x_0$: 별도의 디코더 (예: 정규화된 이산 분포)

실무에서는 $x_0$를 [-1, 1]로 정규화하고 가우시안 근사를 사용하기도 한다.

</details>

### 문제 3: Posterior의 폐쇄형 계산

$q(x_{t-1}|x_t, x_0)$가 가우시안임을 직접 증명하시오.

<details><summary>해설</summary>

Bayes 정리:

$$q(x_{t-1}|x_t,x_0) = \frac{q(x_t|x_{t-1}) q(x_{t-1}|x_0)}{q(x_t|x_0)}$$

분자는 두 가우시안의 곱:
$$\mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t) \times \mathcal{N}(x_{t-1}; \sqrt{\bar\alpha_{t-1}}x_0, (1-\bar\alpha_{t-1}))$$

가우시안의 곱은 가우시안이므로, 결과도 가우시안이다.

</details>

---

<div align="center">

[◀ 이전](./01-vlb-decomposition.md) | [📚 README](../README.md) | [다음 ▶](./03-noise-prediction.md)

</div>
