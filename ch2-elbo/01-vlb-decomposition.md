# Ch2-01: VLB 분해 및 VAE 일반화

## 🎯 핵심 질문

Diffusion Model의 손실함수를 도출하는 가장 기초적인 방법은 무엇인가? 왜 Variational Lower Bound (VLB)를 통해 원본 데이터의 로그 확률을 하한으로 표현할 수 있으며, 이것이 VAE와 어떻게 연결되는가?

## 🔍 왜 이 VLB 분해인가

생성 모델을 학습하려면 데이터 $x_0$의 로그 확률 $\log p_\theta(x_0)$을 최대화해야 한다. 그러나 이 값을 직접 계산하기는 불가능하다 (적분이 intractable). Diffusion Model은 순방향 과정 (forward process)을 통해 잠재 변수 $x_{1:T}$를 도입하고, Jensen의 부등식으로 계산 가능한 하한을 유도한다. 이 접근법은 VAE를 T=1에서 T>1로 일반화한 것이며, Markov 가정이 핵심이다.

## 📐 수학적 선행 조건

- 조건부 확률, 결합 분포, 주변 분포 (marginalization)
- Jensen의 부등식: $\log \mathbb{E}[X] \geq \mathbb{E}[\log X]$ (concave 함수)
- 결합 분포의 인수분해: $p(x_{0:T}) = p(x_0) \prod_{t=1}^T p(x_t \mid x_{0:t-1})$
- 순방향 과정 (forward diffusion): $q(x_{1:T} \mid x_0)$는 Markov 사슬

## 📖 직관적 이해

우리는 역방향 생성 과정 $p_\theta(x_0 \mid x_1, \ldots, x_T)$를 학습하고 싶지만, 이를 직접 모델링할 수 없다. 대신, 계산 가능한 보조 분포 (proxy distribution) $q(x_{1:T} \mid x_0)$를 도입한다. 이는 가우시안 잡음을 점진적으로 더하는 과정이다. 그 후, 원본 로그 확률을 이 보조 분포를 이용한 하한으로 표현한다. 이 하한이 VLB이며, T=1일 때는 정확히 VAE의 ELBO와 같다.

## ✏️ 엄밀한 정의

### 정의 2.1: 순방향 과정 (Forward Process)

주어진 데이터 $x_0 \sim q(x_0)$에 대해, 순방향 과정은 다음과 같이 정의된다:

$$q(x_{1:T} \mid x_0) = \prod_{t=1}^{T} q(x_t \mid x_{t-1})$$

여기서 각 단계는 가우시안:

$$q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t \mathbf{I})$$

파라미터: $0 < \beta_1 < \beta_2 < \cdots < \beta_T < 1$ (variance schedule)

### 정의 2.2: 역방향 생성 모델 (Reverse Process)

역방향 과정은 $x_T \sim \mathcal{N}(0, \mathbf{I})$에서 시작하여:

$$p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^{T} p_\theta(x_{t-1} \mid x_t)$$

각 역방향 스텝은 학습 가능한 가우시안:

$$p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(t))$$

### 정의 2.3: Variational Lower Bound (VLB)

$$L_{\mathrm{vlb}} := \mathbb{E}_{q(x_{1:T}\mid x_0)}\left[-\log \frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}\right]$$

## 🔬 정리와 증명

### 정리 2.1: VLB가 로그 확률의 하한

$$-\log p_\theta(x_0) \leq L_{\mathrm{vlb}}$$

**증명:**

$$-\log p_\theta(x_0) = -\log \mathbb{E}_{q(x_{1:T}\mid x_0)}\left[\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}\right]$$

Jensen의 부등식을 적용하면 (음수 로그는 concave):

$$\geq -\mathbb{E}_{q(x_{1:T}\mid x_0)}\left[\log \frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}\right]$$

$$= L_{\mathrm{vlb}}$$

$\square$

### 정리 2.2: VAE와의 관계 (T=1 특수 경우)

T=1일 때, VLB는 다음과 같이 단순화된다:

$$L_{\mathrm{vlb}} = \mathbb{E}_{q(x_1\mid x_0)}\left[-\log \frac{p_\theta(x_0, x_1)}{q(x_1\mid x_0)}\right]$$

$$= \mathbb{E}_{q}\left[-\log p_\theta(x_0 \mid x_1) - \log \frac{p(x_1)}{q(x_1 \mid x_0)}\right]$$

$$= \mathbb{E}_{q}\left[-\log p_\theta(x_0 \mid x_1)\right] + \mathrm{KL}(q(x_1\mid x_0) \| p(x_1))$$

이는 정확히 VAE의 ELBO 형태이다 (reconstruction + KL 정규화). $\square$

### 정리 2.3: Markov 가정의 역할

순방향 과정에서 Markov 가정 $q(x_t \mid x_{t-1}, x_{t-2}, \ldots, x_0) = q(x_t \mid x_{t-1})$이 없다면:

- 역방향 과정의 posterior가 intractable해진다
- 계산 가능한 form을 얻을 수 없다
- 각 스텝의 독립성 덕분에 시간 단계별 분해가 가능해진다

따라서 Markov 가정은 VLB 유도의 **필수 조건**이다. $\square$

## 💻 NumPy / PyTorch 구현 검증

### 실험 1: VLB의 하한 성질 검증

```python
import torch
from torch.distributions import Normal

# 1D Gaussian 데이터
x0_dist = Normal(loc=0., scale=1.)
x0_samples = x0_dist.sample((10000,))

# 간단한 순방향: q(x1|x0) = N(sqrt(0.9)*x0, sqrt(0.1))
beta_1 = 0.1
alpha_1 = 1 - beta_1
q_x1_given_x0 = Normal(loc=torch.sqrt(torch.tensor(alpha_1)) * x0_samples, 
                        scale=torch.sqrt(torch.tensor(beta_1)))
x1_samples = q_x1_given_x0.rsample()

# 역방향: p_theta(x0|x1)
mu_theta = 0.95 * x1_samples
sigma_theta = torch.sqrt(torch.tensor(0.15))
p_x0_given_x1 = Normal(loc=mu_theta, scale=sigma_theta)

# ELBO 계산
q_x1_marginal = Normal(loc=0., scale=1.)
kl_div = torch.distributions.kl_divergence(q_x1_given_x0, q_x1_marginal)
reconstruction_loss = -p_x0_given_x1.log_prob(x0_samples)

elbo = (reconstruction_loss + kl_div).mean()
print(f"ELBO (VLB): {elbo.item():.4f}")
```

### 실험 2: 순방향 과정의 Markov 분해

```python
import torch

T = 5
x0 = torch.randn(100, 10)
betas = torch.linspace(0.01, 0.5, T)
alphas = 1 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

# Forward trajectory
x = x0
for t in range(T):
    alpha_t = alphas_cumprod[t]
    beta_t = betas[t]
    x = torch.sqrt(alpha_t) * x + torch.sqrt(beta_t) * torch.randn_like(x)

print(f"x_T norm: {x.norm().item():.4f}")
print(f"x_T mean/std: {x.mean().item():.4f} / {x.std().item():.4f}")
```

### 실험 3: VAE ELBO와의 비교

```python
batch_size = 1000
latent_dim = 8

z_mu = torch.randn(batch_size, latent_dim)
z_logvar = torch.randn(batch_size, latent_dim)
z = z_mu + torch.exp(0.5 * z_logvar) * torch.randn_like(z_logvar)

kl_vae = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp(), dim=1).mean()
reconstruction_vae = ((torch.randn_like(z) - z).pow(2)).mean()

elbo_vae = reconstruction_vae + kl_vae
print(f"VAE ELBO: {elbo_vae.item():.4f}")
```

## 🔗 실전 활용

VLB 분해는 다음과 같은 실무에서 핵심 역할을 한다:

1. **손실함수 유도의 기초**: VLB로부터 세 항의 분해가 도출되며, 이는 실제 구현에 직접 쓰인다.
2. **이론적 검증**: 새로운 parameterization (noise prediction, score prediction)을 제안할 때, VLB 프레임워크로부터 그 합리성을 증명할 수 있다.
3. **하이퍼파라미터 설정**: Variance schedule의 선택이 VLB를 통해 이론적으로 정당화된다.

## ⚖️ 가정과 한계

| 가정 / 한계 | 설명 | 영향 |
|-----------|------|------|
| Markov 구조 | $q(x_t \mid x_{t-1}, \ldots, x_0) = q(x_t \mid x_{t-1})$ | 역방향 계산 가능성 보장 |
| 가우시안 전이 | 모든 스텝이 가우시안 | 해석 용이, 확장 가능 |
| T → ∞ 극한 | $q(x_T \mid x_0) \approx \mathcal{N}(0, \mathbf{I})$ | 초기 분포 제어 |
| 고정 인코더 | $q$는 학습하지 않음 | 안정성 증가 |
| 하한만 제공 | VLB는 상한일 뿐 | Tightness 필요 |

## 📌 핵심 정리

$$\boxed{-\log p_\theta(x_0) \leq L_{\mathrm{vlb}} := \mathbb{E}_{q(x_{1:T}\mid x_0)}\left[-\log \frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}\right]}$$

| 개념 | 수식 | 의미 |
|------|------|------|
| 순방향 | $q(x_{1:T}\mid x_0) = \prod_t q(x_t \mid x_{t-1})$ | 고정된 잡음 과정 |
| 역방향 | $p_\theta(x_{0:T}) = p(x_T) \prod_t p_\theta(x_{t-1} \mid x_t)$ | 학습된 제거 과정 |
| VLB | Jensen + 조건부 기댓값 | 계산 가능한 목적함수 |

## 🤔 생각해볼 문제

### 문제 1: Reverse KL vs Forward KL

VLB에서 나타나는 KL 항은 어떤 형태이며, 이것이 학습에 미치는 영향은?

<details><summary>해설</summary>

VLB에서: $L_{\mathrm{vlb}} = \mathbb{E}_q[\log q - \log p_\theta]$ → Reverse KL $\mathrm{KL}(q \| p_\theta)$

이는 mode-covering이므로, 다중 모드 학습에 유리하다.

</details>

### 문제 2: T → ∞일 때의 극한

T를 무한히 증가시키면 VLB는 어떻게 변하는가?

<details><summary>해설</summary>

$T \to \infty$일 때:
- $\bar\alpha_T \to 0$ → $q(x_T | x_0) \to \mathcal{N}(0, \mathbf{I})$
- $x_T$는 $x_0$와 독립
- 초기 분포를 완벽히 제어 가능

</details>

### 문제 3: 왜 역방향도 가우시안인가?

역방향 분포가 반드시 가우시안이어야 하는 이유는?

<details><summary>해설</summary>

1. $q(x_{t-1} \mid x_t, x_0)$가 가우시안이므로 (Ch1 참조)
2. KL 계산이 closed form
3. 신경망으로 평균/분산만 예측하면 됨

</details>

---

<div align="center">

[◀ 이전](../ch1-foundations/05-posterior-derivation.md) | [📚 README](../README.md) | [다음 ▶](./02-elbo-three-terms.md)

</div>
