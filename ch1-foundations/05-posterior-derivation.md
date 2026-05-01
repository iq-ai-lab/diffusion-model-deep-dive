# 05. Posterior Derivation: $q(x_{t-1} \mid x_t, x_0)$

## 🎯 핵심 질문

- Bayes' rule 을 정확히 어떻게 적용하는가?
- 두 Gaussian 의 곱의 closed-form 은 무엇인가?
- Mean $\tilde{\mu}_t$ 와 variance $\tilde{\beta}_t$ 의 정확한 형태는?
- 왜 ELBO 의 $L_{t-1}$ 이 두 Gaussian 사이의 KL divergence 가 되는가?

## 🔍 왜 이 posterior 가 DDPM 의 수심인가

**DDPM 의 핵심 아이디어**:
- Forward: 데이터 $x_0$ 를 노이즈 $x_T$ 로 변환 (결정적 + 사전정의 schedule)
- Reverse: 신경망 $p_\theta$ 가 true posterior $q(x_{t-1} \mid x_t, x_0)$ 를 모방

True posterior 를 안다는 것은:
1. **학습 target** 명확 (KL divergence 최소화)
2. **Loss 함수** 를 정확히 계산 가능 (Ch2 ELBO)
3. **이론적 보증** : converged 신경망이 데이터를 복원

이 모든 것이 posterior 의 closed-form 에서 나온다.

## 📐 수학적 선행 조건

- **Bayes' Rule** : $p(A|B) = \frac{p(B|A) p(A)}{p(B)}$
- **Gaussian Product** : $\mathcal{N}(a,A) \times \mathcal{N}(b,B) \propto \mathcal{N}(\mu, \Sigma)$
- **Precision (역분산)** : $\Lambda = \Sigma^{-1}$
- **Ch1-02, 03** : Markov chain, closed-form
- **Linear Algebra** : Matrix inversion, symmetry

## 📖 직관적 이해

```
Bayes' Rule 적용:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

q(x_{t-1} | x_t, x_0) 
  = q(x_t | x_{t-1}) * q(x_{t-1} | x_0) / q(x_t | x_0)
  
  분자 요소:
  ├─ q(x_t | x_{t-1}) : "한 스텝 forward" Gaussian
  │  (x_{t-1} 에서 시작, x_t 분포)
  │
  └─ q(x_{t-1} | x_0) : "여러 스텝 누적된" Gaussian
     (x_0 에서 시작, x_{t-1} 분포)
  
  분모 (정규화):
  └─ q(x_t | x_0) : closed-form (Ch1-03)

결과:
  세 Gaussian 의 "조화"
  → 다시 Gaussian (분자/분모의 지수 함수 형태 유지)
  → Closed-form mean/variance!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**핵심**: Gaussian 분포들의 곱/나눗셈도 Gaussian.
이 덕분에 posterior 의 exact form 이 나온다.

## ✏️ 엄밀한 정의

### 정의 5.1: Posterior Distribution

주어진 $x_t$ 와 $x_0$, 이전 상태의 조건부 분포:
$$q(x_{t-1} \mid x_t, x_0) = \mathcal{N}(\tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t I)$$

여기서 $\tilde{\mu}_t$ 와 $\tilde{\beta}_t$ 는 Ch1-02, 03 의 schedule 에서 유래.

### 정의 5.2: Gaussian Conditioning Formula

일반적으로, 결합 분포 $\begin{bmatrix} x_a \\ x_b \end{bmatrix} \sim \mathcal{N}(\mu, \Sigma)$ 에서,
조건부 분포 $x_a \mid x_b$ 는:
$$x_a \mid x_b \sim \mathcal{N}(\mu_{a|b}, \Sigma_{a|b})$$

$$\mu_{a|b} = \mu_a + \Sigma_{ab} \Sigma_{bb}^{-1} (x_b - \mu_b)$$
$$\Sigma_{a|b} = \Sigma_{aa} - \Sigma_{ab} \Sigma_{bb}^{-1} \Sigma_{ba}$$

이를 적용하면 posterior 의 closed-form 을 얻을 수 있다.

## 🔬 정리와 증명

### 정리 5.1: Posterior Mean and Variance

**명제**: Forward Markov chain 에서, posterior 는:
$$q(x_{t-1} \mid x_t, x_0) = \mathcal{N}(\tilde{\mu}_t, \tilde{\beta}_t I)$$

$$\tilde{\mu}_t = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t$$

$$\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t$$

**증명** (Bayes' rule + Gaussian product):

$$q(x_{t-1} \mid x_t, x_0) = \frac{q(x_t \mid x_{t-1}) q(x_{t-1} \mid x_0)}{q(x_t \mid x_0)}$$

각 항을 대입:

**Forward term**:
$$q(x_t \mid x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t} x_{t-1}, \beta_t I)$$

Log:
$$\log q(x_t \mid x_{t-1}) = -\frac{1}{2\beta_t} \|x_t - \sqrt{1-\beta_t} x_{t-1}\|^2 + \text{const}$$

**Closed-form term** (Ch1-03):
$$q(x_{t-1} \mid x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_{t-1}} x_0, (1-\bar{\alpha}_{t-1}) I)$$

Log:
$$\log q(x_{t-1} \mid x_0) = -\frac{1}{2(1-\bar{\alpha}_{t-1})} \|x_{t-1} - \sqrt{\bar{\alpha}_{t-1}} x_0\|^2 + \text{const}$$

**Closed-form denominator**:
$$q(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t) I)$$

Log:
$$\log q(x_t \mid x_0) = -\frac{1}{2(1-\bar{\alpha}_t)} \|x_t - \sqrt{\bar{\alpha}_t} x_0\|^2 + \text{const}$$

**Posterior computation**:
$$\log q(x_{t-1} \mid x_t, x_0) = \log q(x_t \mid x_{t-1}) + \log q(x_{t-1} \mid x_0) - \log q(x_t \mid x_0) + \text{const}$$

$$= -\frac{1}{2\beta_t} \|x_t - \sqrt{1-\beta_t} x_{t-1}\|^2$$
$$\quad - \frac{1}{2(1-\bar{\alpha}_{t-1})} \|x_{t-1} - \sqrt{\bar{\alpha}_{t-1}} x_0\|^2$$
$$\quad + \frac{1}{2(1-\bar{\alpha}_t)} \|x_t - \sqrt{\bar{\alpha}_t} x_0\|^2 + \text{const}$$

$x_{t-1}$ 에 대해 이차형식을 정리하면 (전개 후 모음):

**Coefficient of** $x_{t-1}^2$:
$$\frac{1-\beta_t}{\beta_t} + \frac{1}{1-\bar{\alpha}_{t-1}} = \frac{(1-\beta_t)(1-\bar{\alpha}_{t-1}) + \beta_t}{\beta_t(1-\bar{\alpha}_{t-1})}$$

$$= \frac{1 - \bar{\alpha}_{t-1} - \beta_t(1-\bar{\alpha}_{t-1}) + \beta_t}{\beta_t(1-\bar{\alpha}_{t-1})}$$

$$= \frac{1 - \bar{\alpha}_t}{\beta_t(1-\bar{\alpha}_{t-1})}$$

따라서 variance:
$$\tilde{\beta}_t = \frac{\beta_t(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}$$

**Coefficient of** $x_{t-1}$ (linear term):
$$-2 \cdot \frac{\sqrt{1-\beta_t}}{\beta_t} x_t - 2 \cdot \frac{\sqrt{\bar{\alpha}_{t-1}}}{1-\bar{\alpha}_{t-1}} x_0$$

Mean 은 quadratic form 을 complete square 로 변환해 얻음:
$$\tilde{\mu}_t = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t$$

확인: 계수들을 역으로 계산하면 quadratic 과 일치.

$$\square$$

### 정리 5.2: Alternative Parameterization (Noise Reparameterization)

**명제**: Posterior mean 을 다음으로도 표현할 수 있다:
$$\tilde{\mu}_t = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_t \right)$$

여기서 $\epsilon_t = \frac{1}{\sqrt{1-\bar{\alpha}_t}}(x_t - \sqrt{\bar{\alpha}_t} x_0)$ 는 implicit noise (Ch1-03).

**증명**: 
$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon_t$ (reparameterization)

이를 위의 $\tilde{\mu}_t$ 정의에 대입하고 정리하면 동치임을 확인할 수 있다.

## 💻 NumPy / PyTorch 구현 검증

### 실험 1: Posterior Mean/Variance 계산

```python
import torch

# Setup
T = 50
t = 25  # 특정 시간
beta = torch.linspace(0.0001, 0.02, T)
alpha = 1 - beta
alpha_bar = torch.cumprod(alpha, dim=0)

# 데이터와 노이즈
x0 = torch.tensor([1.0])
epsilon = torch.tensor([0.5])

# x_t 계산 (forward closed-form)
x_t = torch.sqrt(alpha_bar[t]) * x0 + torch.sqrt(1 - alpha_bar[t]) * epsilon

# Posterior mean 계산 (정의 5.1)
alpha_bar_t_minus_1 = alpha_bar[t-1] if t > 0 else torch.tensor(1.0)
alpha_t = alpha[t]
beta_t = beta[t]

numerator_1 = torch.sqrt(alpha_bar_t_minus_1) * beta_t
numerator_2 = torch.sqrt(alpha_t) * (1 - alpha_bar_t_minus_1)
denominator = 1 - alpha_bar[t]

tilde_mu_t = (numerator_1 * x0 + numerator_2 * x_t) / denominator
tilde_beta_t = (1 - alpha_bar_t_minus_1) * beta_t / denominator

print(f"Posterior at t={t}:")
print(f"  x0={x0.item():.4f}, x_t={x_t.item():.4f}")
print(f"  tilde_mu_t = {tilde_mu_t.item():.4f}")
print(f"  tilde_beta_t = {tilde_beta_t.item():.6f}")
print(f"  Posterior: N({tilde_mu_t.item():.4f}, {tilde_beta_t.item():.6f})")
```

### 실험 2: Bayes' Rule 검증 (수치)

```python
import torch
from scipy.stats import norm

# Toy 1D case
x0 = torch.tensor([0.5])
t = 30
T = 100

beta = torch.linspace(0.0001, 0.02, T)
alpha = 1 - beta
alpha_bar = torch.cumprod(alpha, dim=0)

# Forward forward one step
alpha_t = alpha[t]
alpha_bar_t = alpha_bar[t]
alpha_bar_t_minus_1 = alpha_bar[t-1]
beta_t = beta[t]

# x_t 값
eps_explicit = torch.tensor([0.3])
x_t_val = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * eps_explicit

# Posterior mean (정의)
tilde_mu = (torch.sqrt(alpha_bar_t_minus_1) * beta_t * x0 + 
            torch.sqrt(alpha_t) * (1 - alpha_bar_t_minus_1) * x_t_val) / (1 - alpha_bar_t)

# Sampling from posterior
n_samples = 100000
tilde_beta = (1 - alpha_bar_t_minus_1) * beta_t / (1 - alpha_bar_t)

samples = torch.randn(n_samples) * torch.sqrt(tilde_beta) + tilde_mu

print(f"Posterior sampling check:")
print(f"  Theoretical mean: {tilde_mu.item():.6f}")
print(f"  Empirical mean:   {samples.mean().item():.6f}")
print(f"  Theoretical var:  {tilde_beta.item():.8f}")
print(f"  Empirical var:    {samples.var().item():.8f}")
```

### 실험 3: 노이즈 Reparameterization

```python
import torch

T = 100
beta = torch.linspace(0.0001, 0.02, T)
alpha = 1 - beta
alpha_bar = torch.cumprod(alpha, dim=0)

x0 = torch.tensor([2.0])
t = 50

# 노이즈 기반 계산
epsilon = torch.randn(1)
x_t = torch.sqrt(alpha_bar[t]) * x0 + torch.sqrt(1 - alpha_bar[t]) * epsilon

# 방법 1: 기본 정의
alpha_bar_tm1 = alpha_bar[t-1]
beta_t = beta[t]
alpha_t = alpha[t]

mean_v1 = (torch.sqrt(alpha_bar_tm1) * beta_t * x0 + 
           torch.sqrt(alpha_t) * (1 - alpha_bar_tm1) * x_t) / (1 - alpha_bar[t])

# 방법 2: 노이즈 reparameterization
# epsilon = (x_t - sqrt(alpha_bar[t]) * x0) / sqrt(1 - alpha_bar[t])
mean_v2 = (1 / torch.sqrt(alpha_t)) * (x_t - beta_t / torch.sqrt(1 - alpha_bar[t]) * epsilon)

print(f"Mean comparison:")
print(f"  Method 1 (definition): {mean_v1.item():.6f}")
print(f"  Method 2 (noise param): {mean_v2.item():.6f}")
print(f"  Difference: {(mean_v1 - mean_v2).abs().item():.8f}")
```

## 🔗 실전 활용

**DDPM Loss (Ch2)**:
```python
# Variational Lower Bound 의 L_{t-1} 항
# Loss = KL( q(x_{t-1} | x_t, x_0) || p_theta(x_{t-1} | x_t) )
# = 0.5 * ( ||tilde_mu_t - mu_theta(x_t, t)||^2 / tilde_beta_t + ... )

def ddpm_loss_term(x0, x_t, t, model, beta, alpha_bar):
    # True posterior mean
    alpha_bar_tm1 = alpha_bar[t-1]
    beta_t = beta[t]
    alpha_t = 1 - beta_t
    
    tilde_mu = (torch.sqrt(alpha_bar_tm1) * beta_t * x0 + 
                torch.sqrt(alpha_t) * (1 - alpha_bar_tm1) * x_t) / (1 - alpha_bar[t])
    
    # Predicted mean
    mu_pred = model(x_t, t)
    
    # KL divergence (simplified)
    kl_loss = ((tilde_mu - mu_pred) ** 2).mean()
    
    return kl_loss
```

**Reverse Process의 학습 target**:
- 신경망 $\mu_\theta(x_t, t)$ 는 $\tilde{\mu}_t$ 를 모방
- 또는 equivalently, noise $\epsilon_\theta(x_t, t)$ 는 $\epsilon$ 를 예측

## ⚖️ 가정과 한계

| 항목 | 설명 | 주의사항 |
|------|------|---------|
| **Gaussian 조건부** | 분자/분모 모두 Gaussian → posterior 도 Gaussian | 정확함; Markov 가정 필수 |
| **Diagonal covariance** | $\Sigma = \beta_t I$ (대각선) | 상관성 무시; 일반화 어려움 |
| **Small-$\beta$ regime** | $\beta_t$ 작을 때만 근사 타당 | 보통 만족; $\beta_t \sim 10^{-4}$-$10^{-2}$ |
| **Fixed schedule** | $\beta$ 를 미리 정함 | 최적 schedule 은 문제마다 다를 수 있음 |

## 📌 핵심 정리

$$\boxed{q(x_{t-1} \mid x_t, x_0) = \mathcal{N}(\tilde{\mu}_t, \tilde{\beta}_t I)}$$

$$\boxed{\tilde{\mu}_t = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t}$$

$$\boxed{\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t}$$

| 항 | 의미 | 역할 |
|-----|------|------|
| **첫 항** ($\propto x_0$) | Signal from data | 초기 신호 기억 |
| **둘째 항** ($\propto x_t$) | Signal from current state | 현재 관찰 반영 |
| **$\tilde{\beta}_t$** | Posterior precision | 불확실성 정량화 |

## 🤔 생각해볼 문제

### 문제 1 (기초): Posterior 의 Mean 두 항 해석

$\tilde{\mu}_t$ 의 두 계수:
$$c_1 = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t}, \quad c_2 = \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}$$

$t$ 가 증가할 때:
- $c_1$ (from $x_0$) 은 어떻게 변하는가?
- $c_2$ (from $x_t$) 은 어떻게 변하는가?

직관적으로 설명하시오.

<details>
<summary>해설</summary>

$t$ 증가 ⟹ $\bar{\alpha}_t$ 감소 (누적 signal decay).

$c_1 = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t}$:
- Numerator: $\sqrt{\bar{\alpha}_{t-1}}$ 감소
- Denominator: $1 - \bar{\alpha}_t$ 증가
- 결과: $c_1$ 빠르게 감소 → $x_0$ 의 영향력 줄어듦

$c_2 = \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}$:
- Numerator: $(1 - \bar{\alpha}_{t-1})$ 증가 (노이즈 누적)
- Denominator: $(1 - \bar{\alpha}_t)$ 증가 (더 빠름)
- 결과: $c_2$ 대체로 증가 → $x_t$ (현재 관찰) 의 영향력 증가

직관: 초반에는 원본 $x_0$ 정보 중요, 
후반 (많이 노이즈) 되면 현재 $x_t$ 만 믿을 수 있음.

</details>

### 문제 2 (심화): Variance $\tilde{\beta}_t$ 의 극한

$\beta_t \to 0$ (very small noise) 일 때,
$$\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t$$

는 어떻게 되는가? Posterior 가 더 "certain" 해지는가?

<details>
<summary>해설</summary>

$\beta_t \to 0$ ⟹ forward noise 가 거의 없음

분자/분모 비율:
$$\frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_{t-1} - \beta_t(1-\bar{\alpha}_{t-1})} \approx 1 + O(\beta_t)$$

따라서:
$$\tilde{\beta}_t \approx (1 + O(\beta_t)) \beta_t \sim O(\beta_t^2)$$

결론: $\beta_t$ 가 작을수록 posterior variance 는 **quadratic하게** 작아짐.
즉, 한 스텝은 noise 가 거의 없으므로 posterior 는 아주 certain.

이는 Feller 1949 의 정당화: small-$\beta$ ⟹ true reverse ≈ Gaussian with tiny variance.

</details>

### 문제 3 (논문 비평): Posterior vs Model Parameterization

DDPM 은 posterior mean $\tilde{\mu}_t$ 를 신경망으로 배우는 대신,
**노이즈 $\epsilon_\theta$** 를 배운다 (noise prediction).

왜 이런 reparameterization 을 할까?
(Hint: 학습 안정성, gradient flow)

<details>
<summary>해설</summary>

**직접 mean 학습**:
- Loss: $\|\tilde{\mu}_t - \mu_\theta(x_t, t)\|^2$
- 문제: $x_t$ 의 scale 이 크면 gradient explode (tail 에서)

**Noise prediction**:
- Loss: $\|\epsilon - \epsilon_\theta(x_t, t)\|^2$ (normalized)
- 장점: noise 는 항상 unit variance ~ N(0,I)
- Gradient flow 가 안정적 (적응형 normalize)

**코드 관점**:
```python
# 직접 mean
loss_mean = (tilde_mu - model_mean(x_t, t)).pow(2).mean()

# Noise prediction
eps = (x_t - torch.sqrt(alpha_bar[t]) * x0) / torch.sqrt(1 - alpha_bar[t])
loss_noise = (eps - model_noise(x_t, t)).pow(2).mean()
```

Noise prediction 이 numerically 더 안정적!

</details>

---

<div align="center">

[◀ 이전](./04-reverse-process.md) | [📚 README](../README.md) | [다음 ▶](../ch2-elbo/01-vlb-decomposition.md)

</div>
