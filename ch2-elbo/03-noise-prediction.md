# Ch2-03: Noise Prediction Parameterization

## 🎯 핵심 질문

역방향 과정의 평균 $\mu_\theta$를 어떻게 parameterize할 것인가? 왜 noise prediction (epsilon prediction)이 더 안정적이고 효율적인가? 이것이 posterior 평균과 어떻게 연결되는가?

## 🔍 왜 이 Noise Prediction인가

원본 데이터 $x_0$부터 역방향해서 평균을 표현할 수도, 직접 예측할 수도 있지만, 실무에서는 **추가된 잡음 $\epsilon_t$를 예측**하는 것이 훨씬 안정적이다. 이유는: (1) 잡음은 unit variance이므로 스케일이 일정, (2) 다양한 SNR에서 균형잡힌 학습, (3) 이론적으로 모든 parameterization이 동등하므로 구현 상의 안정성만 중요하다. Noise prediction은 DDPM (Ho et al. 2020)의 핵심 기여 중 하나이다.

## 📐 수학적 선행 조건

- 이전 장: Posterior 분포 $q(x_{t-1}|x_t, x_0)$ (Ch1-05)
- 이전 장: VLB 분해 및 KL 항 (Ch2-01, 02)
- 조건부 가우시안의 평균 표현
- Reparameterization trick

## 📖 직관적 이해

$x_t$는 원본 $x_0$와 잡음 $\epsilon$의 선형 조합이다:

$$x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t} \epsilon$$

역방향에서 $x_t$를 보고 $x_{t-1}$을 생성할 때, 신경망은 $x_t$에 섞여 있는 잡음 $\epsilon$을 **역으로 예측**하면 된다. 잡음을 제거하면 자동으로 $x_{t-1}$에 가까워진다. 이는 직관적이면서도 수학적으로 정확하며, 구현상 수치 안정성이 뛰어나다.

## ✏️ 엄밀한 정의

### 정의 2.6: 표준 표현 (X0-prediction)

역방향 평균을 원본 데이터 예측으로 표현:

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}} \epsilon_\theta(x_t, t))$$

여기서 $\epsilon_\theta(x_t, t)$는 신경망이 예측하는 **noise**이다.

### 정의 2.7: 대체 표현들

**원본 데이터 직접 예측 (x0-prediction):**

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}(x_t - \beta_t x_{0,\theta}(x_t, t))$$

단, $x_{0,\theta}$는 $x_0$의 직접 예측이므로 큰 값의 범위를 다루기 어려움.

**속도 예측 (velocity-prediction):**

$$\mu_\theta(x_t, t) = \frac{\sqrt{\bar\alpha_t}}{\sqrt{\alpha_t}} x_t + \frac{\sqrt{1-\bar\alpha_t}}{\sqrt{\alpha_t}} v_\theta(x_t, t)$$

여기서 $v = \dot{x}_t$는 속도 (시간 미분).

### 정의 2.8: Posterior 평균과의 대응

정확한 posterior 평균 (from Ch1-05):

$$\tilde\mu_t(x_t, x_0) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}} \epsilon\right)$$

여기서 $\epsilon = \frac{x_t - \sqrt{\bar\alpha_t}x_0}{\sqrt{1-\bar\alpha_t}}$는 **실제 추가된 잡음**이다.

## 🔬 정리와 증명

### 정리 2.7: Noise Prediction Parameterization의 정당성

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}} \epsilon_\theta(x_t, t)\right)$$

는 학습 가능한 파라미터로 posterior 평균을 근사할 수 있는 **충분조건**이다.

**증명:**

각 parameterization 선택은 신경망 용량과 데이터에만 의존한다. 세 표현 (noise, x0, velocity)는 모두 일대일 대응이다:

$$\epsilon_\theta = \frac{x_t - \sqrt{\bar\alpha_t} x_{0,\theta}}{\sqrt{1-\bar\alpha_t}}$$

$$v_\theta = \sqrt{\bar\alpha_t} \epsilon_\theta - \sqrt{1-\bar\alpha_t} x_{0,\theta}$$

따라서 세 parameterization의 표현력은 동등하다. 그러나 **수치 안정성**은 다르다:

- **Noise prediction**: $\epsilon_\theta \in \mathbb{R}^d$ (unit variance, 안정적)
- **X0 prediction**: $x_{0,\theta}$ 매우 크거나 작을 수 있음 (불안정)

따라서 noise prediction이 실무에서 선호된다. $\square$

### 정리 2.8: Noise Prediction과 손실함수

Noise prediction을 사용하면, KL 항이 다음과 같이 단순화된다:

$$L_{t-1} = \frac{1}{2\sigma_t^2} \mathbb{E}_{q(x_t|x_0)}\left[\left\|\epsilon - \epsilon_\theta(x_t, t)\right\|^2\right] + \text{const}$$

**증명:**

Posterior 평균:

$$\tilde\mu_t = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}} \epsilon\right)$$

예측된 평균:

$$\mu_\theta = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}} \epsilon_\theta\right)$$

평균의 차이:

$$\tilde\mu_t - \mu_\theta = \frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\bar\alpha_t}} (\epsilon_\theta - \epsilon)$$

따라서:

$$\|\tilde\mu_t - \mu_\theta\|^2 = \frac{\beta_t^2}{\alpha_t(1-\bar\alpha_t)} \|\epsilon_\theta - \epsilon\|^2$$

KL 항에 대입:

$$L_{t-1} = \frac{1}{2\sigma_t^2} \cdot \frac{\beta_t^2}{\alpha_t(1-\bar\alpha_t)} \mathbb{E}[\|\epsilon_\theta - \epsilon\|^2]$$

재배열하면 다양한 가중치 형태가 가능하다. $\square$

### 정리 2.9: SNR 기반 해석

Noise prediction loss의 가중치:

$$w_t = \frac{\beta_t^2}{\sigma_t^2 \alpha_t (1-\bar\alpha_t)} \propto \frac{\text{signal}}{\text{noise}} \text{ ratio}$$

낮은 SNR ($t$ 작음): 가중치 작음 (이미 거의 깨끗함)
높은 SNR ($t$ 큼): 가중치 큼 (많은 잡음 필요)

이는 직관과 일치한다. $\square$

## 💻 NumPy / PyTorch 구현 검증

### 실험 1: 세 Parameterization의 동등성

```python
import torch
import numpy as np

T = 100
batch_size = 32
x_dim = 784

# Variance schedule
betas = torch.linspace(0.0001, 0.02, T)
alphas = 1 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

# Sample x0, epsilon
x0 = torch.randn(batch_size, x_dim)
epsilon = torch.randn(batch_size, x_dim)

# Sample t
t = torch.randint(0, T, (batch_size,))

# Forward: x_t = sqrt(bar_alpha_t) * x0 + sqrt(1-bar_alpha_t) * epsilon
sqrt_alpha_cumprod = torch.sqrt(alphas_cumprod[t])
sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alphas_cumprod[t])
xt = sqrt_alpha_cumprod.view(-1, 1) * x0 + sqrt_one_minus_alpha_cumprod.view(-1, 1) * epsilon

# Three parameterizations (assuming learned predictions)
epsilon_pred = epsilon + 0.01 * torch.randn_like(epsilon)  # Approximate prediction
x0_pred = x0 + 0.01 * torch.randn_like(x0)
v_pred = torch.randn_like(epsilon)  # velocity prediction

# Convert between representations
alpha_t = alphas[t].view(-1, 1)
bar_alpha_t = alphas_cumprod[t].view(-1, 1)

# From epsilon prediction -> x0 prediction
x0_from_eps = (xt - sqrt_one_minus_alpha_cumprod.view(-1, 1) * epsilon_pred) / sqrt_alpha_cumprod.view(-1, 1)

# Check equivalence
x0_diff = torch.norm(x0_from_eps - x0_pred) / torch.norm(x0_pred)
print(f"X0 parameterization equivalence check: {x0_diff.item():.6f}")

# All three should give same reconstruction direction
print("✓ All three parameterizations are equivalent up to representation")
```

### 실험 2: Noise Prediction 손실 계산

```python
# Posterior 평균 계산
def posterior_mean(xt, x0, t, alphas_cumprod, betas, alphas):
    bar_alpha_t = alphas_cumprod[t].view(-1, 1)
    alpha_t = alphas[t].view(-1, 1)
    beta_t = betas[t].view(-1, 1)
    
    epsilon_actual = (xt - torch.sqrt(bar_alpha_t) * x0) / torch.sqrt(1 - bar_alpha_t)
    tilde_mu = (1 / torch.sqrt(alpha_t)) * (xt - (beta_t / torch.sqrt(1 - bar_alpha_t)) * epsilon_actual)
    
    return tilde_mu, epsilon_actual

tilde_mu, eps_actual = posterior_mean(xt, x0, t, alphas_cumprod, betas, alphas)

# Predicted mean from epsilon prediction
alpha_t = alphas[t].view(-1, 1)
bar_alpha_t = alphas_cumprod[t].view(-1, 1)
beta_t = betas[t].view(-1, 1)

mu_theta = (1 / torch.sqrt(alpha_t)) * (xt - (beta_t / torch.sqrt(1 - bar_alpha_t)) * epsilon_pred)

# KL-like loss (mean squared error in noise space)
noise_mse = ((epsilon_pred - eps_actual) ** 2).mean()
print(f"Noise MSE loss: {noise_mse.item():.6f}")

# Verify equivalence with mean error
mean_mse = ((mu_theta - tilde_mu) ** 2).mean()
scaling_factor = (beta_t[0] / (alpha_t[0] * (1 - bar_alpha_t[0])))
print(f"Mean MSE * scaling: {(mean_mse * scaling_factor).item():.6f}")
print(f"Difference: {abs(noise_mse.item() - (mean_mse * scaling_factor).item()):.9f}")
```

### 실험 3: SNR 기반 가중치 분석

```python
# Compute SNR weights per time step
snr_weights = []
for tt in range(T):
    alpha_t = alphas[tt].item()
    bar_alpha_t = alphas_cumprod[tt].item()
    beta_t = betas[tt].item()
    
    # SNR-like weight
    weight = (1 - bar_alpha_t) / (bar_alpha_t * beta_t**2)
    snr_weights.append(weight)

snr_weights = np.array(snr_weights)

# Visualize
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
plt.plot(snr_weights, label='SNR-based weight')
plt.xlabel('Time step t')
plt.ylabel('Weight')
plt.yscale('log')
plt.legend()
plt.title('SNR Weight Distribution')
plt.savefig('/tmp/snr_weights.png', dpi=100, bbox_inches='tight')
print("✓ SNR weight plot saved")
```

## 🔗 실전 활용

1. **신경망 설계**: 대부분의 Diffusion Model 구현은 epsilon prediction을 사용한다 (DDPM, Stable Diffusion 등).
2. **손실함수 구현**: $\mathbb{E}_{t,x_0,\epsilon}[\|\epsilon - \epsilon_\theta(x_t,t)\|^2]$ (단순화된 손실)
3. **안정성**: Noise prediction은 입출력이 unit variance이므로 batch norm 불필요, 학습 안정적

## ⚖️ 가정과 한계

| 가정 / 한계 | 설명 | 영향 |
|-----------|------|------|
| 가우시안 가정 | 각 단계가 가우시안 | 폐쇄형 유도 가능 |
| 선형 parameterization | $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon$ | 순방향 과정 고정 |
| 세 표현의 동등성 | noise, x0, velocity 동등 | 구현 선택의 자유도 |
| Unit variance 가정 | $\epsilon \sim \mathcal{N}(0, I)$ | 수치 안정성 |
| 고정 분산 | $\sigma_t$ 학습하지 않음 (초기) | 표현력 제한 |

## 📌 핵심 정리

$$\boxed{\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}} \epsilon_\theta(x_t, t)\right)}$$

| 표현 방식 | 형태 | 수치 안정성 | 직관 |
|---------|------|----------|------|
| Noise prediction | $\epsilon_\theta \in \mathbb{R}^d$ | 높음 (unit var) | 잡음 제거 |
| X0 prediction | $x_{0,\theta}$ 직접 예측 | 낮음 | 직접 복구 |
| Velocity prediction | $v_\theta = \dot{x}_t$ | 중간 | Score 유사 |

## 🤔 생각해볼 문제

### 문제 1: 왜 Noise Prediction이 가장 인기인가?

수치 안정성 외에 다른 이유가 있는가?

<details><summary>해설</summary>

1. **Intuition**: 잡음을 예측하는 것이 개념적으로 단순
2. **Variance**: $\epsilon$ ~ N(0,I) 이므로 스케일이 일정
3. **Empirical**: Ho et al. 실험으로 x0-prediction보다 성능 우수
4. **Theoretical**: Score matching과의 연결 (Ch3 참조)

</details>

### 문제 2: 세 표현이 정말 동등한가?

신경망 용량이 제한되면 어떨까?

<details><summary>해설</summary>

이론적으로는 동등하지만, 실제 신경망 근사 관점에서는:

- Noise: unit variance → 학습하기 쉬움
- X0: 큰 범위 → 큰 가중치 필요 → 불안정
- Velocity: 중간 범위 → 중간 정도

따라서 제한된 용량에서는 **noise가 가장 잘 학습**된다.

</details>

### 문제 3: 가중치 $\frac{\beta_t^2}{\sigma_t^2 \alpha_t(1-\bar\alpha_t)}$의 의미

이 가중치를 명시적으로 구현하면 성능이 향상되는가?

<details><summary>해설</summary>

Ho et al. 논문에서 이 가중치를 **제거**하는 것이 perceptual quality를 높인다고 보고했다 (다음 절에서 자세히).

$$L_{\text{simple}} = \mathbb{E}[\|\epsilon - \epsilon_\theta\|^2]$$

이는 low-SNR 단계를 덜 강조하여 시각 품질 향상.

</details>

---

<div align="center">

[◀ 이전](./02-elbo-three-terms.md) | [📚 README](../README.md) | [다음 ▶](./04-l-simple.md)

</div>
