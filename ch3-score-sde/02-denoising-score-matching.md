# 2. Denoising Score Matching

## 🎯 핵심 질문
Vincent (2011)의 Denoising Score Matching은 무엇이고, 어떻게 정확한 score 추정을 피할 수 있는가? 왜 이것이 DDPM의 noise prediction과 동등한가?

## 🔍 왜 Score Matching인가?

$\nabla_x \log p(x)$를 직접 추정하려면 정규화 상수 $Z = \int e^{-E(x)}dx$가 필요하다 (자유 에너지 계산, tractable하지 않음). Score matching은 이를 우회하고, **noisy 데이터에서의 score만 학습**하면 된다는 영리한 아이디어다.

## 📐 수학적 선행 조건
- 변분 미분(functional derivative): $\frac{\delta}{\delta x}$
- 조건부 분포: $q(\tilde x|x)$, 베이즈 정리
- 제곱 오차: $\mathbb{E}[\|f(x)\|^2]$의 gradient
- Gaussian의 $\log$ 미분: $\nabla \log \mathcal{N}(y|\mu, \Sigma) = -\Sigma^{-1}(y - \mu)$

## 📖 직관적 이해

원본 데이터 $x$에 Gaussian 노이즈 $\mathcal{N}(0, \sigma^2I)$를 더해 $\tilde x = x + \sigma\epsilon$를 만든다. 이제 문제는:
- **Forward**: 깨끗한 $x$가 주어졌을 때, 노이지 버전 $\tilde x$의 score는?
- **Answer**: $s(\tilde x|x) = -\epsilon/\sigma$ (직접 계산 가능!)

이를 신경망으로 학습하면, 실제 데이터에 가까운 score를 배울 수 있다.

## ✏️ 엄밀한 정의

**정의 3.3 (Perturbed 분포)**  
데이터 분포 $p(x)$, 노이즈 레벨 $\sigma > 0$에 대해
$$q_\sigma(\tilde x) = \int p(x) \mathcal{N}(\tilde x | x, \sigma^2 I) dx$$
를 perturbed 분포라 한다.

**정의 3.4 (Denoising Score Matching 목적 함수)**  
$$L_{\text{DSM}}(\theta) = \mathbb{E}_{x \sim p, \epsilon \sim \mathcal{N}(0,I)} \left[\left\| s_\theta(\tilde x) - \nabla_{\tilde x} \log q_\sigma(\tilde x|x) \right\|^2 \right]$$
여기서 $\tilde x = x + \sigma\epsilon$.

## 🔬 정리와 증명

**정리 3.3 (Vincent의 Score Matching 항등식)**  
$$\mathbb{E}_{q_\sigma}[\|s_\theta(\tilde x) - \nabla_{\tilde x}\log q_\sigma(\tilde x|x)\|^2] = \mathbb{E}_{q_\sigma}[\|s_\theta(\tilde x) - \nabla_{\tilde x}\log q_\sigma(\tilde x)\|^2] + C$$
여기서 $C$는 $\theta$에 무관한 상수이다.

증명:
$$\|a - b\|^2 = \|a\|^2 - 2a^\top b + \|b\|^2$$

좌변을 전개하면:
$$\mathbb{E}[\|s_\theta - \nabla\log q(\tilde x|x)\|^2] = \mathbb{E}[\|s_\theta\|^2] - 2\mathbb{E}[s_\theta^\top \nabla\log q(\tilde x|x)] + \mathbb{E}[\|\nabla\log q(\tilde x|x)\|^2]$$

우변을 전개하면:
$$\mathbb{E}[\|s_\theta - \nabla\log q(\tilde x)\|^2] = \mathbb{E}[\|s_\theta\|^2] - 2\mathbb{E}[s_\theta^\top \nabla\log q(\tilde x)] + \mathbb{E}[\|\nabla\log q(\tilde x)\|^2]$$

차이를 구하면:
$$\text{LHS} - \text{RHS} = -2\mathbb{E}[s_\theta^\top(\nabla\log q(\tilde x|x) - \nabla\log q(\tilde x))] + \mathbb{E}[\|\nabla\log q(\tilde x|x)\|^2 - \|\nabla\log q(\tilde x)\|^2]$$

Bayes 정리에서 $q(\tilde x|x) = q(\tilde x) p(x|\tilde x) / p(x)$이므로:
$$\nabla_{\tilde x} \log q(\tilde x|x) - \nabla_{\tilde x} \log q(\tilde x) = \nabla_{\tilde x} \log p(x|\tilde x)$$

따라서 우변의 추가 항은 $\theta$에 무관하다. $\square$

**정리 3.4 (Gaussian 경우의 조건부 score)**  
$q(\tilde x|x) = \mathcal{N}(\tilde x | x, \sigma^2 I)$일 때,
$$\nabla_{\tilde x} \log q(\tilde x|x) = -\frac{\tilde x - x}{\sigma^2} = -\frac{\epsilon}{\sigma}$$
(단, $\tilde x = x + \sigma\epsilon$).

증명: Gaussian의 로그 확률은 $\log p = -\frac{1}{2\sigma^2}\|y - \mu\|^2 + \text{const}$이므로 $\nabla \log p = -\frac{1}{\sigma^2}(y - \mu)$. $\square$

## 💻 NumPy / PyTorch 구현 검증

### 실험 1: DSM 목적 함수 구현 및 검증
```python
import torch
import torch.nn as nn

class ScoreNet(nn.Module):
    def __init__(self, input_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
    
    def forward(self, x):
        return self.net(x)

def dsm_loss(model, x_clean, sigma=0.1):
    """Denoising Score Matching loss"""
    epsilon = torch.randn_like(x_clean)
    x_noisy = x_clean + sigma * epsilon
    
    # Score 추정
    score_est = model(x_noisy)
    
    # 조건부 score (정답)
    score_true = -epsilon / sigma
    
    loss = torch.mean((score_est - score_true) ** 2)
    return loss

# 테스트: 1D Gaussian mixture
x_clean = torch.randn(64, 2) * 2  # 2D data
model = ScoreNet(input_dim=2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(1000):
    loss = dsm_loss(model, x_clean, sigma=0.1)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 200 == 0:
        print(f"Step {step}: Loss = {loss.item():.4f}")
```

### 실험 2: DDPM Noise Prediction과의 등가성
```python
# DDPM의 L_simple: E[||epsilon - model(x_t, t)||^2]
# DSM: E[||score_model - (-epsilon/sigma)||^2] with sigma = sqrt(1 - alpha_bar_t)

def ddpm_loss(model, x_0, t, alphas_cumprod):
    """DDPM noise prediction loss"""
    batch_size = x_0.shape[0]
    alpha_bar_t = alphas_cumprod[t]
    sigma_t = torch.sqrt(1 - alpha_bar_t)
    
    # Forward: x_t = sqrt(alpha_bar_t) * x_0 + sigma_t * epsilon
    epsilon = torch.randn_like(x_0)
    x_t = torch.sqrt(alpha_bar_t) * x_0 + sigma_t * epsilon
    
    # Predict noise
    epsilon_pred = model(x_t, t)
    
    loss = torch.mean((epsilon_pred - epsilon) ** 2)
    return loss

def dsm_equivalent(model, x_0, t, alphas_cumprod):
    """Equivalent DSM formulation"""
    alpha_bar_t = alphas_cumprod[t]
    sigma_t = torch.sqrt(1 - alpha_bar_t)
    
    epsilon = torch.randn_like(x_0)
    x_t = torch.sqrt(alpha_bar_t) * x_0 + sigma_t * epsilon
    
    # Score prediction (등가 변환)
    score_pred = model(x_t, t) * (-sigma_t)
    score_true = -epsilon / sigma_t
    
    loss = torch.mean((score_pred - score_true) ** 2)
    return loss

# 두 loss는 동일해야 함 (constant factor 제외)
alphas_cumprod = torch.linspace(0.9999, 0.0001, 1000)
t = torch.tensor([50])
x_0 = torch.randn(4, 2)

loss_ddpm = ddpm_loss(model, x_0, t, alphas_cumprod)
loss_dsm = dsm_equivalent(model, x_0, t, alphas_cumprod)
print(f"DDPM loss: {loss_ddpm.item():.4f}, DSM loss: {loss_dsm.item():.4f}")
```

### 실험 3: 다양한 σ에서의 학습 효과
```python
import matplotlib.pyplot as plt

sigmas = [0.05, 0.1, 0.2, 0.5]
results = {}

for sigma in sigmas:
    model = ScoreNet(input_dim=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    
    for step in range(500):
        x = torch.randn(64, 2)
        loss = dsm_loss(model, x, sigma=sigma)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    results[f'σ={sigma}'] = losses

for label, losses in results.items():
    plt.plot(losses, label=label, alpha=0.7)

plt.xlabel('Training Step')
plt.ylabel('DSM Loss')
plt.legend()
plt.title('Effect of Noise Level on DSM Training')
plt.show()
```

## 🔗 실전 활용

**Diffusion Model 훈련**
- 각 time step $t$에서 noise level $\sigma_t = \sqrt{1 - \bar{\alpha}_t}$로 Gaussian perturbation
- DSM/DDPM loss로 신경망 훈련: 정규화 상수를 알 필요 없음
- 생성 시에는 학습한 score로 역방향 이동

**Score 기반 생성 모델**
- 임의의 시그마에 대해 DSM으로 훈련 가능
- Continuous time의 경우 VE-SDE로 일반화

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| 조건부 score를 정확히 계산 | Gaussian 외 형태에서는 계산 복잡 |
| 데이터에 Gaussian 노이즈만 추가 | 다른 노이즈 형태에는 일반화 필요 |
| 기울기 추정이 unbiased | 유한 batch에서 stochastic gradient bias |
| 일정한 σ로 훈련 | 가변 σ (annealing)가 더 효율적 |

## 📌 핵심 정리

**Denoising Score Matching은 정규화 상수를 모르는 조건에서 score를 학습한다. Gaussian perturbation 하에서 조건부 score는 닫힌 형태 $-\epsilon/\sigma$로 계산되며, 이를 신경망이 추정하는 것이 목표다. 가중 DSM은 DDPM의 noise prediction 목적 함수와 정확히 동등하며, 이는 diffusion model의 이론적 토대를 제공한다.**

## 🤔 생각해볼 문제

### 문제 1
왜 $\mathbb{E}[s_\theta(\tilde x) - \nabla\log q(\tilde x|x)]^2$를 최소화하면 $\mathbb{E}[s_\theta(\tilde x) - \nabla\log q(\tilde x)]^2$도 최소화되는가? 물리적 의미는?

<details>
<summary>해설</summary>

정리 3.3에서 두 목적 함수는 $\theta$에 무관한 상수 차이만 있다. 즉, 조건부 score를 학습하면 자동으로 진정한 데이터 분포의 score 방향으로 수렴한다는 의미다. 이는 노이즈 있는 데이터에서 깨끗한 데이터의 구조를 배우는 강력한 방법이다.
</details>

### 문제 2
DSM에서 σ가 매우 크면 어떻게 되는가? 매우 작으면?

<details>
<summary>해설</summary>

σ가 크면: 노이즈가 지배적이라 score가 원점 방향으로만 작동 (정보 손실)  
σ가 작으면: 조건부 score와 데이터 score의 차이가 커지고 (정리 3.3의 상수 항 증가), 기울기 신호가 약해진다. 최적 σ는 데이터 특성에 따라 결정된다.
</details>

### 문제 3
DDPM의 가중 loss $L_{\text{simple}} = \mathbb{E}_{t,x,\epsilon}[\|s_\theta(x_t,t) - (-\epsilon/\sigma_t)\|^2]$가 왜 DSM과 동등한가?

<details>
<summary>해설</summary>

DDPM에서 $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$이고, score와 noise 간의 관계는 $s(x_t) = -\epsilon/\sigma_t$다 ($\sigma_t = \sqrt{1-\bar{\alpha}_t}$). 따라서 noise prediction loss는 score matching loss와 일치한다.
</details>

---

<div align="center">

[◀ 이전](./01-score-langevin.md) | [📚 README](../README.md) | [다음 ▶](./03-ncsn.md)

</div>
