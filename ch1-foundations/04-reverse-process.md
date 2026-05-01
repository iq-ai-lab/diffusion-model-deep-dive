# 04. Reverse Process: Score Matching

## 🎯 핵심 질문

- Reverse process 는 forward 의 "정확한 역" 인가, 아니면 근사인가?
- Feller (1949) 의 결과가 왜 Gaussian 근사를 정당화하는가?
- Score function $\nabla_x \log p(x_t, t)$ 를 신경망으로 학습할 수 있는가?
- $T \to \infty$ 극한에서 reverse process 가 어떻게 Score-SDE 가 되는가?

## 🔍 왜 이 역과정이 생성 (Generation) 인가

Forward process 는:
$$x_0 \text{ (데이터)} \xrightarrow{+\text{noise}} x_T \text{ (가우시안 노이즈)}$$

**역과정** (reverse):
$$x_T \xrightarrow{-\text{noise prediction}} x_0 \text{ (복원된 데이터)}$$

역과정을 학습하려면, 각 시간 $t$ 에서 **노이즈를 빼야 할 방향** (score) 을 안다.
이것이 **conditional density** $q(x_{t-1} \mid x_t, x_0)$ 에서 나온다.

실제로는 $x_0$ 를 모르므로, 신경망이 **score** 를 학습한다.

## 📐 수학적 선행 조건

- **Bayes' Rule** : $p(A|B) = \frac{p(B|A)p(A)}{p(B)}$
- **Gaussian Multiplication** : $\mathcal{N}(a,A) \times \mathcal{N}(b,B)$ 의 합
- **Score Function** : $\nabla_x \log p(x)$
- **Fokker-Planck / Reverse SDE** (Ch1-01)
- **Closed-form** (Ch1-03)

## 📖 직관적 이해

```
Forward: Data ────────────────────────→ Noise
         x₀ ─(+ε₁)─→ x₁ ─(+ε₂)─→ ... x_T

Reverse: Noise ─────────────────────→ Data
         x_T ─(-predict ε_T)─→ x_{T-1} ─(-predict ε_{T-1})─→ ... x₀

════════════════════════════════════════════════════════════

조건부 분포 q(x_{t-1} | x_t, x_0):
  "x_t 를 봤을 때, 이전 상태 x_{t-1} 은?"
  (두 정보: forward 노이즈 history, 최종 상태)
  
  ← 이것도 Gaussian! (Feller)
  ← Mean 과 Variance closed-form 있음!
  
따라서:
  - q(x_{t-1} | x_t, x_0) 의 mean 을 알면
  - reverse 는 그쪽으로 noise 를 빼면 됨

그런데 x_0 를 모르니까:
  - 신경망이 score (gradient of log p) 학습
  - Score ≈ noise 를 빼는 방향
```

## ✏️ 엄밀한 정의

### 정의 4.1: Reverse Process

Forward process 역의 확률 과정:
$$p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(\mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

여기서:
- $\mu_\theta(x_t, t)$ : 신경망으로 학습되는 mean 함수
- $\Sigma_\theta(x_t, t)$ : variance (고정 또는 학습)
- $p_\theta$ : 학습되는 모델 (forward 의 역)

**Sampling**:
$$x_{t-1} = \mu_\theta(x_t, t) + \sqrt{\Sigma_\theta(x_t, t)} \, \eta, \quad \eta \sim \mathcal{N}(0,I)$$

### 정의 4.2: True Reverse $q(x_{t-1} \mid x_t, x_0)$

주어진 $x_t$ 와 $x_0$, 역과정의 "진실" 분포:
$$q(x_{t-1} \mid x_t, x_0) = \mathcal{N}(\tilde{\mu}_t, \tilde{\beta}_t I)$$

**중요**: 이것은 forward process 의 정의와 Bayes' rule 에서 유도됨.

구체적 형태는 Ch1-05 에서.

### 정의 4.3: Score Function

Density 의 gradient:
$$s_\theta(x_t, t) := \nabla_{x_t} \log p_\theta(x_t, t)$$

**역할**: Reverse mean 의 일부로 나타남.
$$\mu_\theta(x_t, t) \propto x_t + \text{(score term)}$$

## 🔬 정리와 증명

### 정리 4.1: Feller's Result (1949)

**명제** (정성적): Forward process 에서 각 $\beta_t$ 가 충분히 작으면,
reverse process 도 Gaussian 이 된다.

더 정확히, 
$$q(x_{t-1} \mid x_t, x_0) = \mathcal{N}(\tilde{\mu}_t, \tilde{\beta}_t I)$$

는 정확히 Gaussian 이고, variance $\tilde{\beta}_t$ 는 매우 작다 (order $\beta_t$).

**증명 스케치** (Feller 1949, Anderson 1982):

1. Forward: $q(x_t \mid x_{t-1})$ 는 Gaussian
2. Forward: $q(x_t \mid x_0)$ 도 Gaussian (closed-form)
3. 역: $q(x_{t-1} \mid x_t, x_0) = \frac{q(x_t \mid x_{t-1}) q(x_{t-1} \mid x_0)}{q(x_t \mid x_0)}$ (Bayes)
4. 분자: Gaussian × Gaussian = Gaussian (proportional)
5. 따라서 posterior 도 Gaussian

variance 가 작은 이유: $\beta_t \ll 1$ 이면, 
one-step forward 의 randomness 는 크지 않아서,
posterior 의 uncertainty 도 작다.

### 정리 4.2: Score Matching = Variance Minimization

**명제**: Reverse mean 을 다음으로 설정하면,
$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{1-\beta_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)$$

여기서 $\epsilon_\theta$ 는 예측된 노이즈인데, 이는
$$\mathbb{E}[\|\epsilon - \epsilon_\theta(x_t, t)\|^2]$$

최소화하는 것과 같다.

**증명** (sketch):
- Reverse loss: $\mathbb{E}[\|\tilde{\mu}_t - \mu_\theta(x_t, t)\|^2]$ (true reverse 에서 벗어나는 정도)
- $\tilde{\mu}_t$ 는 $x_t$ 와 $x_0$ 에 대한 식
- $x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(x_t - \sqrt{1-\bar{\alpha}_t} \epsilon)$ (Ch1-03 reparameterization 역변환)
- 대입하면 loss 는 $\epsilon$ 예측 error 가 됨

## 💻 NumPy / PyTorch 구현 검증

### 실험 1: Score Function의 기울기 이해

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

# 1D toy: q(x_t | x_0) = N(sqrt(alpha_bar_t) * x0, (1-alpha_bar_t) * I)
# Score s(x_t) = d/dx_t log q(x_t | x_0) = -(x_t - mean) / variance

torch.manual_seed(42)

x0 = torch.tensor([1.0])
alpha_bar = torch.tensor([0.9])  # t=50 과 같음

# x_t 의 분포: N(sqrt(0.9), 0.1)
mean_t = torch.sqrt(alpha_bar) * x0
var_t = 1 - alpha_bar

# 여러 x_t 값에서의 true score 계산
x_t_vals = torch.linspace(-2, 3, 100)
true_score = -(x_t_vals - mean_t) / var_t

print("Score function s(x_t) = d/dx_t log p(x_t | x_0):")
print(f"  mean_t = {mean_t.item():.4f}")
print(f"  var_t = {var_t.item():.4f}")
print(f"\nScore 는 high-density 영역으로의 gradient")
print(f"  At x_t=3.0: score ≈ {(-(3.0 - mean_t) / var_t).item():.2f} (왼쪽으로)")
print(f"  At x_t=0.5: score ≈ {(-(0.5 - mean_t) / var_t).item():.2f} (오른쪽으로)")
```

### 실험 2: Reverse Process Sampling

```python
import torch

torch.manual_seed(42)

# Setup
T = 100
beta = torch.linspace(0.0001, 0.02, T)
alpha = 1 - beta
alpha_bar = torch.cumprod(alpha, dim=0)

# 진짜 데이터에서 하나 뽑기
x0_real = torch.randn(1)

# Forward: x0 -> x_T
x_t = x0_real.clone()
for t in range(T):
    epsilon = torch.randn(1)
    x_t = torch.sqrt(alpha[t]) * x_t + torch.sqrt(beta[t]) * epsilon

print(f"Forward process:")
print(f"  x0 = {x0_real.item():.4f}")
print(f"  x_T = {x_t.item():.4f} (거의 표준정규분포)")

# Reverse: x_T -> x_0 (진짜 reverse, true score 알 때)
x_t_reverse = x_t.clone()
for t_rev in range(T-1, -1, -1):
    # True reverse: q(x_{t-1} | x_t, x_0)
    # 하지만 x_0 를 모르므로, 근사로 score 사용
    # s(x_t, t) ≈ -x_t / (1 - alpha_bar[t]) (무조건부 score)
    
    score = -x_t_reverse / (1 - alpha_bar[t])
    
    # Reverse step (variance 고정)
    mean_reverse = (x_t_reverse + beta[t_rev] * score) / torch.sqrt(alpha[t_rev])
    var_reverse = beta[t_rev]
    
    eta = torch.randn(1)
    x_t_reverse = mean_reverse + torch.sqrt(var_reverse) * eta

print(f"\nReverse process (true score approximation):")
print(f"  x_T = {x_t.item():.4f}")
print(f"  x_0_recovered = {x_t_reverse.item():.4f}")
print(f"  x_0_real = {x0_real.item():.4f}")
print(f"  Error: {(x_t_reverse - x0_real).abs().item():.4f}")
```

### 실험 3: Learned Score Function

```python
import torch
import torch.nn as nn

# 간단한 1-layer MLP 로 score function 학습
class SimpleScoreNet(nn.Module):
    def __init__(self, dim=1, time_dim=32):
        super().__init__()
        self.time_embedding = nn.Embedding(1000, time_dim)
        self.net = nn.Sequential(
            nn.Linear(dim + time_dim, 64),
            nn.ReLU(),
            nn.Linear(64, dim)
        )
    
    def forward(self, x, t):
        t_emb = self.time_embedding(t)
        return self.net(torch.cat([x, t_emb], dim=-1))

# 학습 루프 (toy)
model = SimpleScoreNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

T = 50
beta = torch.linspace(0.0001, 0.02, T)
alpha = 1 - beta
alpha_bar = torch.cumprod(alpha, dim=0)

# 100 배치 학습
for epoch in range(10):
    x0 = torch.randn(32)  # 배치
    
    # 랜덤 t
    t_indices = torch.randint(0, T, (32,))
    
    # x_t 생성
    eps = torch.randn(32)
    alpha_bar_t = alpha_bar[t_indices].unsqueeze(1)
    x_t = torch.sqrt(alpha_bar_t) * x0.unsqueeze(1) + torch.sqrt(1 - alpha_bar_t) * eps.unsqueeze(1)
    
    # True score
    true_score = -eps
    
    # Predicted score
    pred_score = model(x_t.squeeze(), t_indices)
    
    # Loss
    loss = (true_score.unsqueeze(1) - pred_score).pow(2).mean()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 3 == 0:
        print(f"Epoch {epoch}: loss={loss.item():.6f}")

print("Score network 학습 완료")
```

## 🔗 실전 활용

**DDPM 역과정**:
```python
# Reverse with learned model
def ddpm_reverse(model, x_T, T, beta):
    alpha = 1 - beta
    x = x_T
    
    for t in reversed(range(T)):
        z = torch.randn_like(x) if t > 0 else 0
        
        # Predicted noise
        eps_pred = model(x, t)
        
        # Reverse step
        mean = (x - (beta[t] / torch.sqrt(1 - alpha_bar[t])) * eps_pred) / torch.sqrt(alpha[t])
        x = mean + torch.sqrt(beta[t]) * z
    
    return x
```

**Score-based Diffusion (Song et al. 2021)**:
- Continuous time SDE
- Score matching loss: $\|\nabla_x \log p(x,t) - s_\theta(x,t)\|^2$
- Ch3-04 에서 상세

## ⚖️ 가정과 한계

| 항목 | 설명 | 주의사항 |
|------|------|---------|
| **Small-$\beta$ assumption** | $\beta_t \ll 1$ 일 때 reverse 도 Gaussian | 대개 만족; $\beta_t \sim 10^{-4}$-$10^{-2}$ |
| **Score approximation** | 신경망이 true score 를 근사 | 무조건부 score; 조건부는 Ch2 에서 |
| **Variance 고정** | DDPM 은 variance 를 고정 (학습 안 함) | Improved DDPM 은 학습 가능 |
| **Discretization** | Discrete step 으로 근사 (continuous ×) | $T$ 충분히 크면 OK; DDIM 으로 가속 |

## 📌 핵심 정리

$$\boxed{p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(\mu_\theta(x_t, t), \Sigma_\theta(x_t, t))}$$

$$\boxed{q(x_{t-1} \mid x_t, x_0) = \mathcal{N}(\tilde{\mu}_t, \tilde{\beta}_t I) \quad \text{(Feller)}}$$

| 개념 | 식 | 역할 |
|------|-----|------|
| **Reverse Process** | $p_\theta(x_{t-1} \mid x_t)$ | 학습되는 생성 과정 |
| **True Reverse** | $q(x_{t-1} \mid x_t, x_0)$ | Training target (Bayes) |
| **Score Function** | $s_\theta(x_t,t) = \nabla_{x_t} \log p_\theta$ | 신경망 학습 대상 |
| **Variance Preservation** | $\Sigma$ 는 $\beta_t$ 의 함수 | 수치 안정성 |

## 🤔 생각해볼 문제

### 문제 1 (기초): True Reverse의 Gaussian 성

Forward: $q(x_t \mid x_{t-1})$ 와 $q(x_{t-1} \mid x_0)$ 가 모두 Gaussian 이면,
왜 $q(x_{t-1} \mid x_t, x_0)$ 도 Gaussian 인가?

(힌트: Bayes' rule, Gaussian 의 곱)

<details>
<summary>해설</summary>

Bayes' rule:
$$q(x_{t-1} \mid x_t, x_0) = \frac{q(x_t \mid x_{t-1}) q(x_{t-1} \mid x_0)}{q(x_t \mid x_0)}$$

분자: Gaussian × Gaussian (exponential of quadratic form)
분모: Gaussian (constant w.r.t. $x_{t-1}$)

결과: Gaussian proportional → 정확히 Gaussian

그리고 분모가 normalization 상수이므로, 
분자를 정규화하면 posterior Gaussian.

이것이 DDPM 의 아름다운 점: posterior 의 closed-form 존재.

</details>

### 문제 2 (심화): Score와 노이즈 예측의 동등성

다음을 보이시오:
$$\nabla_{x_t} \log q(x_t \mid x_0) = -\frac{1}{\sqrt{1-\bar{\alpha}_t}} \epsilon$$

여기서 $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$.

<details>
<summary>해설</summary>

$q(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t)I)$

Log density:
$$\log q(x_t \mid x_0) = -\frac{1}{2(1-\bar{\alpha}_t)} \|x_t - \sqrt{\bar{\alpha}_t} x_0\|^2 + \text{const}$$

Gradient:
$$\nabla_{x_t} \log q = -\frac{1}{1-\bar{\alpha}_t} (x_t - \sqrt{\bar{\alpha}_t} x_0)$$

Reparameterization 대입:
$$x_t - \sqrt{\bar{\alpha}_t} x_0 = \sqrt{1-\bar{\alpha}_t} \epsilon$$

따라서:
$$\nabla_{x_t} \log q = -\frac{\sqrt{1-\bar{\alpha}_t} \epsilon}{1-\bar{\alpha}_t} = -\frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}} \quad \checkmark$$

이는 score 가 **noise 를 빼는 방향** 임을 보여줌!

</details>

### 문제 3 (논문 비평): Discrete vs Continuous Reverse

DDPM 은 discrete reverse (1000 steps) 이지만,
Song et al. 의 Score-SDE 는 continuous 이다.

장단점을 논의하시오.

<details>
<summary>해설</summary>

**DDPM (Discrete)**:
- 장점: 구현 간단, fast training, DDIM 으로 sampling 가속화 가능
- 단점: Discretization error, 이론적 엄밀성 떨어짐

**Score-SDE (Continuous)**:
- 장점: 수학적 엄밀성 (SDE theory), Probability flow ODE, Exact likelihood
- 단점: 구현 복잡, numerical ODE solver 필요

**혼용**: Practical 에는 DDPM discrete, Theory 에는 Score-SDE.
실제로 둘은 $T \to \infty$ 극한에서 같음.

</details>

---

<div align="center">

[◀ 이전](./03-forward-closed-form.md) | [📚 README](../README.md) | [다음 ▶](./05-posterior-derivation.md)

</div>
