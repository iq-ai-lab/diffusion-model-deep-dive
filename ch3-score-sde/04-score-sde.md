# 4. Score-SDE: 확률미분방정식 통합 이론

## 🎯 핵심 질문
Song (2021)의 Score-SDE는 DDPM, NCSN, DDIM을 어떻게 통합하는가? Anderson (1982)의 reverse-time SDE를 이용한 generation은 무엇인가?

## 🔍 왜 SDE인가?

이산 diffusion step들 (DDPM의 $T$ 스텝, NCSN의 $L$ σ 스케일)을 **연속시간** 확률미분방정식(SDE)으로 보면, 모든 이산 알고리즘이 특수 경우가 된다. 이를 통해 다양한 생성 경로를 하나의 프레임워크로 이해할 수 있다.

## 📐 수학적 선행 조건
- Itô SDE: $dx = f(x,t)dt + g(t)dW$
- Fokker-Planck 방정식: SDE의 density evolution
- Reverse-time SDE (Anderson 1982)
- Girsanov 정리: measure change
- Probability flow ODE: $dx/dt = f - \frac{1}{2}g^2 \nabla \log p_t$

## 📖 직각적 이해

데이터 분포에서 Gaussian noise로 전환하는 과정(forward)을 연속 SDE로 모델링하면:
$$dx = f(x,t)dt + g(t)dW$$

역방향(backward)에서 noise를 제거하려면, Anderson의 정리를 사용해 reverse-time SDE를 구성한다:
$$dx = [f(x,t) - g(t)^2 \nabla_x \log p_t(x)]dt + g(t)d\bar{W}$$

여기서 $\nabla_x \log p_t(x) = s_\theta(x,t)$는 학습한 score다.

## ✏️ 엄밀한 정의

**정의 3.8 (Forward SDE)**  
시간 $t \in [0, T]$에 대해
$$dx = f(x,t)dt + g(t)dW_t$$
를 forward SDE라 한다. 여기서:
- $f(x,t)$: drift coefficient
- $g(t)$: diffusion coefficient
- $W_t$: standard Brownian motion

**정의 3.9 (Reverse-time SDE)**  
Forward SDE의 reverse-time version은
$$dx = [f(x,t) - g(t)^2 \nabla_x \log p_t(x)]dt + g(t)d\bar{W}_t$$
여기서 $d\bar{W}_t$는 역방향 Brownian motion.

**정의 3.10 (Probability Flow ODE)**  
같은 marginal distribution path $\{p_t\}$를 따르는 ODE는
$$\frac{dx}{dt} = f(x,t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x)$$

## 🔬 정리와 증명

**정리 3.7 (Anderson의 Reverse-time SDE)**  
Forward SDE $dx = f(x,t)dt + g(t)dW$의 marginal distribution이 $\{p_t(x)\}$일 때, reverse-time SDE는
$$dx = [f(x,t) - g(t)^2 \nabla_x \log p_t(x)]dt + g(t)d\bar{W}_t$$
의 marginal distribution도 $\{p_T - t(x)\}$이다 (시간 역순).

증명 스케치 (Itô 미분을 이용):
- Forward: $dx = f dt + g dW$
- 시간을 역방향으로 ($s = T - t$) 치환하면, Itô lemma에 의해 reverse drift는 원래 drift에서 diffusion term의 그래디언트를 빼준 형태가 된다:
$$\text{Reverse drift} = f(x,T-s) - g(T-s)^2 \nabla_x \log p_{T-s}(x)$$
이것이 바로 정의 3.9의 형태다. $\square$

**정리 3.8 (Probability Flow ODE의 고유성)**  
Reverse-time SDE와 probability flow ODE는 동일한 일계 moment를 생성한다 (같은 marginal path). ODE는 확정적(deterministic)이므로 likelihood 계산이 용이하다.

증명: Probability flow ODE $dx/dt = f - \frac{1}{2}g^2 \nabla\log p_t$는 reverse-time SDE에서 diffusion term을 제거한 것이므로, expected trajectory는 동일하고 variance만 다르다. Log-likelihood는 시간 적분 $\int_0^T \text{trace}(J_f - \frac{1}{2}g^2 H_{\log p})dt$로 계산된다. $\square$

**정리 3.9 (DDPM·NCSN·DDIM의 SDE 해석)**  
- DDPM: VP-SDE (variance-preserving)의 이산화
- NCSN: VE-SDE (variance-exploding)의 이산화
- DDIM: ODE sampler로 reverse-time SDE의 특수 경우

## 💻 NumPy / PyTorch 구현 검증

### 실험 1: Forward SDE 시뮬레이션
```python
import torch
import numpy as np
import matplotlib.pyplot as plt

def vp_sde_forward(x0, t, alphas_cumprod):
    """VP-SDE: dx = -0.5*beta(t)*x dt + sqrt(beta(t)) dW"""
    # alphas_cumprod: alpha_bar(t) = \prod_0^t (1 - beta_s) ds
    alpha_bar_t = alphas_cumprod[t]
    
    # Forward formula: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
    epsilon = torch.randn_like(x0)
    x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * epsilon
    return x_t, epsilon

def ve_sde_forward(x0, t, sigma_schedule):
    """VE-SDE: dx = 0 dt + d(sigma(t)) dW (no drift)"""
    sigma_t = sigma_schedule[t]
    
    # VE: x_t = x_0 + sigma_t * epsilon
    epsilon = torch.randn_like(x0)
    x_t = x0 + sigma_t * epsilon
    return x_t, epsilon

# 시뮬레이션
x0 = torch.randn(100, 2)
T = 100

# VP-SDE
alphas = torch.linspace(0.9999, 0.0001, T)
alphas_cumprod = torch.cumprod(alphas, dim=0)

vp_trajectory = []
for t in range(T):
    x_t, _ = vp_sde_forward(x0, t, alphas_cumprod)
    vp_trajectory.append(x_t.mean(dim=0).numpy())

vp_trajectory = np.array(vp_trajectory)

# VE-SDE
sigma_schedule = torch.linspace(0.1, 10.0, T)
ve_trajectory = []
for t in range(T):
    x_t, _ = ve_sde_forward(x0, t, sigma_schedule)
    ve_trajectory.append(x_t.mean(dim=0).numpy())

ve_trajectory = np.array(ve_trajectory)

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(vp_trajectory[:, 0], label='VP-SDE', linewidth=2)
axes[0].set_xlabel('Time t')
axes[0].set_ylabel('Mean x[0]')
axes[0].set_title('VP-SDE Forward Diffusion')
axes[0].grid(True, alpha=0.3)

axes[1].plot(ve_trajectory[:, 0], label='VE-SDE', linewidth=2)
axes[1].set_xlabel('Time t')
axes[1].set_ylabel('Mean x[0]')
axes[1].set_title('VE-SDE Forward Diffusion')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 실험 2: Reverse-time SDE 샘플링
```python
class ScoreSDEModel(torch.nn.Module):
    """Time-dependent score network for SDE"""
    def __init__(self, input_dim=2):
        super().__init__()
        self.time_embed = torch.nn.Sequential(
            torch.nn.Linear(1, 64),
            torch.nn.Softplus(),
            torch.nn.Linear(64, 64)
        )
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim + 64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, input_dim)
        )
    
    def forward(self, x, t):
        # t: normalized time in [0, 1]
        t_emb = self.time_embed(t.unsqueeze(-1))
        x_aug = torch.cat([x, t_emb], dim=1)
        return self.net(x_aug)

def reverse_sde_sampling(model, T=100, dt=0.01):
    """
    Reverse-time SDE sampling
    x_t_new = x_t + [f(t) - 0.5*g(t)^2 * score]*dt + sqrt(g(t)^2) * dW
    """
    x = torch.randn(16, 2)
    trajectory = [x.clone()]
    
    times = torch.linspace(1.0, 0.0, T)
    
    for i in range(len(times)-1):
        t = times[i]
        t_tensor = torch.full((x.shape[0],), t)
        
        # Score 추정
        score = model(x, t_tensor)
        
        # Reverse-time SDE (간단한 버전, VP-SDE)
        # Drift: -beta(t) * x + beta(t) * score
        dt_step = times[i+1] - times[i]
        beta_t = 2 * (1 - t)  # 예시
        
        drift = -0.5 * beta_t * x + 0.5 * beta_t * score
        diffusion = torch.sqrt(torch.tensor(beta_t))
        
        dW = torch.randn_like(x)
        x = x + drift * dt_step + diffusion * torch.sqrt(-dt_step) * dW
        trajectory.append(x.clone())
    
    return torch.stack(trajectory)

# 훈련 및 샘플링
model = ScoreSDEModel()
samples = reverse_sde_sampling(model, T=100)
print(f"Generated samples shape: {samples.shape}")
```

### 실험 3: Probability Flow ODE
```python
def probability_flow_ode(model, x0_shape=(16, 2), num_steps=100):
    """
    Probability flow ODE: dx/dt = f(x,t) - 0.5*g(t)^2 * score(x,t)
    (ODE solver 사용)
    """
    from scipy.integrate import odeint
    
    def ode_func(x_flat, t):
        x = torch.tensor(x_flat, dtype=torch.float32).reshape(x0_shape)
        t_tensor = torch.full((x.shape[0],), t)
        
        # Score 추정
        with torch.no_grad():
            score = model(x, t_tensor)
        
        # ODE velocity: f(t) - 0.5*g(t)^2*score
        beta_t = 2 * (1 - t)
        velocity = -0.5 * beta_t * x + 0.5 * beta_t * score
        
        return velocity.numpy().flatten()
    
    # 초기 조건
    x0 = torch.randn(x0_shape)
    t_span = np.linspace(1.0, 0.0, num_steps)
    
    # ODE 풀이 (간단한 예시)
    # 실제로는 더 정교한 ODE solver 필요 (e.g., scipy.integrate.solve_ivp)
    x = x0.clone()
    for i in range(len(t_span)-1):
        t = t_span[i]
        dt = t_span[i+1] - t
        
        t_tensor = torch.full((x.shape[0],), t)
        with torch.no_grad():
            score = model(x, t_tensor)
        
        beta_t = 2 * (1 - t)
        velocity = -0.5 * beta_t * x + 0.5 * beta_t * score
        x = x + velocity * dt
    
    return x

# ODE 샘플링 실행
ode_samples = probability_flow_ode(model, num_steps=100)
print(f"ODE samples shape: {ode_samples.shape}")
```

## 🔗 실전 활용

**Score-SDE 프레임워크의 강점**
1. **통합 이론**: DDPM, NCSN, DDIM을 한 프레임워크로 설명
2. **Likelihood 계산**: Probability flow ODE로 exact log-likelihood 가능
3. **샘플링 유연성**: SDE sampler (stochastic)와 ODE sampler (deterministic) 선택
4. **조건화 확장**: Classifier guidance, 특정 속성 제어 등

**대규모 생성 모델**
- CIFAR-10, CelebA, ImageNet에서 최고 FID/IS 달성
- Class-conditional generation 가능

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| SDE의 drift/diffusion 알려짐 | 실제로는 hyperparameter tuning 필요 |
| Score를 정확히 추정 | 학습된 score의 오차 누적 |
| 시간 이산화 ($dt$ 작음) | 큰 dt에서 수치 오차 증가 |
| Ergodicity (reverse-time) | 병목 영역에서 모드 전환 여전히 어려움 |

## 📌 핵심 정리

**Score-SDE 프레임워크는 forward SDE $dx = f(x,t)dt + g(t)dW$를 정의하고, Anderson (1982)의 reverse-time SDE $dx = [f - g^2 \nabla \log p_t]dt + g d\bar{W}$로 생성한다. Probability flow ODE는 동일한 분포 경로를 따르면서도 결정적이므로 likelihood 계산이 가능하다. DDPM, NCSN, DDIM은 모두 이 SDE의 특수한 이산화 방식이다.**

## 🤔 생각해볼 문제

### 문제 1
Reverse-time SDE에서 drift 항 $f(x,t) - g(t)^2 \nabla \log p_t(x)$는 무엇을 의미하는가? 각 성분의 역할은?

<details>
<summary>해설</summary>

$f(x,t)$는 original forward drift를 반대로 하는 항. $g(t)^2 \nabla \log p_t(x)$는 score-induced drift로, 고확률 영역으로 끌어당기는 "potential gradient" 역할을 한다. 합쳐지면 noise에서 data로 이동하는 궤적이 형성된다.
</details>

### 문제 2
Probability flow ODE와 reverse-time SDE는 같은 분포를 생성하는가, 같은 개별 trajectory를 생성하는가?

<details>
<summary>해설</summary>

**같은 분포**, **다른 개별 trajectory**. ODE는 diffusion term을 제거했으므로 deterministic. 하지만 평균적으로는 같은 분포의 샘플들을 생성한다. ODE가 더 빠르고 likelihood 계산도 가능하므로 실전에서 자주 사용.
</details>

### 문제 3
VP-SDE와 VE-SDE에서 score 추정의 난이도는 다른가? 왜?

<details>
<summary>해설</summary>

VE-SDE는 drift가 없어 high-σ 영역에서 더 균등한 분포. VP-SDE는 drift 때문에 분포가 원점 주변에 축약됨. 결과적으로 VE에서는 low-density tail 영역의 score 추정이 더 어렵고 (NCSN이 강조하는 부분), VP는 high-frequency 성분이 감소해 Fourier 관점에서 다름.
</details>

---

<div align="center">

[◀ 이전](./03-ncsn.md) | [📚 README](../README.md) | [다음 ▶](./05-vp-ve-sde.md)

</div>
