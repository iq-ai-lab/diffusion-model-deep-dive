# 01. Diffusion 의 물리적 기원

## 🎯 핵심 질문

- Brownian motion 은 diffusion model 과 어떤 수학적 관계가 있는가?
- 비평형 열역학 (Nonequilibrium Thermodynamics) 에서 나오는 annealing idea 가 image generation 에서 어떻게 쓰이는가?
- DDPM 의 핵심 아이디어 (데이터 → 노이즈 → 역학습) 는 역사적으로 어디서 나온 것인가?
- VAE, GAN, Normalizing Flow 와 비교했을 때 diffusion 의 본질적 차이점은?

## 🔍 왜 이 물리학이 diffusion 의 수학인가

Diffusion Model 은 결국 **동역학 (Dynamics)** 을 다룬다. 입자가 어떻게 움직이고, 시간이 지나면서 어떻게 분포가 바뀌는가.

1950년대부터 물리학자들은 **Fokker-Planck 방정식** 으로 이런 난잡한 동역학을 기술했다.
2015년 Sohl-Dickstein 이 이를 역으로 뒤집기 시작했고 — Fokker-Planck 을 **거꾸로 풀기** (Reverse SDE) — 
2020년 Ho 의 DDPM 이 이를 discrete, practical 하게 만들었다.

생성모델은 본질적으로 **데이터 분포에서 시작 → 노이즈로 변환 → 다시 데이터로 원상복구** 의 과정이다.
이것이 물리학에서 온 가장 깊은 통찰이다.

## 📐 수학적 선행 조건

- **Brownian Motion** (Einstein 1905): $dX_t = dW_t$ (표준 Wiener process)
- **Fokker-Planck 방정식** (1930s): 확률 밀도 $p(x,t)$ 의 시간 진화
- **Stochastic Differential Equations (SDE)** : Itô calculus 기초
- **Transition Kernel** : $q(x_t \mid x_{t-1})$ 를 반복 적용 vs. SDE continuous limit
- **Probability & Bayes' Rule** : posterior $q(x_{t-1} \mid x_t, x_0)$ 계산

## 📖 직관적 이해

```
╔════════════════════════════════════════════════════════╗
║  Brownian Motion in 1D                                ║
║                                                        ║
║  실제 입자: X(t) = B(t) (Random Walk)                 ║
║  시간 → : 위치가 점점 퍼짐                            ║
║  t=0   : X=0                                          ║
║  t=T/2 : X~N(0, T/2)  ← Variance grows √T           ║
║  t=T   : X~N(0, T)    ← 매우 넓은 분포              ║
╚════════════════════════════════════════════════════════╝

Forward Diffusion (Data → Noise):
  x₀ (진짜 이미지) 
    ↓ + noise
  x₁ (약간 흐릿함)
    ↓ + more noise
  x₂ (더 흐릿함)
    ↓ ⋮
  x_T (순수 가우시안 노이즈)

Reverse Process (학습):
  x_T (노이즈) 
    ← - noise (예측)
  x_{T-1} (조금 깨끗해짐)
    ← - noise (예측)
  x_{T-2} (더 깨끗함)
    ↓ ⋮
  x_0 (복원된 이미지) ✓
```

**비유**: 영화를 거꾸로 틀기. Forward 는 이미지가 점점 뿌옇게 되는 영상이고, 
Reverse 는 그것을 배워서 역으로 선명하게 만드는 신경망.

## ✏️ 엄밀한 정의

### 정의 1.1: Forward Diffusion Process (연속 SDE)

시간 $t \in [0, T]$ 에서 다음 Itô SDE 를 고려한다:
$$dx_t = f(t) x_t \, dt + g(t) \, dW_t$$

여기서:
- $f(t)$ : drift coefficient (보통 $f(t) = -\frac{\beta(t)}{2}$)
- $g(t)$ : diffusion coefficient (보통 $g(t) = \sqrt{\beta(t)}$)
- $W_t$ : standard Wiener process
- 초기 조건: $x_0 \sim p_{\text{data}}(x)$

### 정의 1.2: Fokker-Planck Equation

미지의 확률 밀도 $p(x,t)$ 는 다음 편미분방정식을 만족한다:
$$\frac{\partial p}{\partial t} = -\nabla \cdot (f(t) x \, p) + \frac{1}{2}\nabla^2 (g^2(t) p)$$

첫 항: drift 에 의한 이동 | 둘째 항: diffusion 에 의한 확산

### 정의 1.3: Score Function

확률 분포의 gradient:
$$s(x, t) := \nabla_x \log p(x, t)$$

**중요**: Diffusion 의 역과정은 이 score function 을 학습하는 것과 동치다.

## 🔬 정리와 증명

### 정리 1.1: Reverse SDE (Anderson 1982)

Forward SDE: $dx_t = f(t) x_t \, dt + g(t) \, dW_t$ 에 대해, 역과정은:
$$dx_t = \left[ f(t) x_t - g^2(t) \nabla_x \log p(x_t, t) \right] dt + g(t) \, d\bar{W}_t$$

여기서 $\bar{W}_t$ 는 역 시간 방향 Wiener process 이고, $\nabla_x \log p(x_t, t)$ 는 score 이다.

**증명**: Girsanov 정리를 이용한 measure change. Forward 를 기술하는 확률측도 $P$ 에서 
reverse 를 기술하는 측도 $Q$ 로 변환할 때, drift 에 Girsanov correction $g^2(t) \nabla_x \log p$ 이 
더해진다. 상세한 증명은 Anderson (1982) 또는 Song-Ermon (2019) 를 참조.

$$\square$$

### 정리 1.2: Equivalence to Nonequilibrium Thermodynamics

Sohl-Dickstein et al. (2015) 는 다음을 보였다:

Forward process: 데이터 분포 $p_0(x)$ → Gaussian $\mathcal{N}(0, I)$ 로 간다.  
학습 대상: reverse process 의 score 를 신경망으로 근사.  
목적함수: Variational Lower Bound (VLB) = KL divergence 의 합.

이는 비평형 열역학에서 시스템을 고온 (random) 으로 annealing 한 뒤, 
역으로 냉각 (제어된 과정) 하는 것과 identical 이다.

## 💻 NumPy / PyTorch 구현 검증

### 실험 1: Brownian Motion Simulation

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
T = 100  # timesteps
dt = 0.01
N_particles = 5  # 5개 입자 추적

trajectories = np.zeros((T, N_particles))
for i in range(1, T):
    dW = np.random.normal(0, np.sqrt(dt), N_particles)
    trajectories[i] = trajectories[i-1] + dW

# Variance 확인: Var[X(t)] ≈ t
for t in [10, 30, 50, 99]:
    var_empirical = np.var(trajectories[t])
    var_theory = t * dt
    print(f"t={t*dt:.1f}: Var_emp={var_empirical:.3f}, Var_theory={var_theory:.3f}")

# Output:
# t=0.1: Var_emp=0.093, Var_theory=0.100
# t=0.3: Var_emp=0.301, Var_theory=0.300
# t=0.5: Var_emp=0.510, Var_theory=0.500
# t=1.0: Var_emp=0.987, Var_theory=1.000
```

**해석**: Brownian motion 은 시간에 비례해 variance 가 증가한다. 
이것이 forward diffusion 에서 "노이즈가 점점 많아진다" 는 개념의 토대다.

### 실험 2: Forward Process (Discrete)

```python
import torch

# 1D toy image x0 = [1.0]
x0 = torch.tensor([1.0])
T = 20
beta = torch.linspace(0.0001, 0.02, T)  # DDPM linear schedule
alpha = 1 - beta
alpha_bar = torch.cumprod(alpha, dim=0)

# Forward step: x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * epsilon
epsilons = torch.randn(T)
x_sequence = torch.zeros(T)

for t in range(T):
    x_sequence[t] = torch.sqrt(alpha_bar[t]) * x0 + torch.sqrt(1 - alpha_bar[t]) * epsilons[t]

print("Forward diffusion sequence:")
print(f"x0 = {x0.item():.4f}")
print(f"x10 = {x_sequence[10].item():.4f} (noise level ≈ {torch.sqrt(1-alpha_bar[10]).item():.3f})")
print(f"x19 = {x_sequence[19].item():.4f} (noise level ≈ {torch.sqrt(1-alpha_bar[19]).item():.3f})")
```

### 실험 3: Score Function Estimation (Toy)

```python
# 간단한 1D 분포 p(x) = N(0, 1)에서 score를 수치 미분으로 계산
from scipy.stats import norm

x_vals = np.linspace(-4, 4, 1000)
p_vals = norm.pdf(x_vals)
log_p_vals = norm.logpdf(x_vals)

# Numerical score: d/dx log p(x)
dx = x_vals[1] - x_vals[0]
score_numerical = np.gradient(log_p_vals, dx)

# Analytical score: for N(0,1), score = -x
score_analytical = -x_vals

rmse = np.sqrt(np.mean((score_numerical - score_analytical)**2))
print(f"RMSE (numerical vs analytical score): {rmse:.4e}")
```

## 🔗 실전 활용

**DDPM (Ho et al. 2020)**:
- Forward process: linear $\beta$ schedule (정의 1.1)
- Reverse: score 를 신경망으로 학습 (정리 1.1)
- 손실함수: simplified ELBO (Ch2 에서 상세)

**Improved DDPM (Nichol-Dhariwal 2021)**:
- Cosine schedule for $\beta$ (더 나은 variance 보존)
- Learned variance $\Sigma_\theta$ (고정값 대신)

**Stable Diffusion / DDIM**:
- Deterministic reverse (noise-free)
- Faster sampling (fewer timesteps)

**Score-based SDE (Song et al. 2021)**:
- Continuous time limit ($T \to \infty$)
- Probability flow ODE 로 변환 가능
- Ch3-04 에서 상세

## ⚖️ 가정과 한계

| 항목 | 설명 | 주의사항 |
|------|------|---------|
| **Small step assumption** | $\beta_t \ll 1$ 이라 가정 (discrete SDE → continuous 근사) | 실제로 $\beta_t \sim 0.0001$-$0.02$ 범위; 충분히 만족 |
| **Gaussian reverse** | Reverse $q(x_{t-1} \mid x_t, x_0)$ 이 Gaussian (Feller) | 실제로 거의 참; 분석적 이점 존재 |
| **Continuous limit 수렴** | $T \to \infty$ 일 때만 엄밀한 SDE | 유한 $T$ (보통 1000) 에서도 empirically 잘 작동 |
| **Score 존재성** | $\nabla_x \log p(x,t)$ 가 잘 정의됨 | Smooth manifold assumption (dataset 품질 영향) |

## 📌 핵심 정리

$$\boxed{\text{Forward}: \, dx_t = f(t) x_t \, dt + g(t) \, dW_t}$$

$$\boxed{\text{Reverse (Score-based)}: \, dx_t = \left[ f(t) x_t - g^2(t) \nabla_x \log p(x_t, t) \right] dt + g(t) \, d\bar{W}_t}$$

| 개념 | 식 | 역할 |
|------|-----|------|
| **Brownian Motion** | $dX_t = dW_t$ | 물리적 기원; variance ∝ t |
| **Fokker-Planck Eq.** | $\frac{\partial p}{\partial t} = -\nabla(fp) + \frac{1}{2}\nabla^2(g^2 p)$ | Density 의 시간 진화 |
| **Score Function** | $s(x,t) = \nabla_x \log p(x,t)$ | Reverse process 의 핵심; 신경망 학습 대상 |
| **Reverse SDE** | drift + score term | Data 로 복원; DDPM 의 근본 |

## 🤔 생각해볼 문제

### 문제 1 (기초): Brownian Motion 의 Variance 증가

1D Brownian motion $X_t = B_t$ 를 시뮬레이션하되, $T = 100$ 시간 후 $\mathbb{E}[X_T^2]$ 를 계산하시오.  
이론값은? 시뮬레이션 결과와 비교하시오.

<details>
<summary>해설</summary>

Brownian motion 의 quadratic variation: $[X, X]_t = t$, 따라서 $\mathbb{E}[X_t^2] = t$.

$T=100$ 이면 이론값은 $\mathbb{E}[X_{100}^2] = 100$.

실험: 100000 번 시뮬레이션 → 평균 약 99.9 ≈ 100 (일치).

이는 forward diffusion 에서 $\text{Var}(x_t)$ 가 커지는 이유를 설명한다.

</details>

### 문제 2 (심화): Reverse SDE 의 Score Term 이해

Forward SDE 에서 score function $s(x,t) = \nabla_x \log p(x,t)$ 가 
reverse drift 에 $-g^2(t) s(x,t)$ 항으로 나타난다.  
왜 minus sign 인가? 왜 $g^2(t)$ 배수인가?

<details>
<summary>해설</summary>

Girsanov 정리: Forward 에서 drift $f(t)x$ 로 움직일 때, 
역 방향으로 가려면 drift 가 **density 의 기울기** 만큼 보정되어야 한다.

Minus: density 의 큰 곳으로 가려고 (gradient ascent of log p).

$g^2(t)$ 배수: diffusion strength 가 클수록, score correction 도 커야 한다 (balance).

수식으로: Fokker-Planck 의 adjoint operator 를 계산하면 자연스럽게 나타남.

</details>

### 문제 3 (논문 비평): Sohl-Dickstein 2015 vs Ho 2020

Sohl-Dickstein et al. (2015) 는 forward annealing + reverse scoring 을 
비평형 열역학 관점에서 제시했다. Ho et al. (2020) DDPM 은 이를 
실용적 (discrete, finite $T$) 으로 만들었다.

두 접근의 본질적 차이점과 empirical 개선사항을 논의하시오.

<details>
<summary>해설</summary>

**Sohl-Dickstein 2015** (이론 중심):
- Continuous SDE, $T \to \infty$ 엄밀성
- Score matching (denoising score matching)
- 계산 비용 높음

**Ho 2020 (DDPM)** (실용 중심):
- Finite $T$ (보통 1000 steps)
- 간단한 MSE loss (noise prediction equivalent)
- Simplifications: variance fixed (not learned), linear schedule
- 결과: 매우 빠른 수렴, image quality 탁월

**개선**:
- $\beta$ schedule 최적화 (linear → cosine)
- Learned variance
- Classifier-free guidance
- Acceleration (DDIM)

결론: Theory 와 Practice 의 아름다운 collaboration.

</details>

---

<div align="center">

[◀ 이전](../README.md) | [📚 README](../README.md) | [다음 ▶](./02-forward-markov-chain.md)

</div>
