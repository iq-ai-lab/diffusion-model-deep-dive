# 5. VP-SDE와 VE-SDE: 구체적 모델들

## 🎯 핵심 질문
VP-SDE (Variance-Preserving)와 VE-SDE (Variance-Exploding)는 무엇이고, 각각 어떤 구체적 diffusion model과 대응되는가? Log-likelihood 측면에서는 어떻게 비교되는가?

## 🔍 왜 두 가지 SDE인가?

Diffusion model 설계에는 두 가지 주요 철학이 있다:
1. **VP-SDE** (DDPM 스타일): 분산을 유지하면서 신호 강도를 점진적으로 줄임
2. **VE-SDE** (NCSN 스타일): drift 없이 노이즈만 누적, 신호-노이즈 비율 변화

두 접근은 생성 성능에서 경쟁 관계이며, 각각 장단점이 있다.

## 📐 수학적 선행 조건
- Ito SDE의 해의 성질: $\mathbb{E}[\|X_t\|^2]$
- Stochastic calculus: quadratic variation
- Log-likelihood gradient: score function과의 관계
- Jensen's inequality, trace operator

## 📖 직관적 이해

**VP-SDE 관점**: 신호가 점진적으로 약해지되 총 분산 (신호 + 노이즈)은 일정
- $x_t = \sqrt{\alpha_t} x_0 + \sqrt{1-\alpha_t} \epsilon$의 연속 극한
- $\mathbb{E}[\|x_t\|^2] = 1$ (정규화됨)

**VE-SDE 관점**: 신호는 변하지 않고 노이즈만 자라남
- $x_t = x_0 + \sigma(t) \epsilon$의 연속 극한
- $\mathbb{E}[\|x_t\|^2] = \|x_0\|^2 + \sigma(t)^2$ (증가)

고차원에서는 어느 것이 더 나은가? 경험적으로는 비슷하지만 이론적 분석은 다름.

## ✏️ 엄밀한 정의

**정의 3.11 (VP-SDE)**  
$$dx = -\frac{1}{2}\beta(t) x \, dt + \sqrt{\beta(t)} \, dW_t$$
여기서 $\beta(t) \geq 0$는 noise schedule. 적분하면:
$$x_t = e^{-\frac{1}{2}\int_0^t \beta(s)ds} x_0 + \sqrt{1 - e^{-\int_0^t \beta(s)ds}} \cdot \epsilon$$

$\bar{\alpha}(t) := e^{-\int_0^t \beta(s)ds}$로 정의하면, $x_t = \sqrt{\bar{\alpha}(t)} x_0 + \sqrt{1-\bar{\alpha}(t)} \epsilon$.

**정의 3.12 (VE-SDE)**  
$$dx = \sqrt{\frac{d\sigma^2(t)}{dt}} \, dW_t$$
(drift가 0) 적분하면:
$$x_t = x_0 + \sigma(t) \cdot \epsilon$$

**정의 3.13 (Sub-VP-SDE, 보간)**  
VP와 VE의 중간 형태:
$$dx = -\frac{1}{2}\beta(t) x \, dt + \sqrt{(1 - e^{-2\int_0^t \beta(s)ds}) \beta(t)} \, dW_t$$

## 🔬 정리와 증명

**정리 3.10 (VP-SDE ↔ DDPM)**  
VP-SDE $dx = -\frac{1}{2}\beta(t)xdt + \sqrt{\beta(t)}dW$의 이산화 (시간 이산 $0 = t_0 < t_1 < \cdots < t_N = T$)는 DDPM의 반복식과 정확히 일치한다:
$$x_{n+1} = \sqrt{1 - \beta_n} x_n + \sqrt{\beta_n} \epsilon_n$$
(단, $\beta_n = \int_{t_n}^{t_{n+1}} \beta(t) dt$).

증명: Euler 이산화 $x_{n+1} \approx x_n + (-\frac{1}{2}\beta_n x_n)(t_{n+1}-t_n) + \sqrt{\beta_n(t_{n+1}-t_n)} \xi_n$ 형태에서, 시간 간격이 작으면 drift 항이 1차 근사가 되고, diffusion term이 정확히 $\sqrt{\beta_n}$이 되어 일치한다. $\square$

**정리 3.11 (VE-SDE ↔ NCSN)**  
VE-SDE $dx = \sqrt{d\sigma^2(t)/dt} dW$의 이산화 (σ-steps $\sigma_1 > \sigma_2 > \cdots > \sigma_L$)는 NCSN의 annealed Langevin과 대응된다:
$$x_{n+1} = x_n + \eta \cdot (-x_n/\sigma_n) + \sqrt{2\eta} \epsilon_n$$
(여기서 $\eta$는 step size, score $s = -x/\sigma$ at zero-mean Gaussian).

증명: VE-SDE의 score function은 $\nabla_x \log p_t(x|x_0) = -(x-x_0)/\sigma(t)^2 \approx -x/\sigma(t)$ (zero-mean 가정). Annealed Langevin이 SDE의 discretization과 일치. $\square$

**정리 3.12 (Log-Likelihood: VP vs VE)**  
Probability flow ODE를 따라 $0 \to T$로 이동할 때:
$$\log p_0(x_0) = \log p_T(x_T) - \int_0^T \text{tr}(J_f - \frac{1}{2}g^2 H_{\log p})ds$$

VP-SDE에서는 $x_t$의 분산이 bounded (1 근처)이므로, log-likelihood 적분이 상대적으로 안정적.  
VE-SDE에서는 $x_t$의 분산이 증가하므로, high-variance 영역에서 numerical instability 가능.

증명 스케치: Jacobian trace와 Hessian trace는 각각 drift와 diffusion에 의존. VP는 제약된 영역에서, VE는 무제약 영역에서 작동하므로 수치적 특성이 다름. $\square$

## 💻 NumPy / PyTorch 구현 검증

### 실험 1: VP-SDE와 VE-SDE의 분산 진화
```python
import torch
import numpy as np
import matplotlib.pyplot as plt

# VP-SDE
def vp_sde_trajectory(T=1.0, num_steps=1000):
    """VP-SDE: variance is preserved (roughly 1)"""
    def beta(t):
        # Linear schedule
        return 0.1 + 0.9 * t  # ranges 0.1 to 1.0
    
    times = np.linspace(0, T, num_steps)
    x_var = []
    
    for t in times:
        integral = np.trapz([beta(s) for s in times[:int(t*num_steps)]], times[:int(t*num_steps)])
        alpha_bar = np.exp(-0.5 * integral)
        var_x = alpha_bar**2 + (1 - alpha_bar**2)  # E[||x_t||^2] with unit norm x_0
        x_var.append(var_x)
    
    return times, np.array(x_var)

# VE-SDE
def ve_sde_trajectory(T=1.0, num_steps=1000):
    """VE-SDE: variance explodes"""
    def sigma(t):
        # Linear schedule
        return 0.1 * np.sqrt(1 + 99 * t)
    
    times = np.linspace(0, T, num_steps)
    x_var = np.array([1 + sigma(t)**2 for t in times])
    
    return times, x_var

# 시뮬레이션
times_vp, var_vp = vp_sde_trajectory()
times_ve, var_ve = ve_sde_trajectory()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(times_vp, var_vp, 'b-', linewidth=2, label='VP-SDE')
axes[0].axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
axes[0].set_xlabel('Time t')
axes[0].set_ylabel('E[||x_t||²]')
axes[0].set_title('VP-SDE: Variance Preserving')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].plot(times_ve, var_ve, 'r-', linewidth=2, label='VE-SDE')
axes[1].set_xlabel('Time t')
axes[1].set_ylabel('E[||x_t||²]')
axes[1].set_title('VE-SDE: Variance Exploding')
axes[1].grid(True, alpha=0.3)
axes[1].set_yscale('log')
axes[1].legend()

plt.tight_layout()
plt.show()

print(f"VP-SDE final variance: {var_vp[-1]:.4f}")
print(f"VE-SDE final variance: {var_ve[-1]:.4f}")
```

### 실험 2: DDPM과 VP-SDE의 등가성
```python
# DDPM 스케줄
def ddpm_schedule(num_steps=1000):
    s = 0.008
    steps = torch.arange(num_steps + 1, dtype=torch.float64) / num_steps
    alphas = torch.cos(((steps + s) / (1 + s) * np.pi * 0.5) ** 2)
    alphas = alphas / alphas[0]
    betas = 1 - alphas[1:] / alphas[:-1]
    alphas_cumprod = torch.cumprod(alphas[:-1], dim=0)
    return betas, alphas_cumprod

# VP-SDE equivalent
def vp_sde_ddpm_equivalent(betas):
    """VP-SDE를 DDPM 스타일로 표현"""
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    # VP-SDE: alpha_bar(t) = exp(-0.5 * int beta(s) ds)
    # DDPM: alpha_bar_t = prod alpha_i
    # 근사 관계: exp(sum log(1-beta)) ≈ prod (1-beta) for small beta
    
    return alphas_cumprod

# 검증
betas, alphas_cumprod_ddpm = ddpm_schedule(1000)
alphas_cumprod_vp = vp_sde_ddpm_equivalent(betas)

# 처음 100 스텝에서 비교
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(alphas_cumprod_ddpm[:100], label='DDPM alpha_bar', linewidth=2)
ax.plot(alphas_cumprod_vp[:100], label='VP-SDE approximation', linestyle='--', linewidth=2)
ax.set_xlabel('Step')
ax.set_ylabel('ᾱₜ')
ax.set_title('DDPM vs VP-SDE Schedule')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()

print(f"Max difference in first 100 steps: {(alphas_cumprod_ddpm[:100] - alphas_cumprod_vp[:100]).abs().max():.6f}")
```

### 실험 3: Log-Likelihood 추적
```python
def compute_likelihood_lower_bound(score_model, x_data, T=1.0, num_steps=100):
    """
    Probability flow ODE를 따라 likelihood 하한 계산
    log p_0(x) ≈ log p_T(x_T) - integral of trace terms
    """
    x = x_data.clone()
    times = torch.linspace(0, T, num_steps)
    
    log_likelihood = torch.zeros(x.shape[0])
    
    for i in range(len(times) - 1):
        t = times[i]
        dt = times[i+1] - t
        
        with torch.no_grad():
            score = score_model(x, torch.full((x.shape[0],), t.item()))
        
        # Simplified trace computation
        # tr(f_jacobian - 0.5*g^2*log_p_hessian)
        # For VP-SDE: f = -0.5*beta*x, g = sqrt(beta)
        # For VE-SDE: f = 0, g = sqrt(d sigma^2/dt)
        
        beta_t = 0.1 + 0.9 * t  # VP schedule
        trace_term = -0.5 * beta_t * x.shape[1] - 0.5 * beta_t * torch.sum(score**2, dim=1)
        
        log_likelihood += trace_term * dt
    
    # p(x_T) = N(0, I) under VP-SDE at T=1
    final_logp = -0.5 * torch.sum(x**2, dim=1)
    
    return final_logp - log_likelihood

# 테스트 (간단한 모델)
score_model = torch.nn.Sequential(
    torch.nn.Linear(2, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 2)
)

x_test = torch.randn(10, 2)
ll_estimate = compute_likelihood_lower_bound(score_model, x_test, T=1.0)
print(f"Log-likelihood estimates (mean): {ll_estimate.mean():.4f}")
print(f"Log-likelihood estimates (std): {ll_estimate.std():.4f}")
```

## 🔗 실전 활용

**모델 선택 가이드**
| 특성 | VP-SDE | VE-SDE |
|------|--------|--------|
| 분산 안정성 | 우수 | 폭발적 증가 |
| Low-density 영역 학습 | 약함 | 강함 |
| Likelihood 계산 | 안정적 | 수치불안정 |
| 고해상도 이미지 | DDPM 스타일 우수 | NCSN 스타일도 가능 |

**구현 가이드**
- VP-SDE: DDPM 코드 재사용, 상대적으로 간단
- VE-SDE: 큰 σ 범위 처리, multi-scale noise 필수
- Sub-VP: VP와 VE의 trade-off

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Linear/cosine schedule | 최적 schedule은 문제 의존적 |
| Score 정확 추정 | 실제로는 오차 있음, 누적 |
| 연속 시간 극한 | 이산화 오차 존재 |
| SDE 형태 고정 | 다른 SDE 설계 가능 |

## 📌 핵심 정리

**VP-SDE는 DDPM의 연속 시간 극한이며, 분산을 일정하게 유지하면서 신호를 약화시킨다. VE-SDE는 NCSN의 극한이며, 노이즈를 누적하되 신호는 유지한다. Sub-VP는 그 사이의 보간이다. Log-likelihood 측면에서는 VP가 더 안정적이지만, 저확률 영역 생성에서는 VE (또는 multi-scale 전략)가 유리할 수 있다.**

## 🤔 생각해볼 문제

### 문제 1
왜 VP-SDE의 drift 항에 $-\frac{1}{2}\beta(t)x$가 들어가는가? Coefficient가 1이 아니라 1/2인 이유는?

<details>
<summary>해설</summary>

Ito SDE에서 drift와 diffusion의 상호작용을 고려해야 한다. Ito lemma를 적용하면 $d(\log p_t) = \cdots$ 형태가 되는데, $-\frac{1}{2}$ factor는 quadratic variation에서 나온다. 정확히는 $\mathbb{E}[\|x_t\|^2]$를 보존하기 위한 필요조건이다.
</details>

### 문제 2
VE-SDE에서 drift가 0인 것이 의미하는 바는? 이것이 NCSN의 설계와 어떻게 연결되는가?

<details>
<summary>해설</summary>

Drift가 없다는 것은 신호 성분이 변하지 않는다는 의미. 따라서 $x_0$와 $x_t$의 신호-노이즈 비율이 시간에 따라 변함. NCSN은 이 특성을 이용해 각 σ 스케일에서 별도의 score를 학습하고, annealed sampling으로 다양한 스케일을 순회한다.
</details>

### 문제 3
Sub-VP-SDE는 VP와 VE의 어떤 장점을 결합하는가?

<details>
<summary>해설</summary>

Sub-VP는 diffusion 강도를 줄여서 VP의 분산 안정성과 VE의 저확률 영역 탐색을 모두 달성하려는 시도다. 실제로는 empirical하게 어느 것이 더 좋은지는 task와 hyperparameter에 따라 달라진다. Song et al. (2021)에서 비교 연구를 진행했다.
</details>

---

<div align="center">

[◀ 이전](./04-score-sde.md) | [📚 README](../README.md) | [다음 ▶](../ch4-ddim/01-ddim-motivation.md)

</div>
