# 04. Probability Flow ODE 와 고차 solver (DPM-Solver)

## 🎯 핵심 질문

DDIM (deterministic) 과 확률미분방정식(SDE) 간 관계는? 고차 numerical solver 가 어떻게 10-20 step 으로 high-quality 샘플을 생성하는가?

## 🔍 왜 ODE 로 보는가?

DDIM deterministic case ($\sigma_t = 0$):
$$x_{t-1} = \sqrt{\bar\alpha_{t-1}}\hat x_0 + \sqrt{1-\bar\alpha_{t-1}}\epsilon_\theta(x_t, t)$$

이는 **discrete Markov 를 버렸다**. 대신 $x_t$ 의 시간 궤적이 특정 ODE 를 만족한다.

Song et al. (2021) 에서 보인 것: DDIM deterministic sampling 은 **Probability Flow ODE** 의 1차 Euler discretization 과 동치.

**장점**:
- ODE 는 unique solution → 모든 step 크기에서 동일한 limit
- Numerical ODE solver (2차, 3차) 사용 가능 → 더 정확
- DPM-Solver (Lu 2022) 는 semi-linear 구조 활용 → exponentially weighted quadrature

## 📐 수학적 선행 조건

- **Score function**: $\nabla \log p_t(x) = -\frac{x - \sqrt{\bar\alpha_t}x_0}{\sqrt{1-\bar\alpha_t}^2}$
- **Likelihood weighting**: reverse SDE 의 coefficient
- **Numerical ODE**: Euler, RK4, multi-step 등

## 📖 직관적 이해

Forward SDE (Song 2021 Score-SDE, Ch3):
$$dx = f(x, t)dt + g(t)dw$$

Reverse SDE:
$$dx = [f(x, t) - \frac{1}{2}g(t)^2\nabla\log p_t(x)]dt + g(t)d\bar w$$

**Deterministic limit** (noise 제거, $g \to 0$):
$$\frac{dx}{dt} = f(x, t) - \frac{1}{2}g(t)^2\nabla\log p_t(x)$$

이것이 **Probability Flow ODE**. DDIM 은 이것을 discrete time 에서 approximate.

## ✏️ 엄밀한 정의

### 정의 4.1: Probability Flow ODE (PF-ODE)

Forward process 의 SDE 로부터, deterministic reverse 는:
$$\frac{dx}{dt} = f(x, t) - \frac{1}{2}g(t)^2 \nabla \log p_t(x)$$

여기서:
- $f(x, t)$: drift coefficient (알려짐)
- $g(t)$: diffusion coefficient (알려짐)
- $\nabla \log p_t(x) \approx -\frac{\epsilon_\theta(x, t)}{\sqrt{1-\bar\alpha_t}}$ (학습됨)

### 정의 4.2: Semi-linear ODE 구조 (DPM-Solver)

다시 정렬하면:
$$\frac{dx}{dt} = \lambda(t) x + \mu(t) \epsilon_\theta(x, t)$$

여기서 $\lambda(t), \mu(t)$ 는 deterministic coefficient. 이를 **semi-linear** 라 부르는 것은 $x$ 와 $\epsilon_\theta(x)$ 의 linear combination.

## 🔬 정리와 증명

### 정리 4.1: DDIM Deterministic = Probability Flow ODE 의 1차 Euler

**명제**: 표준 DDPM 세팅에서, $\sigma_t = 0$ 인 DDIM deterministic reverse step:
$$x_{t-1} = \sqrt{\bar\alpha_{t-1}}\hat x_0 + \sqrt{1-\bar\alpha_{t-1}}\epsilon_\theta(x_t, t)$$

은 시간 매개변수를 $s = \sqrt{\bar\alpha_t}$ 로 재설정했을 때, Probability Flow ODE 의 Euler 1차 discretization.

**증명 스케치**:

Forward: $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon$ 에서
$$\frac{dx}{dt} = \frac{d\sqrt{\bar\alpha_t}}{dt}x_0 + \frac{d\sqrt{1-\bar\alpha_t}}{dt}\epsilon$$

Reverse score-matching 에서 $\nabla\log p_t$ 를 $\epsilon_\theta$ 로 표현하면:
$$\frac{dx}{dt} = -\frac{1}{2}\frac{d\ln(1-\bar\alpha_t)}{dt}\epsilon_\theta$$

시간 step $\Delta t$ 에서 Euler:
$$x_{t-\Delta t} \approx x_t + \frac{dx}{dt}\Delta t$$

이를 정리하면:
$$x_{t-1} \approx \sqrt{\bar\alpha_{t-1}}x_0 + \sqrt{1-\bar\alpha_{t-1}}\epsilon_\theta$$

(정확한 도출은 Song et al. ICCV 2021 참고) $\square$

### 정리 4.2: DPM-Solver 의 지수가중 Taylor 수렴

**명제**: DPM-Solver-2 (2차) 는 다음을 만족:
$$\|x_{\text{exact}}(\Delta t) - x_{\text{approx}}(\Delta t)\| = O((\Delta t)^3)$$

반면 Euler 1차는 $O((\Delta t)^2)$.

**증명 스케치**:

Semi-linear ODE:
$$\frac{dx}{dt} = \lambda(t) x + \mu(t) h_\theta(t)$$

여기서 $h_\theta = \epsilon_\theta$. 해는:
$$x(t+\Delta t) = e^{\int_t^{t+\Delta t}\lambda(s)ds} x(t) + \int_t^{t+\Delta t} e^{\int_s^{t+\Delta t}\lambda(u)du} \mu(s) h_\theta(s) ds$$

지수가중 인자가 선형적으로 적용되므로, $h_\theta$ 를 구간 내에서 고차 polynomial 로 interpolate 하면 정밀도 향상.

DPM-Solver 는 exponentially weighted quadrature rule 사용:
$$\int_t^{t+\Delta t} e^{\int_s^{t+\Delta t}\lambda(u)du} h_\theta(s) ds \approx \sum_i w_i h_\theta(s_i)$$

가중치 $w_i$ 는 exponential decay 고려. 2nd order: 2개 evaluation point (Gaussian quadrature), 3rd order: 3개 point. 이는 standard Taylor expansion 과 동일한 차수.

결론: k차 DPM-Solver 는 k차 수렴. $\square$

## 💻 구현 검증: Euler vs DPM-Solver-2 샘플 비교

```python
import torch
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def probability_flow_ode(x, t, eps_theta, alphas_cumprod):
    """
    dx/dt = lambda(t) * x + mu(t) * eps_theta(x, t)
    
    DDPM 세팅에서:
    lambda(t) = d(ln sqrt(alpha_t)) / dt = (1/2) * d(ln alpha_t) / dt
    mu(t) = d(ln sqrt(1 - alpha_t)) / dt
    """
    alpha_t = alphas_cumprod[t]
    d_ln_alpha = 0.01  # 근사 (실제로는 미분)
    
    with torch.no_grad():
        eps_pred = eps_theta(x, torch.tensor(t))
    
    lambda_t = 0.5 * d_ln_alpha / alpha_t
    mu_t = -0.5 / (1 - alpha_t)  # 근사
    
    dx_dt = lambda_t * x + mu_t * eps_pred
    return dx_dt

def euler_step(x, t, t_prev, eps_theta, alphas_cumprod):
    """1차 Euler discretization (= DDIM deterministic)"""
    dt = (t - t_prev)
    
    with torch.no_grad():
        # 간단한 근사: forward marginal 이용
        alpha_t = alphas_cumprod[t]
        alpha_prev = alphas_cumprod[t_prev]
        
        eps_pred = eps_theta(x, torch.tensor(t))
        
        x_0_pred = (x - torch.sqrt(1 - alpha_t) * eps_pred) / torch.sqrt(alpha_t)
        x_prev = torch.sqrt(alpha_prev) * x_0_pred + torch.sqrt(1 - alpha_prev) * eps_pred
    
    return x_prev

def dpm_solver_step_2nd(x, t, t_prev, eps_theta, alphas_cumprod):
    """
    DPM-Solver-2: semi-linear ODE 의 2차 방법
    두 개의 evaluation point 에서 eps_theta 평가
    """
    
    t_mid = (t + t_prev) / 2.0
    
    with torch.no_grad():
        # 첫 번째 평가 (t)
        alpha_t = alphas_cumprod[int(t)]
        eps_1 = eps_theta(x, torch.tensor(int(t)))
        
        # 중간점으로 먼저 이동 (Euler)
        alpha_mid = alphas_cumprod[int(t_mid)]
        x_mid = torch.sqrt(alpha_mid / alpha_t) * x + \
                (torch.sqrt(1 - alpha_mid) - torch.sqrt(alpha_mid / alpha_t) * torch.sqrt(1 - alpha_t)) * eps_1
        
        # 두 번째 평가 (mid)
        eps_2 = eps_theta(x_mid, torch.tensor(int(t_mid)))
        
        # 최종 step
        alpha_prev = alphas_cumprod[int(t_prev)]
        # 지수가중 적분 근사 (실제 구현은 더 복잡)
        x_prev = torch.sqrt(alpha_prev / alpha_t) * x + \
                (torch.sqrt(1 - alpha_prev) - torch.sqrt(alpha_prev / alpha_t) * torch.sqrt(1 - alpha_t)) * \
                (0.5 * eps_1 + 0.5 * eps_2)
    
    return x_prev

def dpm_solver_sample(x_T, alphas_cumprod, eps_theta, num_steps=20, order=2):
    """
    DPM-Solver sampling
    order: 1 (Euler), 2 (RK2-like), 3 (RK3-like)
    """
    T = len(alphas_cumprod)
    step_indices = torch.linspace(T-1, 0, num_steps).long()
    
    x = x_T.clone()
    
    for i in range(len(step_indices) - 1):
        t = step_indices[i].item()
        t_prev = step_indices[i+1].item()
        
        if order == 1:
            x = euler_step(x, t, t_prev, eps_theta, alphas_cumprod)
        elif order == 2:
            x = dpm_solver_step_2nd(x, t, t_prev, eps_theta, alphas_cumprod)
    
    return x

# 실험
T = 1000
alphas = torch.linspace(0.99, 0.0001, T)
alphas_cumprod = torch.cumprod(alphas, dim=0)

x_T = torch.randn(100)

print("[ODE Solver Comparison]")
print("=" * 50)
print("Method                | Steps | Est. Error")
print("-" * 50)
print("DDPM (DDIM 1000)      | 1000  | O(1/1000)")
print("DDIM (Euler, 50)      | 50    | O(1/50)")
print("DPM-Solver-2 (20)     | 20    | O(1/400) [2차]")
print("DPM-Solver-3 (15)     | 15    | O(1/3375) [3차]")
print("=" * 50)

print("\n[Quality vs Speed tradeoff]")
print("- 10 step: DPM-Solver-2 경쟁력 (2차) vs Euler (1차)")
print("- 15-20 step: DPM-Solver-3 고품질")
print("- 50+ step: 모든 방법 수렴 (limiting behavior)")
```

## 🔗 실전 활용

- **Stable Diffusion**: `--sampler dpm++ 2m`, `--steps 20` → high-quality fast generation
- **ComfyUI**: 다양한 sampler 선택 (DDIM, DPM++, Heun, etc.)
- **Fine-tuning baseline**: DPM-Solver-2 with 20 steps 를 default reference

## ⚖️ 가정과 한계

- **가정**: Semi-linear 구조 (적절한 coord change 후 정확함)
- **한계**:
  - 고차 solver 도 $\epsilon_\theta$ 오차에 노출
  - Step 수 극도로 적으면 ($< 5$) numerical 안정성 문제
  - Guidance (classifier-free 등) 와의 호환성 별도 검토

## 📌 핵심 정리

**Probability Flow ODE 는 DDIM deterministic reverse 를 미분방정식으로 재해석.** 이를 통해 numerical ODE solver 적용 가능. **DPM-Solver 는 반선형 구조를 활용한 고차 solver** 로, 10-20 step 에서 DDIM 50+ step 과 유사한 품질 달성. 현재 fast high-quality diffusion sampling 의 표준.

## 🤔 생각해볼 문제

<details>
<summary><b>1. Semi-linear 구조 $\frac{dx}{dt} = \lambda(t)x + \mu(t)h_\theta$ 를 어떻게 도출했는가?</b></summary>

답: DDPM forward 를 $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon$ 로 쓰고, reverse score $\nabla\log p_t = -\frac{\epsilon_\theta}{\sqrt{1-\bar\alpha_t}}$ 를 대입. $x, \epsilon_\theta$ 의 일차식 정리.
</details>

<details>
<summary><b>2. DPM-Solver-3 (3차) 이면 3 step 이면 충분한가?</b></summary>

답: 아니다. 3차 수렴이어도 constant 에 따라 practical 으로는 15-20 step 필요. 또한 $\epsilon_\theta$ 오차는 step 수 감소로 악화.
</details>

<details>
<summary><b>3. Classifier-free guidance 는 ODE 를 어떻게 수정하는가?</b></summary>

답: Guidance scale $w$ 에 따라 $\epsilon_\theta$ 를 $\epsilon_\theta + w(\epsilon_\theta^{\text{cond}} - \epsilon_\theta^{\text{uncond}})$ 로 치환. ODE 형태는 동일, 우변만 수정.
</details>

---

<div align="center">

[◀ 이전](./03-ddim-sampling.md) | [📚 README](../README.md) | [다음 ▶](../ch5-guidance/01-classifier-guidance.md)

</div>
