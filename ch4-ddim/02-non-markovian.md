# 02. Non-Markovian Forward Process 와 동일 Marginal

## 🎯 핵심 질문

Forward process 를 Markovian 으로 고정해야 하는가? 다른 forward path 를 택해도 학습된 역확산 모델을 재사용할 수 있을까?

## 🔍 왜 Non-Markovian Forward 를 고려하는가?

DDPM forward 는:
$$q(x_{1:T}|x_0) = q(x_T|x_0) \prod_{t=1}^{T-1} q(x_{t-1}|x_t, x_{t+1},...,x_T)$$

각 time step $t$ 에서 **미래 정보(future noise)** 를 조건으로 할 수 있다. Song (2021) DDIM 논문에서는 이를 활용하여:

1. **Forward: 임의 $\sigma$ 로 정의** → Markovian 조건 완화
2. **Marginal 유지** → $q(x_t|x_0)$ 동일 (중요!)
3. **Reverse 재사용** → 학습된 $\epsilon_\theta$ 그대로 적용

결과: **같은 marginal 아래 다양한 forward 경로** 가능 → 서로 다른 reverse trajectory 가능.

## 📐 수학적 선행 조건

- Marginal Gaussian: $q(x_t|x_0) = \mathcal{N}(\sqrt{\bar\alpha_t}x_0, (1-\bar\alpha_t)I)$
- $\bar\alpha_t = \prod_{s=1}^t \alpha_s$
- 조건부 가우시안: $\mathcal{N}(\mu_1, \Sigma_1) + \mathcal{N}(\mu_2, \Sigma_2) = \mathcal{N}(\mu_1+\mu_2, \Sigma_1+\Sigma_2)$ (독립일 때)

## 📖 직관적 이해

표준 DDPM forward:
$$x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon, \quad \epsilon \sim \mathcal{N}(0,I)$$

이는 deterministic 부분($\sqrt{\bar\alpha_t}x_0$) + random 부분($\sqrt{1-\bar\alpha_t}\epsilon$) 의 합성.

**Non-Markovian 아이디어**: Random 부분을 **다양한 방식**으로 분해할 수 있다. 예를 들어:

$$x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t-\sigma_t^2}\epsilon_1 + \sigma_t\epsilon_2$$

여기서 $\epsilon_1, \epsilon_2 \sim \mathcal{N}(0,I)$ 독립. Marginal variance 는 동일하지만, step 간 상관구조는 다름.

## ✏️ 엄밀한 정의

### 정의 2.1: Non-Markovian Forward Process (Song 2021)

매개변수 $\sigma = (\sigma_1,...,\sigma_T) \in \mathbb{R}^T_+$ 에 대해, 조건부 분포들:

$$q_\sigma(x_{t-1}|x_t, x_0) = \mathcal{N}(\sqrt{\bar\alpha_{t-1}}x_0 + \sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\frac{x_t - \sqrt{\bar\alpha_t}x_0}{\sqrt{1-\bar\alpha_t}}, \sigma_t^2 I)$$

그리고 joint:
$$q_\sigma(x_{1:T}|x_0) = q(x_T|x_0) \prod_{t=2}^T q_\sigma(x_{t-1}|x_t, x_0)$$

### 정의 2.2: Marginal 동일성 (Crucial!)

위 정의에서 **모든 $\sigma$ 에 대해**:
$$q_\sigma(x_t|x_0) = \mathcal{N}(\sqrt{\bar\alpha_t}x_0, (1-\bar\alpha_t)I)$$

이는 $\sigma$ 에 **무관**하다!

## 🔬 정리와 증명

### 정리 2.1: Non-Markovian Forward 의 Marginal 은 Markovian Forward 의 Marginal 과 동일

**명제**: 정의 2.1 의 $q_\sigma$ 에서,
$$\text{marg}_{q_\sigma}(x_t) = q(x_t|x_0) = \mathcal{N}(\sqrt{\bar\alpha_t}x_0, (1-\bar\alpha_t)I)$$

**증명** (귀납법):

*Base case* ($t=T$): 정의에서 $q_\sigma(x_T|x_0) = q(x_T|x_0)$ (동일).

*Inductive step* ($t \to t-1$): 
$$q_\sigma(x_{t-1}|x_0) = \int q_\sigma(x_{t-1}|x_t, x_0) q_\sigma(x_t|x_0) dx_t$$

$q_\sigma(x_{t-1}|x_t, x_0)$ 는 평균 $\mu(x_t) = \sqrt{\bar\alpha_{t-1}}x_0 + \sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\frac{x_t - \sqrt{\bar\alpha_t}x_0}{\sqrt{1-\bar\alpha_t}}$, 분산 $\sigma_t^2 I$ 인 가우시안.

평균의 변량:
$$\mathbb{E}[\mu(x_t)] = \sqrt{\bar\alpha_{t-1}}x_0 + \sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\frac{\mathbb{E}[x_t] - \sqrt{\bar\alpha_t}x_0}{\sqrt{1-\bar\alpha_t}}$$

귀납 가정에서 $\mathbb{E}[x_t|x_0] = \sqrt{\bar\alpha_t}x_0$ 이므로:
$$\mathbb{E}[\mu(x_t)] = \sqrt{\bar\alpha_{t-1}}x_0$$

분산:
$$\text{Var}[x_{t-1}|x_0] = \mathbb{E}[\sigma_t^2 I] + \text{Var}[\mu(x_t)]$$

$\text{Var}[\mu(x_t)] = (1-\bar\alpha_{t-1}-\sigma_t^2) \cdot \frac{\text{Var}[x_t|x_0]}{1-\bar\alpha_t} = (1-\bar\alpha_{t-1}-\sigma_t^2) \cdot \frac{1-\bar\alpha_t}{1-\bar\alpha_t} = 1-\bar\alpha_{t-1}-\sigma_t^2$

따라서:
$$\text{Var}[x_{t-1}|x_0] = \sigma_t^2 + (1-\bar\alpha_{t-1}-\sigma_t^2) = 1-\bar\alpha_{t-1}$$

결론: $q_\sigma(x_{t-1}|x_0) = \mathcal{N}(\sqrt{\bar\alpha_{t-1}}x_0, (1-\bar\alpha_{t-1})I)$. $\square$

## 💻 구현 검증: 다양한 $\sigma$ 에서 동일 Marginal 확인

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

def non_markovian_sample_trajectory(x_0, alphas_cumprod, sigma_sequence):
    """
    Non-Markovian forward process 샘플링
    x_0: initial data
    alphas_cumprod: alpha_bar
    sigma_sequence: (sigma_1, ..., sigma_T)
    """
    T = len(alphas_cumprod)
    x_t = x_0.clone()
    trajectory = [x_0.clone()]
    
    for t in range(1, T):
        alpha_t = alphas_cumprod[t]
        sigma_t = sigma_sequence[t]
        
        # Non-Markovian forward:
        # x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t - sigma_t^2) * epsilon_1 + sigma_t * epsilon_2
        eps1 = torch.randn_like(x_0)
        eps2 = torch.randn_like(x_0)
        
        x_t = (torch.sqrt(torch.tensor(alpha_t)) * x_0 + 
               torch.sqrt(torch.tensor(1.0 - alpha_t - sigma_t**2 + 1e-8)) * eps1 + 
               torch.tensor(sigma_t) * eps2)
        
        trajectory.append(x_t.clone())
    
    return torch.stack(trajectory)

# 실험: 두 가지 다른 sigma 수열로 forward 진행
alphas = torch.linspace(0.99, 0.0001, 50)
alphas_cumprod = torch.cumprod(alphas, dim=0)

x_0 = torch.randn(1000)  # batch 1000

# Sigma 설정 1: 모두 0 (DDIM 스타일)
sigma1 = torch.zeros(50)

# Sigma 설정 2: 큼 (DDPM 스타일)
sigma2 = torch.sqrt((1 - alphas_cumprod[:-1]) / (1 - alphas_cumprod[1:]) * 
                     (1 - alphas_cumprod[:-1] / alphas_cumprod[:-1]))  # 근사

# Forward trajectory 샘플링
traj1 = non_markovian_sample_trajectory(x_0, alphas_cumprod, sigma1)
traj2 = non_markovian_sample_trajectory(x_0, alphas_cumprod, sigma2)

# Marginal 검증: q(x_t | x_0) 는 sigma 무관
def compute_marginal_var(x_t, x_0, alpha_cumprod_t):
    """x_t 의 variance 계산"""
    return torch.var(x_t - torch.sqrt(torch.tensor(alpha_cumprod_t)) * x_0)

print("[Marginal Variance Check]")
for t in [10, 25, 40]:
    alpha_t = alphas_cumprod[t].item()
    expected_var = (1 - alpha_t)
    
    var1 = compute_marginal_var(traj1[t], x_0, alpha_t)
    var2 = compute_marginal_var(traj2[t], x_0, alpha_t)
    
    print(f"t={t}: Expected={expected_var:.4f}, sigma1={var1:.4f}, sigma2={var2:.4f}")

# 결과: 서로 다른 sigma 에도 불구하고 marginal variance 동일 ✓
```

## 🔗 실전 활용

- **DDIM 구현**: sigma 를 0 으로 설정 → deterministic sampling
- **DDPM 호환**: sigma 를 원래 값으로 유지 → stochastic sampling
- **중간값**: sigma 를 조정하여 스토캐스틱/디터미니스틱 트레이드오프

## ⚖️ 가정과 한계

- **가정**: 조건부 $q(x_{t-1}|x_t, x_0)$ 정의의 valid variance (음이 아님)
- **한계**: $1 - \bar\alpha_{t-1} - \sigma_t^2 \geq 0$ 조건 필요 → sigma 범위 제한

## 📌 핵심 정리

**Non-Markovian forward process 는 같은 marginal 을 유지하면서도 다양한 조건부 구조를 허용한다.** 이를 통해 동일한 학습된 reverse model $\epsilon_\theta$ 를 여러 forward path 에 적용 가능. 특히 $\sigma_t=0$ 선택 (DDIM) 은 deterministic sampling 을 가능하게 하고, 이로써 step 수를 획기적으로 줄일 수 있다.

## 🤔 생각해볼 문제

<details>
<summary><b>1. Non-Markovian forward 에서 조건부 $q(x_{t-1}|x_t, x_0)$ 가 Gaussian 인 이유?</b></summary>

답: Linear system 이기 때문. $x_t, x_0$ 모두 Gaussian 의 linear combination 이고, conditioning 도 Gaussian linear regression → Gaussian conditional.
</details>

<details>
<summary><b>2. $\sigma_t^2 > 1 - \bar\alpha_{t-1}$ 이 되면 무엇이 문제인가?</b></summary>

답: 조건부 분산 $1-\bar\alpha_{t-1}-\sigma_t^2$ 이 음수가 되어 물리적으로 불가능한 가우시안 정의. 따라서 sigma 는 bounded 여야 함.
</details>

<details>
<summary><b>3. Reverse model $\epsilon_\theta$ 는 어떤 forward path 에 맞춰 학습된 것인가?</b></summary>

답: DDPM 은 표준 Markovian forward (sigma_DDPM) 에서 학습되지만, Non-Markovian 에서도 같은 marginal 이므로 역확산 목표는 동일. 따라서 재사용 가능.
</details>

---

<div align="center">

[◀ 이전](./01-ddim-motivation.md) | [📚 README](../README.md) | [다음 ▶](./03-ddim-sampling.md)

</div>
