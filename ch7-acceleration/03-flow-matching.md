# 7.3 Flow Matching (Lipman 2023)

## 🎯 핵심 질문

Diffusion models의 다양한 변형들(VP-SDE, Rectified Flow, etc.)을 통합할 수 있을까? **Flow Matching**은 Continuous Normalizing Flow (CNF)의 일반화로서, 어떤 확률 경로(probability path)와 속도장(vector field)도 직접 학습할 수 있는 유연한 프레임워크를 제공합니다.

## 🔍 왜 Flow Matching인가?

1. **Unification**: VP-SDE, Rectified Flow 등이 모두 Flow Matching의 특수 사례
2. **Flexibility**: 확률 경로 $p_t(x|x_1)$ 를 자유롭게 선택 가능
3. **Direct matching**: Score estimation 없이 바로 velocity 학습
4. **Conditional**: Guidance와 유연한 결합 가능 (Conditional Flow Matching)

## 📐 수학적 선행 조건

- **Continuous Normalizing Flow**: 시간에 따른 미분가능 변환 $\phi_t$
- **Score 함수**: $\nabla_x \log p_t(x)$
- **Probability path**: $p_t(x)$는 $t$에 대해 미분가능
- **ODE 해**: $\frac{\mathrm{d}x}{\mathrm{d}t} = v_t(x)$

## 📖 직관적 이해

**Probability Flow ODE** 관점에서, 시간 $t$에서의 분포 $p_t(x)$는 velocity field $u_t(x)$에 의해 진화합니다:
$$\frac{\partial p_t}{\partial t} + \nabla \cdot (p_t u_t) = 0$$

이를 풀기 위해, 보통은 score function $\nabla \log p_t$를 먼저 추정해야 합니다.

**Flow Matching** 아이디어:
- Probability path $p_t(x | x_1)$를 직접 선택
- 이에 대응하는 "target" velocity field $u_t(x | x_1)$ 정의
- Neural network $v_\theta(x, t)$가 이 velocity를 직접 matching하도록 학습

$$\mathcal{L} = \mathbb{E}_{t, x_1, x} \left\| v_\theta(x, t) - u_t(x | x_1) \right\|^2$$

이렇게 하면 score estimation이 필요 없고, 직접 velocity matching으로 빠르게 학습 가능합니다.

## ✏️ 엄밀한 정의

**정의 (Probability Path)**

$p_t(x | x_1)$: $t=0$에서 data $p_0(x) = p_{\text{data}}(x)$, $t=1$에서 noise $p_1(x) = p_z(x)$로 interpolate하는 조건부 분포.

**정의 (Flow Matching Target)**

$$u_t(x | x_1) = \frac{\mathrm{d}}{\mathrm{d}t} \log p_t(x | x_1) + \frac{1}{p_t(x | x_1)} \nabla_x p_t(x | x_1)$$

실제로는 다음과 같이 간단히 표현:
$$u_t(x | x_1) = \frac{\nabla_x p_t(x | x_1)}{p_t(x | x_1)} = \nabla_x \log p_t(x | x_1)$$

아니면 parametric form으로:
$$u_t(x | x_1) = \frac{\partial_t x_t(x, x_1)}{\text{정규화}}$$

**Flow Matching Loss**
$$\mathcal{L}_{\text{FM}} = \mathbb{E}_{t \sim \mathcal{U}(0,1), x_1 \sim p_z, x \sim p_t(\cdot | x_1)} \left\| v_\theta(x, t) - u_t(x | x_1) \right\|^2$$

**정의 (Conditional Flow Matching, CFM)**

더 효율적인 variant: target을 조건부 분포의 expectation으로 근사
$$u_t^{\text{CFM}}(x | x_1) = \frac{\mathbb{E}_{x_0 \sim p_0(x | x_1)} [\nabla_x \log p_t(x | x_1, x_0) \cdot p_t(x | x_1, x_0)]}{\mathbb{E}_{x_0} [p_t(x | x_1, x_0)]}$$

## 🔬 정리와 증명

**정리 (VP-SDE = OT-CFM)**

Variance Preserving SDE (diffusion의 표준 형태)는 Flow Matching의 특수 경우로, Optimal Transport를 이용한 조건부 분포 선택 하에서 동일합니다.

*증명 스케치* ($\square$)

VP-SDE의 reverse ODE는:
$$\mathrm{d}x = \left[f(x, t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x) \right] \mathrm{d}t$$

이를 Flow Matching 형식으로 다시 쓰면:
$$\mathcal{L} = \mathbb{E}_{x_0 \sim p_{\text{data}}} \int_0^1 \left\| v_\theta(x_t, t) - s_\theta(x_t, t) \right\|^2 \mathrm{d}t$$

여기서 score $s_\theta$는 우리가 이전에 배운 "velocity"로 해석됩니다. 따라서 VP-SDE를 따르는 diffusion은 특정 $p_t(x | x_1) = \mathcal{N}(x; \alpha_t x_1 + (1-\alpha_t)\mu, \sigma_t^2 I)$ 선택에 해당하는 Flow Matching입니다.

**정리 (Rectified Flow ⊂ FM)**

Rectified Flow의 직선 경로 $x_t = (1-t)x_0 + tz$도 Flow Matching으로 표현 가능합니다. 이 경우 conditional distribution은:
$$p_t(x | x_0, z) = \delta(x - ((1-t)x_0 + tz))$$

(Dirac delta), target velocity는:
$$u_t(x | x_0, z) = z - x_0 \quad (\text{constant})$$

따라서 Rectified Flow도 Flow Matching의 deterministic coupling 특수 경우입니다.

## 💻 NumPy / PyTorch 구현 검증

### 실험 1: Probability Path 정의

```python
import torch
import torch.nn as nn
import torch.distributions as dist

def linear_probability_path(x0, z, t):
    """Linear interpolation: p_t(x|x0,z) = N(x; (1-t)*x0 + t*z, sigma^2*I)"""
    mean = (1 - t.unsqueeze(1)) * x0 + t.unsqueeze(1) * z
    sigma = 0.1  # Small noise for numerical stability
    return mean, sigma

x0_sample = torch.randn(8, 2)
z_sample = torch.randn(8, 2)
t_sample = torch.rand(8)

means, sigmas = linear_probability_path(x0_sample, z_sample, t_sample)
print(f"Mean shape: {means.shape}, Sigma: {sigmas}")
```

### 실험 2: Target Velocity Computation

```python
class FlowMatchingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 2)
        )
    
    def forward(self, x, t):
        """v_theta(x, t) predicts target velocity"""
        t_expanded = t.view(-1, 1).expand(x.shape[0], 1)
        return self.net(torch.cat([x, t_expanded], dim=1))

def compute_target_velocity(x0, z, t):
    """For linear path: u_t = dCL/dt (x - mean) / sigma^2 or simply z - x0"""
    # Simple: constant velocity for straight-line path
    return z - x0

model = FlowMatchingNet()
target_vel = compute_target_velocity(x0_sample, z_sample, t_sample)
print(f"Target velocity shape: {target_vel.shape}")  # [8, 2]
```

### 실험 3: Flow Matching Training

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(100):
    # Sample batch
    x0 = torch.randn(16, 2)
    z = torch.randn(16, 2)
    t = torch.rand(16)
    
    # Probability path sampling
    x_t = (1 - t.unsqueeze(1)) * x0 + t.unsqueeze(1) * z
    
    # Target velocity (for linear path)
    u_target = z - x0
    
    # Predicted velocity
    v_pred = model(x_t, t)
    
    # Flow Matching loss
    loss = ((v_pred - u_target) ** 2).mean()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}: FM Loss = {loss.item():.6f}")

# 1-step sampling
z_test = torch.randn(10, 2)
with torch.no_grad():
    t_test = torch.ones(10) * 1.0
    x_sampled = z_test.clone()
    # Simple Euler step backward
    x_sampled = x_sampled - 1.0 * model(x_sampled, t_test)
print("Sampling complete.")
```

## 🔗 실전 활용

1. **Unified framework**: VP-SDE, Rectified Flow, VE-SDE 등 다양한 diffusion 변형을 통일된 Flow Matching 프레임워크로 구현
2. **Conditional Generation**: Text-to-image에서 text conditioning을 CFM으로 직접 통합
3. **Fast sampling**: Flow Matching + few-step ODE solver로 빠른 샘플링 (2-4 steps)
4. **Theoretical clarity**: Score matching의 부재로 계산 더 단순, 이론적 분석 용이

## ⚖️ 가정과 한계

1. **Probability path 선택**: 어떤 경로를 선택할지는 여전히 문제. 데이터마다 최적 경로가 다를 수 있음
2. **CFM 근사**: Conditional Flow Matching은 expectation 근사이므로, 정확도는 선택된 조건부 분포에 의존
3. **Numerical stability**: Probability path가 매우 narrow하면 score 계산 불안정
4. **Guidance 통합**: Classifier-free guidance 등과의 통합은 별도 논의 필요

## 📌 핵심 정리

**Flow Matching** (Lipman 2023)은 Continuous Normalizing Flow의 일반화로, 임의의 확률 경로와 속도장을 직접 matching하는 프레임워크입니다:
- **Unification**: VP-SDE, Rectified Flow 등을 포함
- **Flexibility**: 경로와 velocity target을 자유롭게 설계 가능
- **Direct learning**: Score estimation 없이 velocity 직접 학습
- **Modern backbone**: Diffusion 생성 모델의 통일된 이론적 기초

## 🤔 생각해볼 문제

1. Flow Matching에서 probability path $p_t(x | x_1)$를 선택할 때, 어떤 기준으로 최적 경로를 결정할까요? VP-SDE와 Rectified Flow 중 어느 것이 더 나을까요?

   <details>
   <summary>힌트</summary>
   최적 경로는 "가장 작은 curvature"를 가지거나, "최소 transportation cost"를 만족하는 경로입니다. 이는 데이터 분포와 노이즈 분포에 따라 다릅니다. 실제로는 여러 경로를 시도하고 FID 등으로 비교하는 것이 일반적입니다.
   </details>

2. Conditional Flow Matching (CFM)에서 조건부 분포의 expectation을 취할 때, 샘플링 수(몇 개의 $x_0$을 사용할지)에 따라 근사 오차가 달라질 것 같습니다. 이를 분석하는 방법이 있을까요?

   <details>
   <summary>힌트</summary>
   Law of large numbers에 의해, 샘플 수가 증가하면 expectation 근사는 정확해집니다. 실제로 CFM 논문에서는 batch size와 근사 오차 간의 trade-off를 분석합니다. 일반적으로 작은 batch도 충분히 좋은 근사를 제공합니다.
   </details>

3. Flow Matching과 guidance (예: classifier-free guidance)를 결합하면 어떤 효과가 있을까요? Guidance scale을 velocity level에서 직접 조절할 수 있을까요?

   <details>
   <summary>힌트</summary>
   Guidance는 보통 score level에서 조절되지만, Flow Matching에서는 velocity level에서도 가능합니다. $v_{\text{guided}}(x, t) = v_{\text{uncond}}(x, t) + \lambda \nabla_x \log p_{\text{condition}}(x | c)$ 형태로 정의할 수 있으며, 이는 gradient-based guidance와 유사합니다.
   </details>

---

<div align="center">

[◀ 이전](./02-rectified-flow.md) | [📚 README](../README.md) | [다음 ▶](./04-distillation.md)

</div>
