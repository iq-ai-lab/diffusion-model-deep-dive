# 7.1 Consistency Model (Song 2023)

## 🎯 핵심 질문

기존 Diffusion Model은 reverse process에서 수백 스텝을 요구합니다. **1-step 샘플링도 가능할까?** Consistency Model은 ODE trajectory 위의 모든 시점에서 noise 상태가 같은 clean sample로 매핑되도록 강제하는 consistency property를 통해 이를 실현합니다.

## 🔍 왜 Consistency Model인가?

Diffusion model의 샘플링 속도는 매우 느립니다(보통 50-1000 스텝). 이를 개선하려면:
1. **많은 스텝 필요**: 정밀도 ↑ → 계산량 ↑
2. **Distillation**: 기존 학습된 모델 필요 (cold-start 불가)
3. **Consistency**: 모든 시점에서 같은 값으로 수렴 → 어떤 시점에서든 1-step 가능

## 📐 수학적 선행 조건

- **Probability Flow ODE**: $\mathrm{d}x_t = \left[\mu_t(x) - \frac{\sigma_t^2}{2}\nabla \log p_t(x)\right] \mathrm{d}t$
- **Score 근사**: $s_\theta(x_t, t) \approx \nabla \log p_t(x_t)$
- **ODE 상의 궤적**: 고정된 $x_0$에 대해, 같은 ODE는 유일한 경로를 정의

## 📖 직관적 이해

ODE trajectory를 따라 역진할 때, $t \to 0$에 가까울수록 noisy sample $x_t$가 clean data $x_0$에 가까워집니다. **Consistency property**는 "trajectory 위의 어떤 $(t_1, x_{t_1})$과 $(t_2, x_{t_2})$ (같은 이전 시점)가 같은 $x_0$으로 매핑된다"를 의미합니다.

수식: $f_\theta(x_{t_1}, t_1) = f_\theta(x_{t_2}, t_2) = x_0$

이를 학습하면, 최종 샘플링 스텝 전 $t$에서 바로 $x_0$을 예측할 수 있어 1-step 가능합니다.

## ✏️ 엄밀한 정의

**정의 (Consistency Function)**
$$f_\theta(x, t) : \mathbb{R}^d \times [t_{\min}, T] \to \mathbb{R}^d$$

다음을 만족:
1. **Consistency**: 같은 ODE trajectory 위의 점들은 같은 $f_\theta$값 → $x_0$
2. **Boundary**: $f_\theta(x_0, 0) = x_0$ (clean sample는 자신으로 매핑)
3. **1-step 샘플링**: $x_0 \approx f_\theta(z, T)$, $z \sim \mathcal{N}(0, I)$

**학습 목표 (Consistency Training, CT)**
$$\mathcal{L}_{CT} = \mathbb{E}_{x_0, t_1, t_2} \left\| f_\theta(x_{t_1}, t_1) - f_{\phi}(x_{t_2}, t_2) \right\|^2$$

여기서 $x_{t_i}$는 같은 ODE trajectory 위의 서로 다른 시점.

**Consistency Distillation (CD)**
$$\mathcal{L}_{CD} = \mathbb{E}_{t_1, t_2} \left\| f_\theta(x_{t_1}, t_1) - f_{\text{teacher}}(x_{t_2}, t_2) \right\|^2$$

(Teacher는 학습된 pretrained diffusion ODE solver)

## 🔬 정리와 증명

**정리 (Consistency Preserves ODE Convergence)**

ODE trajectory 위의 임의의 두 점 $(x_{t_1}, t_1)$, $(x_{t_2}, t_2)$에 대해, consistency property를 만족하는 $f_\theta$를 학습했다면, 1-step sampling $x_0 = f_\theta(x_T, T)$의 분포는 diffusion의 원래 reverse process에서 $t_1 \to 0$으로 감소시킨 것과 유사합니다.

*증명 스케치* ($\square$)

Probability Flow ODE의 성질에 의해, 같은 initial $x_T$에서 출발하는 trajectory는 유일합니다. Consistency training이 "서로 다른 $t$에서 출발한 점들이 같은 $x_0$으로 수렴"을 강제하므로, $f_\theta$는 이 유일한 ODE 궤적을 따르는 multi-step sampling과 임의의 중간 $t$에서의 1-step을 연결합니다. 따라서 $f_\theta(x_T, T)$는 full reverse process와 거의 같은 분포에서 샘플을 추출합니다.

## 💻 NumPy / PyTorch 구현 검증

### 실험 1: 1D Consistency Model

```python
import numpy as np
import matplotlib.pyplot as plt

def simulate_ode_trajectory(x0, t_start=1.0, t_end=0.0, num_steps=100):
    """간단한 1D ODE 시뮬레이션: dx/dt = -x (exponential decay)"""
    ts = np.linspace(t_start, t_end, num_steps)
    xs = [x0]
    for i in range(len(ts)-1):
        dt = ts[i+1] - ts[i]
        xs.append(xs[-1] * np.exp(dt))  # x(t) = x0 * exp(t)
    return ts, np.array(xs)

# 테스트
x0_true = 0.5
ts, trajectory = simulate_ode_trajectory(x0_true)
print(f"Starting x_T={trajectory[0]:.4f}, ending x_0={trajectory[-1]:.4f}")
# Expected: x_0 ≈ x0_true (due to exp(1.0) ≈ e)
```

### 실험 2: Consistency Loss

```python
import torch
import torch.nn as nn

class ConsistencyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x, t):
        """f_theta(x, t) -> x_0 prediction"""
        return self.net(torch.cat([x.unsqueeze(-1), t.unsqueeze(-1)], dim=-1))

model = ConsistencyNet()
x_sample = torch.randn(16, 1)
t_sample = torch.rand(16)
x0_pred = model(x_sample, t_sample)
print(f"Batch prediction shape: {x0_pred.shape}")  # [16, 1]
```

### 실험 3: Consistency Training Loop

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 같은 trajectory 위의 (t1, x_t1), (t2, x_t2) sampling
for epoch in range(100):
    t1 = torch.rand(8) * 0.9 + 0.1
    t2 = torch.rand(8) * 0.9 + 0.1
    x_t1 = torch.randn(8, 1) * (1 - t1.unsqueeze(-1))
    x_t2 = torch.randn(8, 1) * (1 - t2.unsqueeze(-1))
    
    pred1 = model(x_t1, t1)
    pred2 = model(x_t2, t2)
    
    loss = ((pred1 - pred2) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
```

## 🔗 실전 활용

1. **CIFAR-10 1-step FID ~9**: Song et al. (2023)에서 consistency model은 1-step으로도 DDIM 50-step과 비슷한 수준의 FID 달성
2. **Multi-step 샘플링**: consistency property 덕분에 2-step, 4-step도 가능하며, 스텝이 많을수록 품질 향상
3. **Cold-start vs. Distillation**: CT는 scratch에서, CD는 pretrained teacher 필요 (하지만 더 빠름)
4. **Real-world**: Latent diffusion과 결합 (예: SDXL 가속화)

## ⚖️ 가정과 한계

1. **ODE trajectory 가정**: Consistency model의 이점은 probability flow ODE의 유일성에 의존. SDE는 별도 처리 필요
2. **Score estimation**: 학습된 score $s_\theta$의 품질이 consistency function 학습에 영향
3. **Multi-step vs. 1-step tradeoff**: 1-step은 빠르지만 품질 손실 가능. Multi-step은 느리지만 더 나음
4. **학습 안정성**: Consistency training은 neighboring timestep 간 loss 최소화 필요 → 신중한 schedule 관리

## 📌 핵심 정리

**Consistency Model**은 diffusion ODE trajectory 위의 모든 점이 같은 clean data로 매핑되는 consistency property를 학습합니다. 이를 통해:
- **1-step 샘플링** 가능 (매우 빠름, 품질 ~9 FID on CIFAR-10)
- **Multi-step도 지원** (더 정밀함)
- **Cold-start (CT) 또는 distillation (CD)** 옵션 제공
- ODE 구조의 우아한 활용으로 새로운 속도-품질 트레이드오프 창출

## 🤔 생각해볼 문제

1. Consistency training에서 neighboring timestep $(t_1, t_2)$를 선택할 때, 너무 가까우면 정보 부족, 너무 멀면 consistency 강제가 어려울 수 있습니다. 최적의 거리를 어떻게 결정할까요?

   <details>
   <summary>힌트</summary>
   Score estimation의 오차가 작은 영역(큰 $t$)에서는 거리를 크게, 오차가 큰 영역(작은 $t$)에서는 거리를 작게 설정하는 것이 일반적입니다.
   </details>

2. Consistency distillation (CD)에서 teacher model이 부정확하면 학생 모델도 부정확해질 것 같습니다. 이 오차 전파를 어떻게 최소화할까요?

   <details>
   <summary>힌트</summary>
   Teacher의 여러 샘플링 경로를 앙상블하거나, EMA (exponential moving average)를 사용하여 teacher를 점진적으로 개선하는 방법이 있습니다.
   </details>

3. Consistency model은 ODE 기반인데, 원래 diffusion은 SDE (확률 미분 방정식)로도 표현됩니다. SDE consistency를 어떻게 정의하고 학습할까요?

   <details>
   <summary>힌트</summary>
   SDE의 경우 다중 경로(path)가 같은 $x_0$에 도달할 수 있으므로, ODE보다 더 약한 consistency property (분포 수준)를 정의하거나, 특정 variance schedule 아래 ODE 근사를 사용할 수 있습니다.
   </details>

---

<div align="center">

[◀ 이전](../ch6-latent-arch/05-cascaded-diffusion.md) | [📚 README](../README.md) | [다음 ▶](./02-rectified-flow.md)

</div>
