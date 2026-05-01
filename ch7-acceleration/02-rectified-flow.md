# 7.2 Rectified Flow (Liu 2022)

## 🎯 핵심 질문

Diffusion model의 경로가 "굽어있다"면 어떨까요? **Straight-line path**로 단순화할 수 있다면, 1-step 샘플링이 더 정확할 것입니다. Rectified Flow는 forward process를 직선 경로로 정의하고, 학습된 ODE를 다시 학습(reflow)하는 반복을 통해 경로를 더욱 직선화합니다.

## 🔍 왜 Rectified Flow인가?

1. **Straight path의 장점**: 굽은 경로 → 1-step 어려움. 직선 경로 → 1-step 용이
2. **Coupling constraint**: $(x_0, z)$ 쌍을 명시적으로 연결 → 경로의 형태 결정
3. **Reflow 메커니즘**: 반복 학습으로 직선도 증가 → 극한에서 선형 경로
4. **SD3 기반**: Stable Diffusion 3의 기본 구조

## 📐 수학적 선행 조건

- **ODE trajectory**: $\frac{\mathrm{d}x_t}{\mathrm{d}t} = v_t(x_t)$ (velocity field)
- **Optimal transport**: Wasserstein distance 최소화
- **Score matching**: $\mathbb{E}\|s_\theta - \nabla \log p\|^2$와 유사 개념
- **Coupling**: $\pi(x_0, z)$ = joint distribution

## 📖 직관적 이해

**Forward process** (생성 ← 노이즈)는 보통 가우시안 노이즈에서 시작합니다:
$$x_t = (1-t)x_0 + tz, \quad z \sim \mathcal{N}(0, I), \quad t \in [0,1]$$

이는 기하학적으로 $x_0$과 $z$ 사이의 **직선**입니다.

**Velocity field** $v_\theta(x_t, t)$는 이 직선 상의 instantaneous motion을 나타냅니다:
$$v_{\text{ideal}}(x_t, t) = z - x_0$$

학습 목표는:
$$\min_\theta \mathbb{E} \left\| v_\theta(x_t, t) - (z - x_0) \right\|^2$$

**Reflow** (다시 학습):
1. 학습된 ODE solver를 이용해 $(x_0, z)$를 새로운 쌍으로 변환
2. 이 새로운 쌍의 경로가 더 직선에 가까워짐
3. 반복하면 극한에서 완벽한 직선

## ✏️ 엄밋한 정의

**정의 (Straight-line Forward Process)**
$$x_t = (1-t)x_0 + tz, \quad t \in [0,1]$$

여기서 $z \sim p_z$ (e.g., $\mathcal{N}(0, I)$)이고, coupling $\pi(x_0, z)$가 주어졌을 때.

**정의 (Velocity Field)**
$$v_\theta(x_t, t) : \mathbb{R}^d \times [0,1] \to \mathbb{R}^d$$

ODE: $\frac{\mathrm{d}x_t}{\mathrm{d}t} = v_\theta(x_t, t)$, $x_0 \sim p_{\text{data}}$, $x_1 \sim p_z$

**학습 목표 (Velocity Matching)**
$$\mathcal{L} = \mathbb{E}_{(x_0, z) \sim \pi} \int_0^1 \left\| v_\theta(x_t, t) - (z - x_0) \right\|^2 \mathrm{d}t$$

또는 discrete approximation:
$$\mathcal{L} = \mathbb{E}_{(x_0, z) \sim \pi, t} \left\| v_\theta(x_t, t) - (z - x_0) \right\|^2$$

**정의 (Reflow)**

학습된 velocity $v_\theta$로부터:
1. $(x_0, z)$ 쌍을 ODE 상의 새로운 쌍 $(x_0', z')$으로 변환:
   - $x_0' \sim$ pushforward of $x_0$ through ODE
   - $z'$ = $t=1$에서의 ODE output
2. 새로운 쌍 $(x_0', z')$에 대해 velocity matching 재학습
3. 반복 → coupling 구조 개선 → 직선도 증가

## 🔬 정리와 증명

**정리 (Reflow Convergence to Linear Transport)**

Independent coupling (i.e., $\pi(x_0, z) = p_{\text{data}}(x_0) \cdot p_z(z)$)에서 시작하면, 각 reflow 단계마다 learned ODE의 trajectories가 optimal linear transport에 더 가까워집니다. 극한에서 velocity field는 상수가 되어, 1-step sampling이 정확해집니다.

*증명 스케치* ($\square$)

Monge-Kantorovich theorem에 의해, OT 문제의 최적해는 transport map $T^*$입니다. Straight-line coupling $(x_0, T^*(x_0))$를 사용하면, velocity는 $v^*(x_t, t) = T^*(x_0) - x_0$ (constant).

Reflow는 현재 learned $v_\theta$에서 $(x_0, x_1^\theta)$ (ODE의 출력)로 새 coupling을 정의합니다. 이는 $v_\theta$가 $v^*$에 가까워질수록, coupling도 optimal transport에 수렴하게 합니다. 따라서 반복 reflow 후 $v_\theta \to v^*$ (constant field).

## 💻 NumPy / PyTorch 구현 검증

### 실험 1: Straight-line Path Visualization

```python
import numpy as np
import matplotlib.pyplot as plt

# x_0, z 쌍 샘플링
np.random.seed(42)
x0 = np.random.randn(20, 2)
z = np.random.randn(20, 2)

# Straight-line path
t_vals = np.linspace(0, 1, 50)
for i in range(min(5, len(x0))):
    path = (1 - t_vals[:, None]) * x0[i] + t_vals[:, None] * z[i]
    plt.plot(path[:, 0], path[:, 1], alpha=0.5, label=f"Sample {i}")

plt.scatter(x0[:, 0], x0[:, 1], c='blue', label='x0', s=50)
plt.scatter(z[:, 0], z[:, 1], c='red', label='z', s=50)
plt.legend()
plt.title("Straight-line Paths in Rectified Flow")
plt.savefig("rectified_flow_paths.png", dpi=100)
print("Paths visualized.")
```

### 실험 2: Velocity Field Matching

```python
import torch
import torch.nn as nn

class VelocityNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 2)  # output velocity (2D)
        )
    
    def forward(self, x, t):
        """v_theta(x, t) predicts instantaneous velocity"""
        t_expanded = t.view(-1, 1).expand(x.shape[0], 1)
        return self.net(torch.cat([x, t_expanded], dim=1))

model = VelocityNet()
x_sample = torch.randn(16, 2)
t_sample = torch.rand(16)
v_pred = model(x_sample, t_sample)
print(f"Velocity shape: {v_pred.shape}")  # [16, 2]
print(f"Mean velocity norm: {v_pred.norm(dim=1).mean():.4f}")
```

### 실험 3: Reflow Simulation

```python
def ode_solver(x0, v_net, t_steps=10):
    """Simple Euler solver for ODE"""
    x = x0.clone()
    for i in range(t_steps - 1):
        t = torch.tensor(i / (t_steps - 1), dtype=x.dtype, device=x.device)
        dt = 1.0 / (t_steps - 1)
        v = v_net(x, t)
        x = x + dt * v
    return x

# Initial coupling (independent)
x0_initial = torch.randn(8, 2)
z_initial = torch.randn(8, 2)
x1_initial = (1 - 0) * x0_initial + 1 * z_initial

# Train 1st rectified flow
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(50):
    t = torch.rand(8)
    xt = (1 - t.unsqueeze(1)) * x0_initial + t.unsqueeze(1) * z_initial
    v_true = z_initial - x0_initial  # Constant in straight-line
    v_pred = model(xt, t)
    loss = ((v_pred - v_true) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Reflow: generate new coupling
x1_refined = ode_solver(x0_initial, model, t_steps=10)
print(f"Epoch 50 velocity loss: {loss.item():.6f}")
print(f"x1 before reflow - z difference (MSE): {((x1_initial - z_initial)**2).mean():.6f}")
print(f"x1 after reflow - z difference (MSE): {((x1_refined - z_initial)**2).mean():.6f}")
```

## 🔗 실전 활용

1. **Stable Diffusion 3 기반**: SD3는 Rectified Flow ODE를 텍스트-이미지 생성의 backbone으로 사용
2. **1-step 생성**: Reflow를 충분히 반복하면, 1-step으로도 고품질 샘플 가능
3. **Explicit coupling**: optimal transport를 고려하여 데이터-노이즈 쌍을 의도적으로 구성 가능
4. **속도-품질**: Straight path는 few-step (2-4) 샘플링에서 우수한 성능

## ⚖️ 가정과 한계

1. **Coupling 선택**: Independent coupling에서 시작하면 reflow가 필요. Optimal transport coupling을 미리 계산하면 더 빠름 (계산 비용 ↑)
2. **Reflow 반복 필요**: 한 번의 학습으로 완벽한 직선 경로는 어려움. 여러 회 reflow 필요 (학습 시간 ↑)
3. **Velocity 상수 가정**: Optimal case에서 velocity는 상수이지만, 실제로는 약간의 곡률 존재 가능
4. **이산화 오차**: ODE solver의 이산화 오차 누적 (higher-order solver 필요)

## 📌 핵심 정리

**Rectified Flow**는 forward process를 $(1-t)x_0 + tz$ 직선으로 정의하고, velocity field $v_\theta$를 학습합니다. **Reflow** 메커니즘을 통해:
- **경로 직선화**: 반복 학습으로 더욱 직선에 가까워짐
- **1-step 가능**: 극한에서 velocity가 상수 → perfect 1-step
- **Optimal Transport 연결**: Monge-Kantorovich 최적해와의 이론적 연결
- **SD3의 기초**: 현대 대규모 생성 모델의 핵심 구조

## 🤔 생각해볼 문제

1. Rectified Flow에서 처음 coupling을 $(x_0, z)$ independent로 시작할 때, 왜 여러 번의 reflow가 필요할까요? 수렴 속도를 예측하는 방법이 있을까요?

   <details>
   <summary>힌트</summary>
   Independent coupling은 데이터 분포와 노이즈 분포를 임의로 연결하므로, 경로가 많이 굽어있습니다. Reflow마다 "직선 경로에 가까운 쌍"으로 재구성되므로, 이론적으로는 exponential convergence가 예상됩니다. 실제 수렴 곡선은 optimal transport distance의 감소 패턴을 따릅니다.
   </details>

2. Reflow 과정에서 ODE solver의 오차가 누적되면, 새로운 coupling $(x_0', z')$이 부정확해질 것 같습니다. 이를 완화하는 방법이 있을까요?

   <details>
   <summary>힌트</summary>
   Higher-order ODE solver (e.g., Runge-Kutta 4) 사용, 또는 adaptive step size로 오차 조절. 또한 여러 경로를 앙상블하여 reflow coupling을 구성하는 방법도 있습니다.
   </details>

3. Optimal transport를 미리 계산한 coupling으로 시작하면 reflow가 불필요해질 것 같습니다. 하지만 왜 실제로는 여전히 velocity matching 학습이 필요할까요?

   <details>
   <summary>힌트</summary>
   Optimal transport map은 deterministic하지만, 그 map을 실제로 $v_\theta$로 정확히 매개변수화하는 것은 별개입니다. 또한 finite sample approximation, discretization, neural network 용량 제약 등으로 인해 정확한 학습이 필요합니다.
   </details>

---

<div align="center">

[◀ 이전](./01-consistency-model.md) | [📚 README](../README.md) | [다음 ▶](./03-flow-matching.md)

</div>
