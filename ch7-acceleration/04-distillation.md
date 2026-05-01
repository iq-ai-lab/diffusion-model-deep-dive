# 7.4 Distillation for Diffusion (Salimans & Ho 2022, Sauer 2024)

## 🎯 핵심 질문

이미 학습된 "좋은" diffusion model이 있다면, 이를 바탕으로 **더 빠른 버전을 만들 수 있을까?** Distillation은 teacher model의 다중 스텝 샘플링을 student가 한 스텝으로 emulate하도록 학습시켜, 샘플링 속도를 극적으로 가속합니다.

## 🔍 왜 Distillation인가?

1. **Teacher 활용**: 이미 좋은 teacher model 있음 → 새로 학습 불필요
2. **Progressive 감소**: 1024 step → 512 → 256 → ... → 4 step 점진적 가속
3. **Knowledge transfer**: Teacher의 knowledge를 student에게 이전
4. **Production ready**: SDXL-Turbo, SD Lightning 등 실제 사용 모델

## 📐 수학적 선행 조건

- **Diffusion sampling**: $x_{t-\Delta t} = \alpha x_t + \beta \epsilon_\theta(x_t, t)$
- **ODE/SDE solver**: DDIM, Euler, RK45 등
- **KL divergence / Wasserstein distance**: 분포 간 거리
- **GAN loss**: Discriminator 기반 adversarial training

## 📖 직관적 이해

**Progressive Distillation** (Salimans & Ho 2022):

1. **Teacher (1024 step)**: 매우 느리지만 좋은 샘플
2. **Student v1 (512 step)**: Teacher의 2-step을 1-step으로 emulate
   - Teacher로 2-step 진행: $x_t \to x_{\Delta t}$ (2번)
   - Student는 한 번에: $x_t \to x_{\Delta t}$ (1번)
   - Student output이 Teacher output과 가까우도록 학습
3. **Student v2 (256 step)**: Student v1을 teacher로, 다시 2→1 step 압축
4. 반복 → 최종 4 step 가능

**Adversarial Diffusion Distillation (ADD)** (Sauer 2024):

Consistency loss (MSE) 대신 adversarial loss 추가:
- **Generator**: Student diffusion model
- **Discriminator**: "이 샘플이 teacher의 것인가?"를 판정
- GAN-style training으로 더 빠른 수렴, 더 좋은 품질

## ✏️ 엄밀한 정의

**정의 (Progressive Distillation)**

Timestep set $\{t_1, \ldots, t_n\}$ (내림차순)에 대해, teacher $\epsilon_\theta^{\text{teacher}}$와 student $\epsilon_\phi^{\text{student}}$를 다음과 같이 정의:

$$x_{t_{i+1}} = \text{DDIM}_{\text{teacher}}(x_{t_i}, t_i, t_{i+1})$$
(Teacher 2-step)

$$\tilde{x}_{t_{i+1}} = \text{DDIM}_{\text{student}}(x_{t_i}, t_i, t_{i+1})$$
(Student 1-step)

Student loss:
$$\mathcal{L}_{\text{student}} = \mathbb{E}_{x_0, i} \left\| \tilde{x}_{t_{i+1}} - x_{t_{i+1}} \right\|^2$$

**정의 (Adversarial Diffusion Distillation, ADD)**

Generator loss:
$$\mathcal{L}_G = \mathbb{E}_{x_0} \left[ -D(\tilde{x}_{\text{student}}) \right] + \lambda \left\| \tilde{x}_{\text{student}} - x_{\text{teacher}} \right\|^2$$

Discriminator loss:
$$\mathcal{L}_D = \mathbb{E}_{x_0} \left[ D(\tilde{x}_{\text{student}}) - D(x_{\text{teacher}}) \right]$$

여기서 $D$는 판별자, $\lambda$는 consistency term의 가중치.

**정의 (Multi-stage Distillation)**

$n$-step teacher를 $n/2$-step student로 가속:

$$\mathcal{L}^{(k)} = \mathbb{E} \left\| \epsilon_{\phi}^{(k)} - \text{RunODE}(\epsilon_{\theta}^{(k-1)}, 2\text{-step}) \right\|^2$$

Stage $k$마다 반복하여 최종 4 step까지 감소.

## 🔬 정리와 증명

**정리 (Progressive Distillation Convergence)**

만약 teacher가 optimal diffusion model이고 (무한 데이터, perfect training), student의 model capacity가 충분하다면, progressive distillation을 거쳐도 student의 최종 샘플 분포는 teacher와 거의 같습니다.

*증명 스케치* ($\square$)

각 stage $k$에서 student는:
$$\epsilon_\phi^{(k)}(x, t) \approx \mathbb{E}[\epsilon_\theta^{(k-1)}(x, t) | x_\phi^{\text{prev}}]$$

를 최소 제곱으로 학습합니다. 즉, student는 teacher의 조건부 expectation을 근사합니다.

DDIM의 deterministic 성질에 의해, 이 근사가 정확하면 (즉, $\epsilon_\phi^{(k)} \approx \epsilon_\theta^{(k-1)}$), ODE trajectory도 같아집니다. 따라서:
$$x_\phi^{(k)} \text{ follows similar trajectory as } x_\theta^{(k-1)}$$

반복 적용하면, 초기 teacher $x_\theta^{(0)}$의 샘플링 경로 위의 모든 step을 절반으로 줄인 student도 같은 경로를 따릅니다.

## 💻 NumPy / PyTorch 구현 검증

### 실험 1: Teacher DDIM Solver

```python
import torch
import torch.nn as nn

def ddim_step(x_t, eps_pred, t, t_prev, alphas):
    """DDIM single step: x_t -> x_prev"""
    alpha_t = alphas[t]
    alpha_prev = alphas[t_prev]
    
    sigma = 0.0  # Deterministic
    sqrt_1_alpha = (1 - alpha_t) ** 0.5
    sqrt_1_alpha_prev = (1 - alpha_prev) ** 0.5
    
    x_prev = (x_t - sqrt_1_alpha * eps_pred) / (alpha_t ** 0.5)
    x_prev = x_prev * (alpha_prev ** 0.5) + sqrt_1_alpha_prev * eps_pred
    return x_prev

class TeacherDiffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 2)
        )
        self.alphas = torch.linspace(0.9, 0.1, 1024)
    
    def forward(self, x, t):
        t_norm = t / 1000.0
        return self.net(torch.cat([x, t_norm.unsqueeze(-1)], dim=1))

teacher = TeacherDiffusion()
print("Teacher model created.")
```

### 실험 2: Student Distillation Loss

```python
class StudentDiffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )
    
    def forward(self, x, t):
        t_norm = t / 1000.0
        return self.net(torch.cat([x, t_norm.unsqueeze(-1)], dim=1))

student = StudentDiffusion()

def progressive_distillation_step(x_init, teacher, student, t_steps_teacher, t_steps_student):
    """
    Teacher: 2 steps (t_i -> t_mid -> t_i+1)
    Student: 1 step (t_i -> t_i+1)
    """
    batch_size = x_init.shape[0]
    
    # Teacher 2-step
    x_mid = x_init.clone()
    t_curr = torch.tensor(500.0, dtype=x_init.dtype)
    for _ in range(2):  # Dummy 2 iterations
        eps_teacher = teacher(x_mid, t_curr.expand(batch_size))
        x_mid = x_mid - 0.01 * eps_teacher  # Simplified update
    
    # Student 1-step
    t_init = torch.tensor(500.0, dtype=x_init.dtype)
    eps_student = student(x_init, t_init.expand(batch_size))
    x_final_student = x_init - 0.02 * eps_student
    
    # Distillation loss
    loss = ((x_final_student - x_mid) ** 2).mean()
    return loss

x_sample = torch.randn(8, 2)
loss = progressive_distillation_step(x_sample, teacher, student, 1024, 512)
print(f"Distillation loss: {loss.item():.6f}")
```

### 실험 3: Multi-stage Training Loop

```python
optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)

for epoch in range(50):
    x_batch = torch.randn(16, 2)
    
    # Progressive distillation
    loss = progressive_distillation_step(x_batch, teacher, student, 1024, 512)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: Distillation Loss = {loss.item():.6f}")

# At the end of stage 1, student becomes new teacher for stage 2
print("Stage 1 distillation complete. Ready for progressive compression.")
```

## 🔗 실전 활용

1. **SDXL-Turbo** (Sauer et al. 2024): SDXL을 1-step으로 가속화. Latent diffusion + ADD로 고품질 유지
2. **SD Lightning**: Stability AI의 다단계 distillation. 4-step까지 실시간 생성 가능
3. **Real-time inference**: Mobile, edge device에서 이미지/비디오 생성 가능
4. **Quality-speed tradeoff**: 1-step (ultra-fast), 2-4 step (balanced), multi-step (high-quality) 옵션 제공

## ⚖️ 가정과 한계

1. **Teacher 필수**: Scratch distillation 불가능. 미리 학습된 좋은 teacher 필요
2. **Knowledge loss**: 각 stage마다 정보 손실 발생. 너무 많은 stage면 최종 품질 저하
3. **Capacity constraint**: Student model의 capacity가 부족하면 teacher를 완전히 mimick 불가능
4. **Domain-specific**: Teacher가 좋지 않으면 student도 나쁨 (garbage in, garbage out)

## 📌 핵심 정리

**Distillation**은 이미 학습된 teacher diffusion model을 바탕으로, student model이 fewer steps에서 같은 품질을 달성하도록 가속합니다:
- **Progressive distillation**: 1024→512→256→...→4 step 점진적 감소
- **ADD (Adversarial)**: GAN-style loss로 더 빠른 수렴, 더 좋은 품질
- **Production-ready**: SDXL-Turbo, SD Lightning 등 실제 서비스에 적용
- **Quality-speed fusion**: 극적인 속도 향상 (50배+)과 용인 가능한 품질 손실의 균형

## 🤔 생각해볼 문제

1. Progressive distillation에서 teacher를 2-step으로 실행하고 student를 1-step으로 하는 이유가 무엇일까요? 더 큰 gap (예: 4→1)은 안 될까요?

   <details>
   <summary>힌트</summary>
   너무 큰 gap은 student가 배우기 어려워집니다. 각 stage마다 2배 감소는 안정적인 학습과 품질 유지의 균형점입니다. 더 큰 gap은 가능하지만, 추가 stage가 필요할 수 있습니다.
   </details>

2. Adversarial Diffusion Distillation (ADD)에서 discriminator의 역할은 무엇이며, 왜 이것이 MSE loss 만으로는 부족할까요?

   <details>
   <summary>힌트</summary>
   MSE loss는 pixel-level에서만 matching을 강제합니다. GAN discriminator는 high-level perceptual similarity를 강제하여, 더 자연스러운 샘플을 생성합니다. 이는 LPIPS 같은 perceptual metric과 유사합니다.
   </details>

3. Distillation 후 student model이 teacher보다 더 좋은 샘플을 생성할 수 있을까요? 즉, teacher의 knowledge를 넘어설 수 있을까요?

   <details>
   <summary>힌트</summary>
   이론적으로는 어렵습니다. Distillation은 knowledge transfer이므로, student가 teacher를 초과하려면 추가 학습이 필요합니다. 다만 student가 특정 domain에 overfit되어 있다면, generalization에서는 더 좋을 수 있습니다.
   </details>

---

<div align="center">

[◀ 이전](./03-flow-matching.md) | [📚 README](../README.md) | [다음 ▶](../README.md)

</div>
