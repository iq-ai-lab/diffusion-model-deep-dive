# Ch2-05: Improved DDPM (Nichol & Dhariwal 2021)

## 🎯 핵심 질문

DDPM의 성공 이후, Improved DDPM은 어떤 세 가지 핵심 개선을 도입했는가? 각각 (1) 코사인 schedule, (2) 학습 가능한 분산, (3) 하이브리드 loss가 실제로 얼마나 중요한가?

## 🔍 왜 이 개선들인가

Ho et al. (2020)의 DDPM은 단순했지만, 생성 속도가 느렸고 로그 우도가 낮았다. Nichol & Dhariwal (2021)의 Improved DDPM은 세 가지 관찰에서 출발한다: (1) variance schedule의 선택이 중요 (cosine schedule이 linear보다 우수), (2) 분산을 고정하는 대신 학습하면 로그 우도 향상, (3) simple loss와 weighted loss를 혼합하면 두 목표 (생성 품질 + 로그 우도)의 trade-off를 개선할 수 있다. 이 장은 최신 diffusion model 구현의 표준을 정의한다.

## 📐 수학적 선행 조건

- 이전 장: Noise prediction parameterization (Ch2-03)
- 이전 장: Simple loss (Ch2-04)
- Variance schedule의 개념 (Ch1-02)
- 로그 우도 (log-likelihood) 평가
- 가우시안 분산의 학습 파라미터화

## 📖 직관적 이해

DDPM의 핵심 한계는 세 가지다: (1) 초기 단계에서 분산이 너무 커서 불안정, (2) 역방향 분산을 고정값으로 두면 이상적이지 않음, (3) simple loss는 샘플 품질에는 좋지만 로그 우도에는 약함. Improved DDPM은 이들을 해결한다. 코사인 schedule은 초기에 천천히, 후기에 빠르게 증가하여 신호 소실을 균형있게 제어한다. 학습 가능한 분산은 신경망이 각 단계에서 "얼마나 확신하는지"를 나타낸다. 하이브리드 loss는 두 가지 목표를 동시에 달성한다.

## ✏️ 엄밀한 정의

### 정의 2.12: 코사인 분산 스케줄

$$\bar\alpha_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\left(\frac{\pi t / T + s}{1 + s} \cdot \frac{\pi}{2}\right)^2$$

여기서 $s$는 스몰 offset (보통 0.008)으로, 초기에 너무 작아지는 것을 방지한다.

**특성:**
- 초기 ($t=0$): $\bar\alpha_0 \approx 1$ (신호 대부분 유지)
- 중기 ($t=T/2$): $\bar\alpha_{T/2}$ 천천히 감소
- 후기 ($t=T$): $\bar\alpha_T \approx 0$ (완전 잡음)

### 정의 2.13: 학습 가능한 분산 (Learned Variance)

역방향 분산을 다음과 같이 parameterize:

$$\Sigma_\theta(t) = \exp(v \log \beta_t + (1-v) \log \tilde\beta_t)$$

여기서:
- $\beta_t = 1 - \alpha_t$ : 순방향 분산
- $\tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t} \beta_t$ : posterior 분산
- $v \in [0,1]$ : 신경망이 학습하는 스칼라 (per time step)

따라서 $\Sigma_\theta$는 두 값 사이를 보간한다.

### 정의 2.14: 하이브리드 손실

$$L_{\mathrm{hybrid}} = L_{\mathrm{simple}} + \lambda L_{\mathrm{vlb}}$$

여기서:
- $L_{\mathrm{simple}} = \mathbb{E}_{t,x_0,\epsilon}[\|\epsilon - \epsilon_\theta\|^2]$
- $L_{\mathrm{vlb}} = $ 분산 관련 항 포함한 weighted loss
- $\lambda = 0.001$ : 믹싱 계수

## 🔬 정리와 증명

### 정리 2.13: 코사인 스케줄의 장점

선형 스케줄 대비, 코사인 스케줄은:

1. **초기 안정성**: 초기 $\bar\alpha_0 \approx 1$이므로 $x_1 \approx x_0$ (안정적 시작)
2. **균등한 SNR 분포**: $\text{SNR}(t)$ 감소가 더 균등함
3. **신호-잡음 균형**: 초기에 신호 유지, 중기에 gradual 감소

**수학적 형태:**

선형: $\bar\alpha_t = 1 - \frac{t}{T} \beta_{\max}$ (가파른 초기 감소)

코사인: $f(t) = \cos^2(\cdot)$ (부드러운 감소)

코사인이 초기에 더 완만하므로 학습 안정성 향상. $\square$

### 정리 2.14: 학습 가능한 분산의 KL 형태

분산 매개변수 $v_t$가 추가되면, KL 항은:

$$\mathrm{KL}(q(x_{t-1}|x_t, x_0) \| p_\theta(x_{t-1}|x_t)) = \frac{1}{2v_t}[\log\frac{\Sigma_\theta}{\tilde\beta_t} + \frac{\tilde\beta_t}{\Sigma_\theta} - 1]$$

이를 최소화하기 위해 신경망은:

$$v_t \approx \log \frac{\Sigma_\theta}{\tilde\beta_t}$$

학습 과정:
1. 초기: $v_t \approx 0$ (작은 분산 학습)
2. 후기: $v_t \approx 1$ (큰 분산 학습)

직관: 초기 단계에서는 분산 작게 (신호 많음), 후기에는 크게 (잡음 많음) 학습. $\square$

### 정리 2.15: 하이브리드 손실의 효과

$$L_{\mathrm{hybrid}} = L_{\mathrm{simple}} + \lambda L_{\mathrm{vlb}}, \quad \lambda = 0.001$$

- $L_{\mathrm{simple}}$ (주요, 99.9%): 샘플 품질 최적화
- $L_{\mathrm{vlb}}$ (보조, 0.1%): 로그 우도 개선

결과:
- **NLL (Negative Log-Likelihood)**: 향상 (CIFAR-10에서 3.49 bits/dim)
- **FID**: 유지 또는 소폭 향상
- **IS (Inception Score)**: 향상

따라서 두 목표를 모두 달성 가능. $\square$

## 💻 NumPy / PyTorch 구현 검증

### 실험 1: 코사인 vs 선형 스케줄 비교

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

T = 1000

# Linear schedule
betas_linear = torch.linspace(0.0001, 0.02, T)
alphas_linear = 1 - betas_linear
alphas_cumprod_linear = torch.cumprod(alphas_linear, dim=0)

# Cosine schedule
def cosine_schedule(t, T, s=0.008):
    return torch.cos((torch.arange(t) / T + s) / (1 + s) * np.pi / 2) ** 2

alphas_cumprod_cosine = cosine_schedule(T, T)

# SNR comparison
snr_linear = alphas_cumprod_linear / (1 - alphas_cumprod_linear)
snr_cosine = alphas_cumprod_cosine / (1 - alphas_cumprod_cosine)

snr_linear_db = 10 * torch.log10(snr_linear + 1e-8)
snr_cosine_db = 10 * torch.log10(snr_cosine + 1e-8)

# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Alpha cumulative product
axes[0, 0].plot(alphas_cumprod_linear, label='Linear', linewidth=2)
axes[0, 0].plot(alphas_cumprod_cosine, label='Cosine', linewidth=2, linestyle='--')
axes[0, 0].set_xlabel('Time step t')
axes[0, 0].set_ylabel(r'$\bar{\alpha}_t$')
axes[0, 0].set_title('Cumulative Product of Alpha')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# SNR (linear scale)
axes[0, 1].semilogy(snr_linear, label='Linear', linewidth=2)
axes[0, 1].semilogy(snr_cosine, label='Cosine', linewidth=2, linestyle='--')
axes[0, 1].set_xlabel('Time step t')
axes[0, 1].set_ylabel('SNR')
axes[0, 1].set_title('Signal-to-Noise Ratio (Log Scale)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# SNR (dB)
axes[1, 0].plot(snr_linear_db, label='Linear', linewidth=2)
axes[1, 0].plot(snr_cosine_db, label='Cosine', linewidth=2, linestyle='--')
axes[1, 0].set_xlabel('Time step t')
axes[1, 0].set_ylabel('SNR (dB)')
axes[1, 0].set_title('Signal-to-Noise Ratio (dB)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Early step detail (0-100)
axes[1, 1].plot(snr_linear_db[:100], label='Linear', linewidth=2)
axes[1, 1].plot(snr_cosine_db[:100], label='Cosine', linewidth=2, linestyle='--')
axes[1, 1].set_xlabel('Time step t')
axes[1, 1].set_ylabel('SNR (dB)')
axes[1, 1].set_title('Early Steps Detail (t=0-100)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/schedule_comparison.png', dpi=100)
print("✓ Schedule comparison plot saved")

print(f"Linear - SNR at t=0: {snr_linear_db[0]:.2f} dB")
print(f"Cosine - SNR at t=0: {snr_cosine_db[0]:.2f} dB")
print(f"Linear - SNR at t=T/2: {snr_linear_db[T//2]:.2f} dB")
print(f"Cosine - SNR at t=T/2: {snr_cosine_db[T//2]:.2f} dB")
```

### 실험 2: 학습 가능한 분산 구현

```python
class ImprovedDDPM(torch.nn.Module):
    def __init__(self, model, T):
        super().__init__()
        self.model = model
        self.T = T
        
        # Learned variance parameters (per time step)
        self.log_var = torch.nn.Parameter(
            torch.zeros(T), requires_grad=True
        )
    
    def forward_diffusion(self, x0, t, alphas_cumprod, betas):
        """Forward process with learned variance option"""
        alpha_t = alphas_cumprod[t]
        beta_t = betas[t]
        alpha_t_prev = alphas_cumprod[t-1] if t > 0 else 1.0
        
        # Posterior variance bounds
        var_t = (1 - alpha_t_prev) / (1 - alpha_t) * beta_t
        
        return var_t
    
    def reverse_process(self, x_t, t, alphas_cumprod, betas):
        """
        Reverse process with learned variance.
        
        Output: (mean, log_variance, epsilon)
        """
        # Model predicts epsilon
        epsilon_pred = self.model(x_t, t)
        
        # Compute mean using noise prediction
        alpha_t = alphas_cumprod[t]
        beta_t = betas[t]
        
        mean = (x_t - beta_t / torch.sqrt(1 - alpha_t) * epsilon_pred) / torch.sqrt(alpha_t)
        
        # Learned variance (interpolation)
        v_t = self.log_var[t].sigmoid()  # v in [0, 1]
        
        beta_t = betas[t]
        beta_t_tilde = (1 - alphas_cumprod[t-1]) / (1 - alphas_cumprod[t]) * beta_t
        
        log_var = v_t * torch.log(beta_t) + (1 - v_t) * torch.log(beta_t_tilde)
        
        return mean, log_var, epsilon_pred
    
    def compute_loss(self, x0, t, alphas_cumprod, betas, lambda_vlb=0.001):
        """Hybrid loss: L_simple + lambda * L_vlb"""
        # Forward diffusion
        epsilon = torch.randn_like(x0)
        x_t = torch.sqrt(alphas_cumprod[t]) * x0 + \
              torch.sqrt(1 - alphas_cumprod[t]) * epsilon
        
        # Reverse prediction
        mean_pred, log_var, epsilon_pred = self.reverse_process(x_t, t, alphas_cumprod, betas)
        
        # L_simple (main loss)
        l_simple = ((epsilon_pred - epsilon) ** 2).mean()
        
        # L_vlb (auxiliary loss) - variance part
        # Simplified: variance KL
        l_vlb = 0.5 * torch.exp(-log_var) * ((epsilon_pred - epsilon) ** 2).mean()
        
        # Hybrid loss
        loss = l_simple + lambda_vlb * l_vlb
        
        return loss, l_simple, l_vlb

# Usage example:
# model = ImprovedDDPM(unet, T=1000)
# loss, l_simple, l_vlb = model.compute_loss(batch, t, alphas_cumprod, betas)

print("✓ Improved DDPM implementation provided")
```

### 실험 3: 하이브리드 손실의 효과 시뮬레이션

```python
# Simulate loss curves over training
num_steps = 10000
lambda_vlb_values = [0.0, 0.001, 0.01, 0.1]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for lambda_vlb in lambda_vlb_values:
    l_simple_curve = []
    l_vlb_curve = []
    l_total_curve = []
    
    for step in range(num_steps):
        # Simulate loss decay
        l_simple = 1.0 * np.exp(-step / 3000)
        l_vlb = 0.5 * np.exp(-step / 5000)
        l_total = l_simple + lambda_vlb * l_vlb
        
        l_simple_curve.append(l_simple)
        l_vlb_curve.append(l_vlb)
        l_total_curve.append(l_total)
    
    axes[0].semilogy(l_total_curve, label=f'λ={lambda_vlb}', linewidth=2)
    axes[1].semilogy(l_simple_curve, label=f'L_simple', linewidth=2, alpha=0.7)

axes[0].set_xlabel('Training step')
axes[0].set_ylabel('Total loss')
axes[0].set_title('Hybrid Loss with Different λ')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel('Training step')
axes[1].set_ylabel('L_simple')
axes[1].set_title('L_simple Component (λ=0.001)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/hybrid_loss.png', dpi=100)
print("✓ Hybrid loss comparison saved")
```

## 🔗 실전 활용

1. **최신 구현**: 대부분의 2021년 이후 Diffusion Model (Latent Diffusion, Imagen 등)이 이 세 기법을 사용한다.
2. **로그 우도 개선**: 생성 품질과 우도를 동시에 최적화할 수 있다.
3. **Variance 스케줄**: 코사인은 업계 표준이 되었다.

## ⚖️ 가정과 한계

| 가정 / 한계 | 설명 | 영향 |
|-----------|------|------|
| 코사인 보편성 | 모든 데이터셋에 최적? | 실제로는 task 의존적 |
| 학습 가능 분산 | 신경망이 분산을 적절히 학습 | 과잉/과소 학습 가능 |
| 하이브리드 계수 | λ=0.001 고정 | 데이터셋마다 다를 수 있음 |
| 분산 범위 | $\beta_t$와 $\tilde\beta_t$ 사이만 | 더 큰 범위 불가 |
| 계산 비용 | 약간 증가 (분산 계산) | 무시할 수준 |

## 📌 핵심 정리

$$\boxed{\text{Improved DDPM} = \text{Cosine Schedule} + \text{Learned Variance} + \text{Hybrid Loss}}$$

| 개선 | 수식 | 효과 | 중요도 |
|------|------|------|--------|
| Cosine schedule | $\bar\alpha_t = \cos^2(\cdot)$ | SNR 균등화, 안정성 | 높음 |
| Learned variance | $\Sigma_\theta = \exp(v\log\beta_t + \ldots)$ | NLL 개선 | 중간 |
| Hybrid loss | $L = L_s + 0.001 L_v$ | 품질 + 우도 | 중간 |

## 🤔 생각해볼 문제

### 문제 1: 코사인 스케줄은 데이터 독립적인가?

이미지, 오디오, 텍스트 데이터에 모두 같은 코사인 스케줄을 쓸 수 있는가?

<details><summary>해설</summary>

**이론적**: 코사인은 SNR 분포를 균등화하므로 보편적

**실무적**: 
- 이미지: 코사인 표준
- 오디오/음성: 약간 다른 offset 필요 (더 빠른 decay)
- 텍스트: tokenize 후 임베딩이므로 다를 수 있음

따라서 기본은 코사인이지만, task별 fine-tuning 가능.

</details>

### 문제 2: 학습 가능한 분산이 정말 필요한가?

고정 분산으로도 충분하지 않을까?

<details><summary>해설</summary>

비교 (Nichol & Dhariwal 2021):
- **고정 분산**: FID ~3.0, NLL ~3.75 bits/dim
- **학습 분산**: FID ~2.9, NLL ~3.49 bits/dim

따라서:
- FID (샘플 품질): 소폭 향상
- NLL: **유의미하게 향상** (3.75 → 3.49)

로그 우도가 중요하면 필수, 샘플 품질만 중요하면 선택.

</details>

### 문제 3: λ=0.001이 최적인가?

하이퍼파라미터 tuning으로 더 나은 값이 있을까?

<details><summary>해설</summary>

Nichol & Dhariwal 실험:
- λ=0: Simple loss만 → FID 최고, NLL 낮음
- λ=0.001: 균형 (기본값)
- λ=0.01: NLL 더 좋음, FID 소폭 하락
- λ=0.1: NLL 최고, FID 명백히 악화

따라서 **task 목표에 따라**:
- FID 중심: λ=0 또는 매우 작음
- NLL 중심: λ=0.01 ~ 0.1
- 균형: λ=0.001 (기본값)

</details>

---

<div align="center">

[◀ 이전](./04-l-simple.md) | [📚 README](../README.md) | [다음 ▶](../ch3-score-sde/01-score-langevin.md)

</div>
