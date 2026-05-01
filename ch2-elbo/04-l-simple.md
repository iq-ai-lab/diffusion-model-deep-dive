# Ch2-04: $L_{\mathrm{simple}}$ 손실함수 유도

## 🎯 핵심 질문

왜 Ho et al. (2020)은 이론적으로 도출된 가중치를 **제거**하는 것이 실제로 더 좋은 생성 결과를 만드는가? 단순화된 손실 $L_{\mathrm{simple}}$의 근거는 무엇이며, 이것이 SNR 관점에서 어떤 의미인가?

## 🔍 왜 이 $L_{\mathrm{simple}}$인가

VLB 분해로부터 이론적으로 도출된 loss는 복잡한 가중치를 포함한다:

$$L_{t-1} = \frac{\beta_t^2}{2\sigma_t^2 \alpha_t(1-\bar\alpha_t)} \mathbb{E}[\|\epsilon - \epsilon_\theta\|^2]$$

그러나 실무에서는 이 가중치를 제거하고 단순하게:

$$L_{\mathrm{simple}} = \mathbb{E}_{t,x_0,\epsilon}[\|\epsilon - \epsilon_\theta(x_t, t)\|^2]$$

만 사용해도 더 좋은 결과를 얻는다. 이는 직관적으로는 이상하지만, SNR 분석을 통해 설명할 수 있다. 낮은 SNR 단계 (초기 잡음)를 덜 강조하면, 시각적 품질이 향상된다는 경험적 발견이다.

## 📐 수학적 선행 조건

- 이전 장: KL 발산과 noise prediction (Ch2-03)
- 이전 장: 3항 분해 (Ch2-02)
- 신호 대 잡음비 (SNR) 개념
- 가우시안 KL 발산의 폐쇄형
- 확률가중 기댓값의 성질

## 📖 직관적 이해

가우시안 KL로부터 유도된 loss를 들여다보면, 가중치 $w_t \propto \frac{1-\bar\alpha_t}{\bar\alpha_t}$는 **신호가 약한 초기 단계를 더 강조**한다. 하지만 초기 단계에서는 아주 약한 신호를 정확히 예측하기 어렵다. 따라서 이 가중치를 무시하고 모든 시간 단계를 균등하게 취급하면, 신경망이 **고SNR 영역 (시각적으로 중요한 부분)에 더 집중**할 수 있게 된다. 결과적으로 생성된 이미지의 시각적 품질이 높아진다. 이는 이론과 실무의 trade-off를 보여주는 좋은 예이다.

## ✏️ 엄밀한 정의

### 정의 2.9: 가중 Noise Prediction Loss

KL 발산으로부터 유도된 loss:

$$L_{t-1}^{\mathrm{weighted}} = \frac{\beta_t^2}{2\sigma_t^2 \alpha_t(1-\bar\alpha_t)} \mathbb{E}_{q(x_t|x_0)}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

평균화 (시간 단계 균등 샘플링):

$$L_{\mathrm{weighted}} = \mathbb{E}_{t \sim \text{Unif}(1,T)}\left[L_{t-1}^{\mathrm{weighted}}\right]$$

### 정의 2.10: 단순화된 Loss

Ho et al. (2020)이 제안한 단순화:

$$L_{\mathrm{simple}} := \mathbb{E}_{t \sim \text{Unif}(1,T)}\left[\mathbb{E}_{x_0 \sim p(x_0), \epsilon \sim \mathcal{N}(0,I)}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]\right]$$

즉, **모든 시간 단계를 균등한 가중치**로 처리.

### 정의 2.11: SNR (Signal-to-Noise Ratio)

시간 $t$에서의 SNR:

$$\text{SNR}(t) := \frac{\mathbb{E}[\|x_{\mathrm{signal}}\|^2]}{\mathbb{E}[\|x_{\mathrm{noise}}\|^2]} = \frac{\bar\alpha_t}{1-\bar\alpha_t}$$

여기서:
- Signal: $\sqrt{\bar\alpha_t} x_0$
- Noise: $\sqrt{1-\bar\alpha_t} \epsilon$

## 🔬 정리와 증명

### 정리 2.10: 가중치와 SNR의 관계

$$w_t := \frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar\alpha_t)} = \frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar\alpha_t)}$$

는 **역 SNR에 비례**한다:

$$w_t \propto \frac{1-\bar\alpha_t}{\bar\alpha_t} = \frac{1}{\text{SNR}(t)}$$

**증명:**

$$w_t = \frac{(1-\alpha_t)^2}{2\sigma_t^2 \alpha_t(1-\bar\alpha_t)}$$

상수 인수를 무시하면:

$$w_t \sim \frac{1-\bar\alpha_t}{\bar\alpha_t} = \text{SNR}^{-1}(t)$$

따라서 낮은 SNR (초기, 잡음 많음)일수록 가중치가 크다. $\square$

### 정리 2.11: Weighted Loss와 Simple Loss의 비교

$t$를 균등 분포에서 샘플링할 때:

$$\mathbb{E}_t[w_t \cdot \text{MSE}_t] \neq \mathbb{E}_t[\text{MSE}_t]$$

이유:
1. **Weighted loss**: 낮은 SNR (초기)를 강조 → 약한 신호 학습에 포커스
2. **Simple loss**: 모든 단계 균등 → 고 SNR (중간~후기) 학습에 포커스

**경험적 발견**: Simple loss가 시각적 품질 더 우수 (Ho et al. 2020)

**수학적 설명**: 신경망 근사 관점에서, 초기 단계의 약한 신호는:
- 정확한 예측이 어려움
- 최종 결과에 미치는 영향 작음
- 따라서 덜 가중치를 두면 전체 성능 향상

$\square$

### 정리 2.12: Simple Loss의 정당성 (최적성 추론)

Single step reconstruction error:

$$\mathbb{E}_t[\text{dist}(x_{t-1}^{\mathrm{true}}, x_{t-1}^{\mathrm{pred}})] \propto \mathbb{E}_t[\text{MSE}_t^{1/2}]$$

누적 오류 (iterative generation):

$$\text{Total error} = \sum_{t=1}^T \text{dist}_t$$

각 단계의 기여도가 같다고 가정하면:

$$\text{Total} \propto \mathbb{E}_t[\text{MSE}_t] = L_{\mathrm{simple}}$$

따라서 simple loss를 최소화하는 것이 **평균 재구성 오류 최소화**와 대응된다. $\square$

## 💻 NumPy / PyTorch 구현 검증

### 실험 1: 가중치 vs Simple Loss 비교

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

T = 1000
batch_size = 64

# Variance schedule (linear)
betas = torch.linspace(0.0001, 0.02, T)
alphas = 1 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

# Simulate noise prediction errors per time step
mse_per_t = []
weights = []

for t in range(T):
    alpha_t = alphas[t]
    bar_alpha_t = alphas_cumprod[t]
    beta_t = betas[t]
    
    # Simulated MSE (decreasing with t since prediction gets harder for early t)
    mse = 1.0 / (1 + t / 100)  # Arbitrary simulation
    mse_per_t.append(mse)
    
    # Theoretical weight (inverse SNR)
    weight = (1 - bar_alpha_t) / (bar_alpha_t * (beta_t ** 2))
    weights.append(weight)

mse_per_t = np.array(mse_per_t)
weights = np.array(weights)
weights_normalized = weights / weights.sum()  # Normalize

# Weighted loss
weighted_loss = (weights_normalized * mse_per_t).sum()

# Simple loss
simple_loss = mse_per_t.mean()

print(f"Weighted loss: {weighted_loss:.6f}")
print(f"Simple loss: {simple_loss:.6f}")
print(f"Weight emphasis on early steps: {weights_normalized[:10].sum():.4f}")
print(f"Weight emphasis on late steps: {weights_normalized[-10:].sum():.4f}")

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(weights_normalized, label='Normalized weight $w_t$', color='red', alpha=0.7)
ax1.axhline(y=1/T, color='blue', linestyle='--', label=f'Simple loss (uniform 1/{T})')
ax1.set_xlabel('Time step t')
ax1.set_ylabel('Weight')
ax1.set_yscale('log')
ax1.legend()
ax1.set_title('Loss Weight Distribution')
ax1.grid(True, alpha=0.3)

# Cumulative contribution
cum_weighted = np.cumsum(weights_normalized)
cum_simple = np.linspace(0, 1, T)
ax2.plot(cum_weighted, label='Weighted loss cumulative', color='red')
ax2.plot(cum_simple, label='Simple loss cumulative', color='blue')
ax2.set_xlabel('Time step t')
ax2.set_ylabel('Cumulative weight')
ax2.legend()
ax2.set_title('Cumulative Weight Distribution')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/loss_comparison.png', dpi=100)
print("✓ Loss comparison plot saved")
```

### 실험 2: SNR-기반 분석

```python
# SNR 계산
snr = alphas_cumprod / (1 - alphas_cumprod)
snr_db = 10 * np.log10(snr + 1e-8)

# 역 SNR와 가중치의 관계 검증
inverse_snr = (1 - alphas_cumprod) / (alphas_cumprod + 1e-8)
weight_snr_based = inverse_snr / (betas.numpy() ** 2 * alphas.numpy())

# 정규화
weight_snr_normalized = weight_snr_based / weight_snr_based.sum()

# 비교
correlation = np.corrcoef(weights_normalized, weight_snr_normalized)[0, 1]
print(f"Correlation between theoretical weight and SNR-based weight: {correlation:.6f}")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(snr_db, label='SNR (dB)', color='green', linewidth=2)
ax2 = ax.twinx()
ax2.plot(weights_normalized, label='Normalized weight', color='red', alpha=0.7)
ax.set_xlabel('Time step t')
ax.set_ylabel('SNR (dB)', color='green')
ax2.set_ylabel('Weight', color='red')
ax.legend(loc='upper left')
ax2.legend(loc='upper right')
ax.set_title('SNR vs Loss Weight Over Time')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/tmp/snr_weight.png', dpi=100)
print("✓ SNR-weight plot saved")
```

### 실험 3: Simple Loss 구현 (실제 DDPM)

```python
# PyTorch implementation of L_simple
def simple_loss(model, x0, t, alphas_cumprod):
    """
    Simplified DDPM loss: E_t,x0,eps[||eps - eps_theta(x_t, t)||^2]
    """
    batch_size = x0.shape[0]
    
    # Sample noise
    epsilon = torch.randn_like(x0)
    
    # Forward diffusion
    sqrt_alpha_cumprod_t = torch.sqrt(alphas_cumprod[t]).view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - alphas_cumprod[t]).view(-1, 1, 1, 1)
    
    x_t = sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * epsilon
    
    # Model prediction
    epsilon_pred = model(x_t, t)
    
    # Simple MSE loss (no weights)
    loss = ((epsilon_pred - epsilon) ** 2).mean()
    
    return loss

# Example usage:
# batch = next(dataloader)
# x0 = batch['images']
# t = torch.randint(0, T, (batch_size,))
# loss = simple_loss(unet_model, x0, t, alphas_cumprod)
# optimizer.zero_grad()
# loss.backward()
# optimizer.step()

print("✓ Simple loss implementation template provided")
```

## 🔗 실전 활용

1. **DDPM 구현**: 거의 모든 open-source 구현이 $L_{\mathrm{simple}}$을 사용한다.
2. **하이퍼파라미터 튜닝**: Simple loss로 학습하면 variance schedule 선택이 덜 critical하다.
3. **다중 스케일 생성**: 해상도별로 다른 가중치를 사용할 수 있지만, simple loss도 충분히 좋다.

## ⚖️ 가정과 한계

| 가정 / 한계 | 설명 | 영향 |
|-----------|------|------|
| 균등 시간 샘플링 | $t \sim \text{Unif}(1,T)$ | Simple loss 유도 기반 |
| 신경망 표현력 | 모든 단계를 충분히 학습 가능 | 실제로는 근사 |
| 누적 오류 균등성 | 각 단계의 오류 영향이 같음 | 초기/후기 영향 다를 수 있음 |
| 시각적 품질 기준 | Perceptual quality가 SNR과 대응 | 주관적 |
| 초기 단계 무시 | Low SNR 단계 예측 부정확 | 원리적 정당화 |

## 📌 핵심 정리

$$\boxed{L_{\mathrm{simple}} = \mathbb{E}_{t,x_0,\epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]}$$

**이론과 실무의 차이:**

| 관점 | 이론 (VLB) | 실무 (Ho et al.) |
|------|-----------|-----------------|
| Loss | 가중 MSE | 단순 MSE |
| 가중치 | $w_t \propto 1/\text{SNR}$ | 없음 (균등) |
| 근거 | 확률론 | 경험적 |
| 결과 | 이론적 bound | 더 좋은 시각 품질 |
| 구현 | 복잡 | 간단 |

## 🤔 생각해볼 문제

### 문제 1: Simple Loss가 정말 더 나은가?

실제 실험에서 weighted loss vs simple loss를 비교하면 어떨까?

<details><summary>해설</summary>

Ho et al. (2020) DDPM 논문에서 명시적으로 비교:

**Weighted loss**: 수렴 느림, 초기 단계에 stuck
**Simple loss**: 빠른 수렴, 더 나은 FID 점수

직관: 약한 신호 정확 예측 어려움 → 가중치 제거 → 포커스 전환 → 성능 향상

</details>

### 문제 2: 최적 가중치는 따로 있을까?

Simple (1.0) 과 full weight 사이의 최적 값이 있을까?

<details><summary>해설</summary>

이론적으로는 **full weight가 최적** (KL 기반 유도)

실무적으로는:

$$w_t = (1-\bar\alpha_t) / \bar\alpha_t)^{\text{power}}$$

power ∈ [0, 1] 사이에서 그리드 서치 가능

하지만 대부분 power=0 (simple loss)로 설정

</details>

### 문제 3: 다른 loss가 가능한가?

Weighted와 simple 외에 다른 선택지는?

<details><summary>해설</summary>

가능한 선택:
1. **Huber loss**: 이상값에 robust
2. **Perceptual loss**: 시각적 거리 사용 (LPIPS 등)
3. **Weighted by schedule**: SNR 기반 custom weight
4. **Min-SNR clipping** (Improved DDPM, Ch2-05): $w_t = \min(\text{SNR}(t), 5)$

최근 trend는 SNR 기반 clipping 사용

</details>

---

<div align="center">

[◀ 이전](./03-noise-prediction.md) | [📚 README](../README.md) | [다음 ▶](./05-improved-ddpm.md)

</div>
