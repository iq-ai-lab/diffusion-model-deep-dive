# 3. NCSN: Noise Conditional Score Networks

## 🎯 핵심 질문
Song & Ermon (2019)의 NCSN은 왜 여러 노이즈 스케일을 사용하는가? Multi-scale score와 annealed Langevin sampling이 어떻게 저확률 영역 추정을 개선하는가?

## 🔍 왜 Multi-scale인가?

단일 노이즈 스케일 σ로 score를 학습하면, 저확률 영역(tail)에서는 충분한 데이터가 없어 score 추정이 매우 부정확하다. 큰 σ부터 시작해 점진적으로 줄이면, 각 스케일에서 더 안정적인 추정을 할 수 있다. 이를 **annealed Langevin sampling**이라 한다.

## 📐 수학적 선행 조건
- Multi-task learning: 조건부 신경망 $f(x; c)$
- Embedding: 스칼라/벡터를 고차원 표현으로 변환
- Spectral normalization: 신경망 안정성
- Log-likelihood 개념: $\log p(x) = \int \nabla \log p_t(x) dt$ (연속 시간)

## 📖 직관적 이해

큰 산 정상에서 내려올 때를 생각하자:
1. **큰 σ**: 안개가 짙어서 방향을 대략적으로만 안다. 하지만 모든 인물이 보인다 (mode가 구별됨).
2. **중간 σ**: 안개가 걷히기 시작, 더 세밀한 구조가 보인다.
3. **작은 σ**: 선명하지만 좁은 계곡에 빠질 위험. 큰 σ에서의 trajectory로 안내받는다.

각 σ마다 신경망을 따로 훈련하거나, 조건부 신경망으로 모두를 학습한다.

## ✏️ 엄밀한 정의

**정의 3.5 (Multi-scale Noise Levels)**  
음이 아닌 정수 $L$과 $\sigma_1 > \sigma_2 > \cdots > \sigma_L > 0$에 대해, noise schedule을 정의한다.

**정의 3.6 (Noise Conditional Score Network)**  
$$s_\theta(x, \sigma) : \mathbb{R}^d \times \mathbb{R}_{>0} \to \mathbb{R}^d$$
는 입력 $x$와 노이즈 스케일 $\sigma$에 조건화된 score 함수다. $\theta$는 신경망의 가중치.

**정의 3.7 (NCSN 훈련 목적 함수)**  
$$L_{\text{NCSN}}(\theta) = \sum_{i=1}^{L} \lambda_i \mathbb{E}_{x \sim p, \epsilon \sim \mathcal{N}} \left[\left\| s_\theta(x + \sigma_i \epsilon, \sigma_i) - \left(-\frac{\epsilon}{\sigma_i}\right) \right\|^2 \right]$$
여기서 $\lambda_i$는 가중치(보통 $\sigma_i^2$).

## 🔬 정리와 증명

**정리 3.5 (Weighted DSM = 가중 NCSN)**  
고정된 $\sigma$ 스케일에서 DSM loss에 $\sigma^2$를 곱하면:
$$\sigma^2 \mathbb{E}[\|s_\theta(\tilde x) - (-\epsilon/\sigma)\|^2] = \mathbb{E}[\|\sigma s_\theta(\tilde x) + \epsilon\|^2]$$

이는 $s_\theta$와 노이즈의 크기를 모두 스케일링하므로 고차원에서의 기울기 크기를 정규화한다.

증명: 양변을 전개하면
$$\sigma^2\mathbb{E}[\|s - (-\epsilon/\sigma)\|^2] = \sigma^2\mathbb{E}[\|s\|^2 + \epsilon^T s / \sigma + \epsilon^T \epsilon / \sigma^2]$$
$$= \mathbb{E}[\|\sigma s\|^2 + 2\sigma s \cdot \epsilon + \|\epsilon\|^2]$$
$$= \mathbb{E}[\|\sigma s + \epsilon\|^2]$$
$\square$

**정리 3.6 (Annealed Langevin의 수렴)**  
큰 σ부터 작은 σ로 순차적으로 Langevin을 실행하면, 최종 샘플의 분포가 true $p$에 더 가까워진다 (정확한 수렴 보장은 아니지만 empirically 유효).

증명 스케치: 각 단계에서 Langevin의 mixing time이 개선되고, 누적된 기울기 오차가 작은 σ에서는 large basin of attraction 때문에 덜 민감하다.

## 💻 NumPy / PyTorch 구현 검증

### 실험 1: 조건부 Score Network 구현
```python
import torch
import torch.nn as nn
import numpy as np

class ConditionalScoreNet(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128):
        super().__init__()
        # σ를 embedding으로 변환
        self.sigma_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.net = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x, sigma):
        # sigma: shape (B,) or (B, 1)
        if sigma.dim() == 1:
            sigma = sigma.unsqueeze(1)
        
        sigma_emb = self.sigma_embedding(torch.log(sigma))  # log-embedding
        x_aug = torch.cat([x, sigma_emb], dim=1)
        return self.net(x_aug)

# NCSN 손실 함수
def ncsn_loss(model, x_clean, sigmas, lambda_fn=lambda s: s**2):
    """
    sigmas: list of noise levels
    lambda_fn: weighting function (default: sigma^2)
    """
    loss = 0.0
    for sigma in sigmas:
        epsilon = torch.randn_like(x_clean)
        x_noisy = x_clean + sigma * epsilon
        
        score_pred = model(x_noisy, torch.full((x_clean.shape[0],), sigma))
        score_true = -epsilon / sigma
        
        mse = torch.mean((score_pred - score_true)**2)
        loss += lambda_fn(sigma) * mse
    
    return loss / len(sigmas)

# 테스트
x_clean = torch.randn(64, 2)
model = ConditionalScoreNet(input_dim=2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

sigmas = [1.0, 0.5, 0.25, 0.1]  # 4개 스케일

for step in range(500):
    loss = ncsn_loss(model, x_clean, sigmas)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 100 == 0:
        print(f"Step {step}: Loss = {loss.item():.4f}")
```

### 실험 2: Annealed Langevin Sampling
```python
def annealed_langevin_sampling(model, sigmas, x_shape, steps_per_scale=100, eta=0.1):
    """
    Annealed Langevin 샘플링
    sigmas: list of noise levels (큰 것부터 작은 것 순서)
    """
    x = torch.randn(x_shape)
    
    for sigma in sigmas:
        print(f"Sampling at σ = {sigma:.4f}")
        sigma_tensor = torch.full((x.shape[0],), sigma)
        
        for step in range(steps_per_scale):
            score = model(x, sigma_tensor)
            noise = torch.randn_like(x)
            x = x + (eta * score) + np.sqrt(2 * eta) * noise
    
    return x

# 샘플링 실행
sigmas_schedule = [1.0, 0.5, 0.25, 0.1, 0.05]
samples = annealed_langevin_sampling(model, sigmas_schedule, (16, 2))
print(f"Generated samples shape: {samples.shape}")
```

### 실험 3: σ별 성능 비교
```python
import matplotlib.pyplot as plt

# 다양한 σ 스케일에서 score 추정 정확도 비교
def evaluate_score_accuracy(model, x_clean, sigmas):
    """각 σ에서 score 추정 오차 계산"""
    results = {}
    
    for sigma in sigmas:
        epsilon = torch.randn_like(x_clean)
        x_noisy = x_clean + sigma * epsilon
        
        score_pred = model(x_noisy, torch.full((x_clean.shape[0],), sigma))
        score_true = -epsilon / sigma
        
        mse = torch.mean((score_pred - score_true)**2).item()
        results[f'σ={sigma:.2f}'] = mse
    
    return results

sigmas = [0.05, 0.1, 0.25, 0.5, 1.0]
errors = evaluate_score_accuracy(model, x_clean, sigmas)

for sigma_str, error in errors.items():
    print(f"{sigma_str}: MSE = {error:.4f}")

# 시각화
plt.figure(figsize=(10, 6))
sigmas_vals = [float(k.split('=')[1]) for k in errors.keys()]
mses = list(errors.values())
plt.semilogx(sigmas_vals, mses, 'o-', linewidth=2, markersize=8)
plt.xlabel('Noise Level σ')
plt.ylabel('Score Estimation MSE')
plt.title('NCSN: Score Accuracy vs Noise Level')
plt.grid(True, alpha=0.3)
plt.show()
```

## 🔗 실전 활용

**고해상도 이미지 생성**
- NCSN은 32×32, 64×64 이미지 생성에서 FID 개선 달성
- 큰 σ (coarse)에서 레이아웃/mode, 작은 σ (fine)에서 디테일 학습

**Inpainting & Image Restoration**
- 일부 픽셀을 고정하고 나머지를 annealed Langevin으로 샘플링
- 각 σ에서 조건부 score로 복원

**Score 기반 음성 합성**
- NCSN-inspired 방법으로 spectral data의 score 학습

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| 다양한 σ가 충분히 촘촘함 | 너무 많으면 훈련 시간 증가 |
| 각 σ에서 DSM이 정확함 | 저확률 영역에서도 여전히 오차 있음 |
| Annealing이 경로 의존성 해결 | 모드 전환은 여전히 어려울 수 있음 |
| σ scheduling이 고정적 | 적응적 scheduling이 더 나을 수 있음 |

## 📌 핵심 정리

**NCSN은 여러 노이즈 스케일 σ₁ > σ₂ > ... > σₗ을 사용하여 조건부 score s_θ(x, σ)를 학습한다. Annealed Langevin sampling은 큰 σ부터 작은 σ로 순차적으로 이동하여 고차원 모드 믹싱 문제를 완화한다. 이는 저확률 영역의 score 추정 부정확성을 효과적으로 해결하는 핵심 기법이다.**

## 🤔 생각해볼 문제

### 문제 1
왜 가중치 $\lambda_i = \sigma_i^2$로 설정하는가? 다른 가중치는 어떻게 영향을 미치는가?

<details>
<summary>해설</summary>

$\sigma_i^2$의 가중치는 고차원에서 기울기 크기를 정규화한다. 정리 3.5에서 보았듯이, 이는 노이즈 항과 score 항의 스케일을 균형있게 유지한다. 더 큰 가중치는 큰 σ를 강조 (global structure), 더 작은 가중치는 작은 σ를 강조 (fine details).
</details>

### 문제 2
Annealed Langevin에서 각 σ 단계에서 몇 개의 스텝을 해야 하는가? 적다면? 많다면?

<details>
<summary>해설</summary>

너무 적으면: 수렴하기 전에 다음 σ로 넘어가 정상 분포를 놓친다.  
너무 많으면: 계산 비용 증가, 노이즈 오차 누적. 보통 각 σ마다 100~500 스텝이 경험적으로 좋다.
</details>

### 문제 3
NCSN에서 σ 스케줄을 균등하게 배치하는 것(arithmetic)과 기하학적으로 배치하는 것(geometric) 중 어느 것이 더 나은가?

<details>
<summary>해설</summary>

기하학적 배치가 일반적으로 더 좋다. 이는 로그-정규 분포 가정과 일치하며, low-frequency (큰 σ)와 high-frequency (작은 σ) 정보의 균형을 유지한다. Song et al. (2021)에서도 geometric schedule을 사용한다.
</details>

---

<div align="center">

[◀ 이전](./02-denoising-score-matching.md) | [📚 README](../README.md) | [다음 ▶](./04-score-sde.md)

</div>
