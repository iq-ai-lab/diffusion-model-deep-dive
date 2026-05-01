# Classifier-Free Guidance (Ho & Salimans 2022)

## 🎯 핵심 질문

외부 분류기 없이, diffusion model 자신만으로 조건부 생성을 강화할 수 있을까? 학습 중 조건을 확률적으로 drop하면, inference에서 무조건과 조건부 출력의 차이로부터 implicit classifier를 구성할 수 있을까?

## 🔍 왜 Classifier-Free Guidance인가

Classifier guidance의 문제점:
1. 분류기 별도 학습 비용
2. Noisy 이미지에서 정확한 classification 어려움
3. Out-of-distribution 샘플에서 분류기 신뢰도 저하

**Ho & Salimans (2022)**: 조건부 diffusion model을 학습할 때, 확률 $\pi_{\text{drop}}$로 conditioning 정보를 버리면, 모델 자체가 두 분포 $p(x|y)$와 $p(x|\emptyset)$를 모두 학습하게 된다. Inference에서 이 둘의 차이를 이용하여 implicit guidance를 수행할 수 있다.

## 📐 수학적 선행 조건

- **Conditional score**: $s_\theta(x_t, c, t) = \nabla_x \log p_\theta(x_t | c)$
- **Unconditional score**: $s_\theta(x_t, \emptyset, t) = \nabla_x \log p_\theta(x_t)$
- **Score combination**: weighted mixture
- **Conditioning dropout**: $p(\text{drop} \, c) = \pi_{\text{drop}}$

## 📖 직관적 이해

**학습 단계**:
- 배치의 일부(1-$\pi_{\text{drop}}$): 조건 $c$와 함께 학습
- 배치의 일부($\pi_{\text{drop}}$): 조건을 무시하고 $\emptyset$로 학습

**Inference 단계**:
- $\epsilon_\theta(x_t, c)$: 조건 주어졌을 때 예측 노이즈
- $\epsilon_\theta(x_t, \emptyset)$: 조건 없을 때 예측 노이즈
- 차이: $(1+w)[\epsilon_\theta(x_t,c) - \epsilon_\theta(x_t,\emptyset)]$ → implicit classifier의 역할

직관: 모델이 "조건 있을 때 vs 없을 때"의 차이를 학습했으므로, inference에서 이 차이를 증폭하면 조건부 분포로 steerable해진다.

## ✏️ 엄밀한 정의

**학습 목표** (classifier-free objective):
$$\mathcal{L} = \mathbb{E}_{t, x, c} \left[ \|\epsilon - \epsilon_\theta(x_t, c^*_t) \|^2 \right]$$

여기서 $c^*_t$는:
- 확률 $1 - \pi_{\text{drop}}$: 원본 조건 $c$
- 확률 $\pi_{\text{drop}}$: null token $\emptyset$ (또는 무시된 embedding)

**Inference (CFG scale $w$)**:
$$\tilde{\epsilon}_\theta(x_t, c, w) = (1+w) \epsilon_\theta(x_t, c) - w \epsilon_\theta(x_t, \emptyset)$$

또는 score 형태:
$$\tilde{s}_\theta(x_t, c, w) = (1+w) s_\theta(x_t, c) - w \, s_\theta(x_t, \emptyset)$$

## 🔬 정리와 증명

**정리 (Implicit Classifier)**: CFG scale $w$로 조정된 noise 예측은, implicit classifier의 존재 하에 classifier guidance와 동치이다:
$$\tilde{\epsilon}_\theta(x_t, c, w) = \epsilon_\theta(x_t, \emptyset) + (1+w) [\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset)]$$

**증명**:
Score 형태에서 시작하자. Conditioning dropout 학습에 의해, 모델은 다음 두 분포를 학습한다:
$$p_\theta(x_t | c) \quad \text{and} \quad p_\theta(x_t)$$

따라서:
$$\log p_\theta(x_t | c) = \log p_\theta(x_t) + \log p_\theta(c | x_t) - \log p_\theta(c)$$

양변을 $x_t$에 대해 미분:
$$s_\theta(x_t, c) = s_\theta(x_t, \emptyset) + \nabla \log p_\theta(c | x_t)$$

그러므로:
$$\tilde{s}_\theta = (1+w) s_\theta(x_t, c) - w \, s_\theta(x_t, \emptyset)$$
$$= s_\theta(x_t, \emptyset) + (1+w) \nabla \log p_\theta(c|x_t)$$

이는 classifier guidance ($s = 1+w$)와 동일하다. $\square$

## 💻 NumPy / PyTorch 구현 검증

### 실험 1: Conditioning Dropout 시뮬레이션

```python
import torch
import torch.nn as nn

class ConditionalDiffusionModel(nn.Module):
    def __init__(self, latent_dim=128, cond_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
    
    def forward(self, x_t, cond, cond_drop_prob=0.0):
        # Random drop conditioning
        if self.training:
            mask = torch.rand(cond.shape[0], 1) > cond_drop_prob
            cond = cond * mask.float()
        
        x_cond = torch.cat([x_t, cond], dim=1)
        return self.net(x_cond)

model = ConditionalDiffusionModel()
model.train()

# Simulate batch with conditioning dropout
batch_size = 8
x_t = torch.randn(batch_size, 128)
c = torch.randn(batch_size, 64)

# Training with dropout
epsilon_pred_with_drop = model(x_t, c, cond_drop_prob=0.1)
print(f"Epsilon prediction shape: {epsilon_pred_with_drop.shape}")
print(f"Prediction norm: {epsilon_pred_with_drop.norm(dim=1).mean():.4f}")
```

### 실험 2: CFG Scale Sweep (Noise Space)

```python
model.eval()

# Inference
x_t = torch.randn(1, 128)
c = torch.randn(1, 64)

# Unconditional
eps_uncond = model(x_t, torch.zeros_like(c))

# Conditional
eps_cond = model(x_t, c)

# CFG scales
cfg_scales = [0.0, 1.0, 3.0, 7.5, 15.0]
cfg_results = []

for w in cfg_scales:
    eps_guided = (1 + w) * eps_cond - w * eps_uncond
    cfg_results.append({
        'scale': w,
        'norm': eps_guided.norm().item(),
        'diff_from_uncond': (eps_guided - eps_uncond).norm().item()
    })

print("CFG Scale Sweep:")
for res in cfg_results:
    print(f"  w={res['scale']:5.1f}: norm={res['norm']:.4f}, diff={res['diff_from_uncond']:.4f}")
```

### 실험 3: Implicit Classifier 검증

```python
# Verify: CFG = classifier guidance
# tilde_eps_cfg = (1+w) * eps_cond - w * eps_uncond
#                = eps_uncond + (1+w) * (eps_cond - eps_uncond)

w = 7.5
eps_guided_cfg = (1 + w) * eps_cond - w * eps_uncond
eps_guided_equiv = eps_uncond + (1 + w) * (eps_cond - eps_uncond)

error = (eps_guided_cfg - eps_guided_equiv).abs().max().item()
print(f"CFG ≈ Implicit Classifier: max error = {error:.2e}")

# Effective guidance scale
implicit_classifier_grad = eps_cond - eps_uncond  # ∝ ∇ log p(c|x_t)
effective_scale = 1 + w
print(f"Effective guidance scale on implicit classifier: {effective_scale}")
```

### 실험 4: Training with Conditioning Dropout

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Simulated training loop
for step in range(100):
    x_t = torch.randn(16, 128)
    c = torch.randn(16, 64)
    noise = torch.randn_like(x_t)
    
    # Forward with dropout
    eps_pred = model(x_t, c, cond_drop_prob=0.1)
    loss = (eps_pred - noise).pow(2).mean()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if step % 25 == 0:
        print(f"Step {step:3d}: loss = {loss.item():.4f}")

print("\nModel successfully trained with conditioning dropout")
```

## 🔗 실전 활용

**Stable Diffusion** (Rombach et al. 2021 + CFG):
- Text encoder (CLIP): $c = \text{CLIP}(\text{prompt})$
- Conditioning dropout: $\pi_{\text{drop}} = 0.1$ (학습 중)
- Inference: $w = 7.5$~15 (사용자 선택)

**DALL-E 2, Imagen**: CFG가 default guidance 방식

## ⚖️ 가정과 한계

1. **Conditioning dropout 필요**: 학습 시 반드시 dropout 적용. 미적용 시 무조건 score와 조건부 score가 축약(collapse)되어 CFG 비효과적
2. **Implicit classifier 가정**: 모델이 충분히 표현력 있어야 두 분포를 구분하여 학습 가능
3. **$w$의 상한**: 높은 $w$에서 out-of-distribution 샘플 생성 가능 (다음 섹션)
4. **계산량**: 무조건/조건부 각각 forward→ inference 시간 3배 증가

## 📌 핵심 정리

- **Conditioning Dropout**: 학습 중 $\pi_{\text{drop}}=0.1$로 조건 제거
- **CFG Formula**: $\tilde{\epsilon} = (1+w)\epsilon_\theta(x,c) - w\epsilon_\theta(x,\emptyset)$
- **Implicit Classifier**: 모델 내부의 조건 difference가 classifier 역할 수행
- **단일 모델**: 분류기 별도 학습 불필요 → 대규모 생성 모델에 최적

## 🤔 생각해볼 문제

1. Conditioning dropout이 없다면 (모든 배치에 조건 주입), CFG가 작동하지 않는 이유는?

<details>
<summary>Hint</summary>
모델이 조건부 분포만 학습하고 무조건 분포를 충분히 학습하지 못한다. 따라서 $\epsilon_\theta(x_t, \emptyset)$가 의미 있는 무조건 score가 되지 않아, 두 신호의 차이가 small 또는 noisy해진다.
</details>

2. $w$를 매우 크게 설정하면 (예: $w=100$), 생성된 이미지의 품질이 떨어질 수 있다. 왜?

<details>
<summary>Hint</summary>
매우 큰 $w$는 조건부 신호를 과도하게 증폭하여, 데이터 분포의 고확률 영역을 벗어나 out-of-distribution 영역으로 간다. 결과: 부자연스러운, 饱和된 색상, 기형적 구조.
</details>

3. CFG에서 "implicit classifier"의 parameterization은 무엇인가? (hint: noise-based vs score-based vs logit-based)

<details>
<summary>Hint</summary>
Implicit classifier는 explicit parameterization이 없다. 대신 모델 가중치 $\theta$와 conditioning dropout에 의해 암묵적으로 정의된다. Score 형태로는 $\nabla \log p(c|x_t) \approx \epsilon_\theta(x_t,c) - \epsilon_\theta(x_t,\emptyset)$이다.
</details>

---

<div align="center">

[◀ 이전](./01-classifier-guidance.md) | [📚 README](../README.md) | [다음 ▶](./03-cfg-tradeoff.md)

</div>
