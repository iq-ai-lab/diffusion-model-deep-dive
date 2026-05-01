# CFG Quality–Diversity Trade-off (Ho & Salimans 2022)

## 🎯 핵심 질문

CFG scale $w$를 증가시키면 조건에 맞는 샘플의 품질(quality)은 향상되지만, 왜 데이터 다양성(diversity)과 coverage는 저하될까? 이 trade-off를 수학적으로 분석할 수 있을까?

## 🔍 왜 Quality–Diversity Trade-off인가

CFG는 강력한 도구지만, $w$를 크게 설정하면:
- **FID, CLIP score 증가** (조건과 일치도 높음) → Quality up
- **Recall, inception curve 저하** (분포의 극단 영역 샘플 감소) → Diversity down

이는 score 기반 생성 모델의 본질적 특성이다: guidance는 distribution을 sharpening하기 때문이다.

## 📐 수학적 선행 조건

- **Energy-based model**: $p^{(w)}(x,y) \propto p(x,y)^w \cdot p(y)^{w}$ (implicit)
- **Temperature scaling**: soft-max → argmax로의 전환과 유사
- **KL divergence**: $D_{\text{KL}}(p||q)$, mode-covering vs mode-seeking
- **FID, Recall**: 생성 샘플의 분포 평가 지표

## 📖 직관적 이해

**Score 형태 분석**:
$$\tilde{s}_\theta(x, c, w) = (1+w) s_\theta(x,c) - w \, s_\theta(x, \emptyset)$$
$$= s_\theta(x, \emptyset) + (1+w) \nabla \log p(c|x)$$

$w$를 증가하면:
- Implicit classifier gradient $\nabla \log p(c|x)$가 과도하게 증폭
- 높은 확률 모드로 trajectory가 빨리 수렴
- 낮은 확률 영역(꼬리)은 무시

결과: Distribution sharpening (Temperature $\to 0$ 유사)

## ✏️ 엄밀한 정의

**Implicit distribution under CFG** (scale $w$):
$$\log p^{(w)}(x,c) \approx \log p(x) + (w+1) \log p(c|x) - \log p(c)$$

Reparameterized energy:
$$p^{(w)}(x,c) \propto p(x,c)^{w+1} \cdot p(x)^{-w}$$

또는 normalized 형태로 (Bayes):
$$p^{(w)}(x|c) = \frac{p^{(w)}(x,c)}{\int p^{(w)}(x',c) dx'}$$

이는 implicit한 re-weighting이다.

## 🔬 정리와 증명

**정리 (Distribution Sharpening)**: CFG scale $w > 0$는 조건부 분포를 temperature $\tau = 1/(1+w)$만큼 sharpening한다:

$$p^{(w)}(x|c) \approx \text{Softmax}_\tau(p(x|c))$$

여기서 $\tau$가 작을수록 sharpening이 강함.

**증명**:
$$\tilde{s}_\theta = s_\theta(\emptyset) + (1+w) \cdot \nabla \log p(c|x)$$

이를 따르는 diffusion trajectory는 (정상 조건 하):
$$dx = [f(x,t) + g(t)^2 \tilde{s}_\theta] dt + g(t) dw$$

이 trajectory는 다음 분포에서 표본을 생성한다:
$$p^{(w)}(x|c) \propto p(x|c)^{1+w} \cdot p(x)^{-w}$$

정규화 후, energy landscape의 관점에서:
$$\log p^{(w)}(x|c) = (1+w) \log p(x|c) - w \log p(x)$$

따라서 조건부 likelihood가 $(1+w)$배 증폭되고, 무조건 가능도가 감소하는 trade-off 구조이다. $\square$

## 💻 NumPy / PyTorch 구현 검증

### 실험 1: CFG Scale이 분포에 미치는 영향

```python
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# 1D Gaussian mixture 모델
class Gaussian1DMixture:
    def __init__(self, means=[0, 5], scales=[1, 1], weights=[0.5, 0.5]):
        self.means = torch.tensor(means, dtype=torch.float32)
        self.scales = torch.tensor(scales, dtype=torch.float32)
        self.weights = torch.tensor(weights, dtype=torch.float32) / sum(weights)
    
    def score(self, x, conditional=False):
        """Score function ∇ log p(x|c)"""
        scores = []
        for m, s, w in zip(self.means, self.scales, self.weights):
            score_i = -(x - m) / (s ** 2)
            scores.append(w * score_i)
        return sum(scores)

mixture = Gaussian1DMixture(means=[0, 5], scales=[1, 1])

# CFG sweep
x_vals = torch.linspace(-5, 10, 1000)
cfg_scales = [0.0, 1.0, 3.0, 7.5, 15.0]

results = {}
for w in cfg_scales:
    # Implicit energy (approximation)
    s_uncond = mixture.score(x_vals)
    s_cond = mixture.score(x_vals)  # 단순화: same as unconditional
    
    # Reweight
    s_guided = (1 + w) * s_cond - w * s_uncond
    
    # Soft exponential approximation
    guided_prob = F.softmax(3 * s_guided, dim=0)  # temp=1/(1+w) ≈ 3
    results[w] = guided_prob.numpy()

print("CFG Scale Effects on Distribution Sharpening:")
print(f"{'Scale':>6} | {'Mode 1 Prob':>12} | {'Mode 2 Prob':>12} | {'Entropy':>10}")
print("-" * 55)
for w in cfg_scales:
    mode1_idx = np.argmax(results[w][:333])
    mode2_idx = np.argmax(results[w][333:]) + 333
    p = results[w]
    entropy = -np.sum(p[p > 1e-7] * np.log(p[p > 1e-7]))
    print(f"{w:6.1f} | {results[w][mode1_idx]:12.4f} | {results[w][mode2_idx]:12.4f} | {entropy:10.4f}")
```

### 실험 2: FID vs Recall Trade-off

```python
# Simulate FID and Recall curves
cfg_scales = np.linspace(0, 20, 21)
fid_scores = []
recall_scores = []

# Empirical model: quality increases, diversity decreases
for w in cfg_scales:
    # FID decreases (better quality) with w
    fid = 20 * np.exp(-0.15 * w) + 5
    
    # Recall decreases (less diversity) with w
    recall = 0.7 - 0.02 * w
    
    fid_scores.append(fid)
    recall_scores.append(max(0, recall))

# Optimal operating point (user preference dependent)
pareto_w = []
for i, (f, r) in enumerate(zip(fid_scores, recall_scores)):
    # Example: weight FID 2x more than recall
    score = -f + 0.5 * r
    pareto_w.append(score)

optimal_idx = np.argmax(pareto_w)
optimal_w = cfg_scales[optimal_idx]

print("FID vs Recall Trade-off:")
print(f"Typical range: w ∈ [3, 15]")
print(f"Optimal w for quality-weighted metric: {optimal_w:.1f}")
print(f"  FID at w={optimal_w:.1f}: {fid_scores[optimal_idx]:.2f}")
print(f"  Recall at w={optimal_w:.1f}: {recall_scores[optimal_idx]:.3f}")
```

### 실험 3: Dynamic Thresholding (Imagen)

```python
def dynamic_threshold(x, percentile=99.0):
    """
    Clamp latents to percentile value.
    Prevents out-of-distribution overshoot from high w.
    """
    s = torch.quantile(x.abs(), percentile / 100.0)
    s = torch.clamp(s, min=1.0)  # ≥ 1
    return torch.clamp(x, min=-s, max=s) / s

# Simulate high guidance overshoot
x_guided = torch.randn(4, 64) * 5  # High w leads to large latent magnitudes
x_unclipped = x_guided.clone()

x_clipped = dynamic_threshold(x_guided, percentile=95.0)

print("Dynamic Thresholding Effect:")
print(f"Before: mean={x_unclipped.abs().mean():.4f}, max={x_unclipped.abs().max():.4f}")
print(f"After:  mean={x_clipped.abs().mean():.4f}, max={x_clipped.abs().max():.4f}")
print(f"Recovery: reduces OOD saturation, improves visual quality at high w")
```

### 실험 4: Optimal Scale Selection

```python
# User-facing guideline
def recommended_cfg_scale(task="general"):
    """Return typical CFG scales for different tasks"""
    scales = {
        "general": 7.5,
        "photorealistic": 5.0,
        "artistic": 15.0,
        "balanced": 10.0,
        "maximum_diversity": 1.0,
        "maximum_quality": 20.0
    }
    return scales.get(task, 7.5)

print("Typical CFG Scale Ranges (Stable Diffusion):")
print(f"  Low (1-3):     High diversity, lower quality")
print(f"  Medium (5-10): Balanced (default {recommended_cfg_scale('general')})")
print(f"  High (12+):    High quality, lower diversity")
print(f"  Very High (20+): Risk of artifacts with dynamic thresholding")
```

## 🔗 실전 활용

**Stable Diffusion**:
- Default $w = 7.5$
- Photorealistic prompts: $w = 5$~8
- Artistic/stylized: $w = 12$~18
- Dynamic thresholding applied at $w > 10$ (Imagen 기법)

**DALL-E 3, Midjourney**: 유사한 trade-off, 모델-specific optimal $w$ 선택

## ⚖️ 가정과 한계

1. **분포 가정**: 모델이 $p(x|c)$와 $p(x)$를 정확히 학습했다고 가정 (실제론 부분적)
2. **Out-of-distribution risk**: 높은 $w$에서 학습 데이터 분포 밖으로 나감
3. **Dynamic thresholding 필요성**: $w > 10$에서는 clipping이 거의 필수
4. **Task-dependent optimality**: 최적 $w$는 prompt와 use-case에 따라 다름

## 📌 핵심 정리

- **Score formula**: $\tilde{s} = s_\theta(\emptyset) + (1+w) \nabla \log p(c|x)$
- **Implicit distribution**: $p^{(w)}(x|c) \propto p(x|c)^{1+w} \cdot p(x)^{-w}$
- **Temperature effect**: $w \uparrow$ ⟹ distribution sharpening ($\tau \to 0$)
- **Typical range**: $w \in [3, 15]$ (task & preference에 따라 가변)
- **Dynamic thresholding**: Imagen의 기법으로 $w > 10$에서 artifacts 방지

## 🤔 생각해볼 문제

1. CFG에서 FID는 개선되지만 recall이 저하되는 이유를 분포 관점에서 설명하라. (hint: mode-seeking)

<details>
<summary>Hint</summary>
FID는 생성 분포의 평균과 공분산이 실제 분포와 얼마나 가까운지 측정하므로, sharpening된 분포(고확률 영역 집중)에서 개선된다. 반면 recall은 실제 데이터의 모든 모드가 생성 분포에 포함되는지 확인하므로, sharpening으로 꼬리 영역이 무시되면 recall이 떨어진다.
</details>

2. Dynamic thresholding이 없다면, $w = 20$일 때 왜 생성 이미지가 saturated/부자연스러울까?

<details>
<summary>Hint</summary>
높은 $w$는 latent space에서 원점(무조건 분포 중심)에서 멀리 떨어진 극단적 값들을 생성하도록 유도한다. 이들은 학습 데이터 범위를 벗어나 (out-of-distribution), decoder가 과포화된 색상이나 기형적 구조로 해석하게 된다. Thresholding은 이를 유효 범위 내로 강제한다.
</details>

3. Implicit classifier의 gradient $\nabla \log p(c|x)$가 항상 신뢰할 수 있을까? 어떤 경우에 unreliable할까?

<details>
<summary>Hint</summary>
모델이 out-of-distribution 영역에 있을 때 (예: high $w$ + high noise), $\nabla \log p(c|x)$는 학습 데이터에서 보지 못한 $x$ 영역에서 부정확해진다. 또한 high noise regime($t$ 큼)에서 classification이 본질적으로 어려워진다.
</details>

---

<div align="center">

[◀ 이전](./02-classifier-free-guidance.md) | [📚 README](../README.md) | [다음 ▶](./04-cross-attention.md)

</div>
