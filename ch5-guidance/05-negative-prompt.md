# Negative Prompt & Compositional Guidance

## 🎯 핵심 질문

생성 과정에서 "원하지 않는 것"을 명시적으로 지정할 수 있을까? Guidance를 확장하여 positive prompt와 negative prompt를 동시에 활용하면, 더 정교한 제어가 가능할까?

## 🔍 왜 Negative Prompt인가

사용자 입장에서:
- "a dog" → 다양한 개들, 원치 않은 특성 많음
- "a dog, NOT blurry, NOT cartoon" → 더 정교한 제어

기술적으로:
- CFG는 positive guidance만 지원 (무조건 score에서 조건부로)
- 하지만 negative condition도 동일한 score framework에서 표현 가능

**Liu et al. (2022)** (Composable Diffusion): 여러 조건(positive/negative)을 weighted combination으로 통합하는 방법 제안. **Bansal et al. (2023)** (Universal Guidance): forward pass 하나로 모든 조건을 처리하는 범용 프레임워크.

## 📐 수학적 선행 조건

- **Classifier-free guidance**: $\tilde{\epsilon} = (1+w)\epsilon(x,c) - w\epsilon(x,\emptyset)$
- **Negative guidance**: $w_- > 0$ scale for unwanted condition
- **Weighted score combination**: linear combination of score functions
- **Compositionality**: independence assumption (또는 weighted mixture)

## 📖 직관적 이해

**Positive guidance** ("a red dog"):
$$\tilde{\epsilon}_+ = \epsilon(\emptyset) + w_+ \cdot [\epsilon(x, c_+) - \epsilon(x, \emptyset)]$$

**Negative guidance** ("NOT blurry"):
$$\tilde{\epsilon}_- = \epsilon(\emptyset) - w_- \cdot [\epsilon(x, c_-) - \epsilon(x, \emptyset)]$$

**Combined**:
$$\tilde{\epsilon}_{\text{combined}} = \epsilon(\emptyset) + w_+ \cdot [\epsilon(x, c_+) - \epsilon(x, \emptyset)] - w_- \cdot [\epsilon(x, c_-) - \epsilon(x, \emptyset)]$$

즉, positive와 negative guidance의 gradient를 반대 부호로 적용.

직관: 모델이 negative condition을 "피하도록" 유도. 예를 들어 "NOT blurry"는 sharpness 방향으로 scores를 조정.

## ✏️ 엄밀한 정의

**Negative Prompt Formula**:
$$\tilde{\epsilon}_\theta = \epsilon_\theta(x, \emptyset) + w_+ [\epsilon_\theta(x, y_+) - \epsilon_\theta(x, \emptyset)] - w_- [\epsilon_\theta(x, y_-) - \epsilon_\theta(x, \emptyset)]$$

재정렬하면:
$$\tilde{\epsilon}_\theta = (1-w_-) \epsilon_\theta(x, \emptyset) + w_+ \epsilon_\theta(x, y_+) - w_- \epsilon_\theta(x, y_-)$$

또는 normalized 형태:
$$\tilde{\epsilon}_\theta = \frac{w_+ \epsilon_\theta(x, y_+) + (1-w_+ - w_-) \epsilon_\theta(x, \emptyset) - w_- \epsilon_\theta(x, y_-)}{w_+ + (1-w_+ - w_-) + w_-}$$

(분모는 가중치 합 = 1)

**Compositional Guidance** (Liu et al. 2022):
여러 조건 $c_1, c_2, \ldots, c_K$ (일부는 negative)에 대해:
$$\tilde{\epsilon} = \sum_{k=1}^{K} w_k \epsilon(x, c_k) + \left(1 - \sum w_k\right) \epsilon(x, \emptyset)$$

여기서 $w_k > 0$ (positive) 또는 $w_k < 0$ (negative).

## 🔬 정리와 증명

**정리 (Negative Guidance의 Score-based 해석)**: Negative prompt는 implicit하게 unwanted condition에서의 확률을 감소시킨다.

**증명**:
$$\tilde{s}_\theta = s_\theta(\emptyset) + w_+ \nabla \log p(y_+ | x) - w_- \nabla \log p(y_- | x)$$

이는 다음 분포에서의 샘플링과 동치:
$$p^{(w_+, w_-)}(x | y_+, \neg y_-) \propto p(x | y_+)^{w_+} \cdot p(x | y_-)^{-w_-} \cdot p(x)^{1-w_+-w_-}$$

따라서:
$$\log p^{(w_+, w_-)} = w_+ \log p(y_+ | x) - w_- \log p(y_- | x) + \text{const}$$

$w_- > 0$이면, $\log p(y_- | x)$가 감소하는 방향으로 학습된다. $\square$

## 💻 NumPy / PyTorch 구현 검증

### 실험 1: Negative Prompt 기본 구현

```python
import torch
import torch.nn as nn

class GuidedDiffusionModel(nn.Module):
    """Supports positive and negative prompts"""
    def __init__(self, latent_dim=128, cond_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
    
    def forward(self, x_t, cond):
        x_cond = torch.cat([x_t, cond], dim=1)
        return self.net(x_cond)

model = GuidedDiffusionModel()

# Inference with positive and negative prompts
x_t = torch.randn(1, 128)

# Embeddings
c_positive = torch.randn(1, 64)
c_negative = torch.randn(1, 64)
c_null = torch.zeros(1, 64)

# Predictions
eps_positive = model(x_t, c_positive)
eps_negative = model(x_t, c_negative)
eps_null = model(x_t, c_null)

# Negative prompt guidance
w_positive = 7.5
w_negative = 5.0

eps_guided = (eps_null 
              + w_positive * (eps_positive - eps_null) 
              - w_negative * (eps_negative - eps_null))

print("Negative Prompt Guidance:")
print(f"  ||eps_positive||: {eps_positive.norm():.4f}")
print(f"  ||eps_negative||: {eps_negative.norm():.4f}")
print(f"  ||eps_guided||:   {eps_guided.norm():.4f}")
print(f"  Guidance direction: "
      f"pos={((eps_positive - eps_null) / (eps_positive - eps_null).norm()).mean():.3f}, "
      f"neg={((eps_negative - eps_null) / (eps_negative - eps_null).norm()).mean():.3f}")
```

### 실험 2: Compositional Guidance (Multiple Conditions)

```python
def compositional_guidance(model, x_t, conditions, weights, c_null):
    """
    conditions: dict {'c1': emb1, 'c2': emb2, ...}
    weights: dict {'c1': 7.5, 'c2': -3.0, ...}  (negative = avoid)
    """
    eps_result = model(x_t, c_null)
    
    for cond_name, cond_emb in conditions.items():
        if cond_name not in weights:
            continue
        
        w = weights[cond_name]
        eps_cond = model(x_t, cond_emb)
        
        if w > 0:
            # Positive guidance: move towards this condition
            eps_result = eps_result + w * (eps_cond - model(x_t, c_null))
        else:
            # Negative guidance: move away from this condition
            eps_result = eps_result - abs(w) * (eps_cond - model(x_t, c_null))
    
    return eps_result

# Example: Generate "a dog" but "NOT blurry, NOT cartoon"
conditions = {
    'positive': torch.randn(1, 64),   # "a dog"
    'avoid_blur': torch.randn(1, 64),  # "blurry"
    'avoid_cartoon': torch.randn(1, 64) # "cartoon"
}
weights = {
    'positive': 7.5,
    'avoid_blur': -5.0,
    'avoid_cartoon': -3.0
}

x_t = torch.randn(1, 128)
c_null = torch.zeros(1, 64)
eps_compositional = compositional_guidance(model, x_t, conditions, weights, c_null)

print("\nCompositional Guidance (multiple conditions):")
for name, w in weights.items():
    sign = "+" if w > 0 else "-"
    print(f"  {name:15s}: {sign}{abs(w):.1f}")
print(f"  ||eps_compositional||: {eps_compositional.norm():.4f}")
```

### 실험 3: Guidance Scale Sweep (Positive + Negative)

```python
# Parameter sweep
w_pos_range = [1.0, 5.0, 10.0, 15.0]
w_neg_range = [0.0, 2.0, 5.0, 10.0]

results = {}

for w_pos in w_pos_range:
    for w_neg in w_neg_range:
        eps = (eps_null 
               + w_pos * (eps_positive - eps_null) 
               - w_neg * (eps_negative - eps_null))
        
        key = (w_pos, w_neg)
        results[key] = {
            'norm': eps.norm().item(),
            'ratio': (eps / (eps_null + 1e-8)).abs().mean().item()
        }

print("\nGuidance Scale Sweep (w_pos vs w_neg):")
print(f"{'w_pos':>6} | {'w_neg':>6} | {'||eps||':>8} | {'Influence':>10}")
print("-" * 40)
for (w_pos, w_neg) in sorted(results.keys()):
    res = results[(w_pos, w_neg)]
    print(f"{w_pos:6.1f} | {w_neg:6.1f} | {res['norm']:8.4f} | {res['ratio']:10.4f}")
```

### 실험 4: Practical Negative Prompt Examples

```python
import json

# Common negative prompts in practice
negative_prompt_patterns = {
    "quality": {
        "positive": "high quality, detailed, sharp",
        "negative": "blurry, low quality, distorted"
    },
    "style": {
        "positive": "photorealistic, natural lighting",
        "negative": "cartoon, anime, abstract"
    },
    "artifacts": {
        "positive": "anatomically correct, realistic proportions",
        "negative": "deformed, extra limbs, distorted face"
    },
    "misc": {
        "positive": "",
        "negative": "watermark, text, signature"
    }
}

print("Common Negative Prompt Patterns:\n")
for category, prompts in negative_prompt_patterns.items():
    print(f"{category.upper()}:")
    print(f"  Positive: {prompts['positive']}")
    print(f"  Negative: {prompts['negative']}\n")

# Recommended w_positive and w_neg based on category
w_recommendations = {
    "quality": {"w_pos": 7.5, "w_neg": 5.0},
    "style": {"w_pos": 10.0, "w_neg": 7.5},
    "artifacts": {"w_pos": 15.0, "w_neg": 10.0}
}

print("Recommended Guidance Scales:")
for category, scales in w_recommendations.items():
    print(f"  {category:15s}: w_pos={scales['w_pos']:.1f}, w_neg={scales['w_neg']:.1f}")
```

## 🔗 실전 활용

**Stable Diffusion** (negative prompt):
```
Prompt:  "a red dog, high quality, detailed"
Negative: "blurry, cartoon, distorted face, extra limbs"
w_positive: 7.5
w_negative: 5.0
```

**Imagen, DALL-E 3**:
- Negative prompt 공식 지원
- Multi-condition guidance (색상, 스타일, 카테고리 동시 제어)

**Universal Guidance** (Bansal et al. 2023):
- Forward 한 번에 모든 조건 처리
- 임의의 조건(NLI, semantic similarity 등) 가능

## ⚖️ 가정과 한계

1. **독립성 가정**: 여러 조건을 weighted sum으로 처리하면, 상호작용(interaction) 무시
2. **Weight 선택**: $w_+$, $w_-$의 최적값은 task-dependent. 휴리스틱에 의존
3. **Semantic 품질**: Negative condition이 "무엇을 피할 것인가"를 명확히 표현해야 함
4. **계산 복잡도**: 조건 K개 → forward K+1번 (null 포함)

## 📌 핵심 정리

- **Negative prompt formula**: $\tilde{\epsilon} = \epsilon(\emptyset) + w_+ [\epsilon(x,y_+) - \epsilon(x,\emptyset)] - w_- [\epsilon(x,y_-) - \epsilon(x,\emptyset)]$
- **Compositional form**: weighted sum of epsilon predictions with opposite signs for negative
- **Implicit effect**: negative condition의 확률을 감소시키는 implicit classifier gradient
- **Practical ranges**: $w_+ \in [5, 15]$, $w_- \in [2, 10]$ (사용자 선호)
- **Universal guidance**: 단일 forward로 모든 조건 처리 (미래 방향)

## 🤔 생각해볼 문제

1. Negative prompt를 너무 강하게 설정하면 (예: $w_- = 20$), 왜 생성 이미지가 "피하는 것"이 이상하게 표현될까?

<details>
<summary>Hint</summary>
강한 negative guidance는 implicit classifier score를 음의 방향으로 크게 증폭하여, model을 out-of-distribution 영역으로 몬다. 결과: 피하려던 특성이 과장되거나 왜곡되어 나타난다. (예: "NOT cartoon"을 매우 강하게 하면, 오히려 초현실적이고 불자연스러운 스타일 출현)
</details>

2. Positive와 negative를 동시에 사용할 때, 다음 두 설정의 차이는?
   - 설정 A: $w_+ = 7.5$, $w_- = 0$
   - 설정 B: $w_+ = 7.5$, $w_- = 5.0$ (negative: "cartoon")

<details>
<summary>Hint</summary>
설정 A는 "a dog"에만 집중한다. 설정 B는 "a dog" 방향으로 가면서 동시에 "cartoon" 방향에서 멀어진다. 결과: 더 photorealistic하고 자연스러운 개의 모습. 하지만 조건부 임베딩이 부정확하면 B가 뜻하지 않은 스타일 변화를 초래할 수 있다.
</details>

3. Liu et al. (2022)의 Compositional Diffusion에서, 여러 positive 조건들이 상충(conflict)할 경우 어떻게 해결할까?

<details>
<summary>Hint</summary>
예: "a red dog" + "a blue dog" (상충). Weighted sum 방식에서는 두 조건의 gradient를 더하므로, 결과적으로 보라색(빨강+파랑 혼합) 개가 생길 가능성 높음. 또는 alternating guidance, attention mechanism 등으로 조건 간 우선순위 정의. 또는 사용자가 명시적으로 하나를 선택.
</details>

---

<div align="center">

[◀ 이전](./04-cross-attention.md) | [📚 README](../README.md) | [다음 ▶](../ch6-latent-arch/01-latent-diffusion.md)

</div>
