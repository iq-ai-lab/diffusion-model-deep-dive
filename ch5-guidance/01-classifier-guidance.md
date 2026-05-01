# Classifier Guidance (Dhariwal & Nichol 2021)

## 🎯 핵심 질문

주어진 클래스 조건 y를 만족하는 샘플을 생성할 때, 사전학습된 분류기(classifier)의 gradients를 활용하여 확산 과정을 가이드할 수 있을까? Bayes' rule을 통해 조건부 score를 명시적으로 부스트할 수 있을까?

## 🔍 왜 Classifier Guidance인가

기본 diffusion model은 $p(x)$ (무조건 생성)만 학습한다. 조건부 생성 $p(x|y)$를 원할 때, 두 가지 선택지가 있다:
- **Conditional training**: $y$를 입력으로 받도록 재학습 (비용 큼)
- **Guidance via classifier**: 학습된 분류기 $p(y|x)$로부터 gradient 활용

Dhariwal & Nichol (2021)은 후자를 제안했고, 이는 classifier-free guidance의 선행 연구다.

## 📐 수학적 선행 조건

- **Bayes' rule**: $\nabla_x \log p(x|y) = \nabla_x \log p(x) + \nabla_x \log p(y|x)$
- **Score function**: $s_\theta(x,t) = \nabla_x \log p_\theta(x_t)$
- **Classifier score**: $\nabla_{x_t} \log p_\phi(y|x_t)$ (noisy 입력에서 정의)
- **Noise parameter**: $\epsilon$-parameterization vs score-matching

## 📖 직관적 이해

생성 과정 중 각 스텝에서:
1. 무조건 score $s_\theta(x_t, t)$: 데이터 분포로 향함
2. 분류기 score $\nabla \log p_\phi(y|x_t)$: 클래스 y에 유리한 방향
3. 합산: 두 힘이 동시에 작용하여 $p(x|y)$에 가까워짐

이 직관은 gradient ascent (또는 score matching의 관점에서 stochastic gradient Langevin dynamics)이다.

## ✏️ 엄밀한 정의

**Score augmentation** (Bayes' rule 기반):
$$\tilde{s}_\theta(x_t, y, t) = s_\theta(x_t, t) + s \cdot \nabla_{x_t} \log p_\phi(y | x_t)$$

여기서 $s > 0$는 guidance scale이다.

**Reverse ODE/SDE에서 대체**:
$$dx = [f(x,t) + g(t)^2 \tilde{s}_\theta(x_t,y,t)] dt + g(t) dw$$

**$\epsilon$-parameterization**으로 변환:
$$\tilde{\epsilon}_\theta(x_t, y, t) = \epsilon_\theta(x_t, t) - \sqrt{1 - \bar{\alpha}_t} \cdot s \cdot \nabla_{x_t} \log p_\phi(y | x_t)$$

## 🔬 정리와 증명

**정리**: Bayes' rule 하에서 augmented score는 조건부 분포의 로그-스무딩(log-smoothing)을 따른다.

**증명**:
$$\log p(x_t | y) = \log p(x_t) + \log p(y | x_t) - \log p(y)$$

양변을 $x_t$에 대해 미분:
$$\nabla_{x_t} \log p(x_t | y) = \nabla_{x_t} \log p(x_t) + \nabla_{x_t} \log p(y | x_t)$$

따라서:
$$\tilde{s}_\theta(x_t, y, t) = s_\theta(x_t, t) + s \cdot \nabla_{x_t} \log p_\phi(y | x_t)$$

는 $p(x_t | y)$의 score를 (scale $s$만큼) 근사한다. $\square$

## 💻 NumPy / PyTorch 구현 검증

### 실험 1: 분류기 gradients 계산

```python
import torch
import torch.nn as nn

class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)  # 10 classes
        )
    
    def forward(self, x_t):
        return self.net(x_t)

classifier = SimpleClassifier()
x_t = torch.randn(4, 128, requires_grad=True)
y_target = 3

logits = classifier(x_t)
log_probs = torch.log_softmax(logits, dim=1)
loss = log_probs[:, y_target].mean()
loss.backward()

grad_log_p_y_xt = x_t.grad  # ∇_{x_t} log p(y|x_t)
print(f"Classifier gradient shape: {grad_log_p_y_xt.shape}")
print(f"Gradient norm: {grad_log_p_y_xt.norm(dim=1).mean():.4f}")
```

### 실험 2: Score augmentation

```python
# 무조건 score (mock)
s_theta = torch.randn_like(x_t)

# Guidance scale
guidance_scale = 7.5

# Augmented score
tilde_s = s_theta + guidance_scale * grad_log_p_y_xt

print(f"Original score norm: {s_theta.norm(dim=1).mean():.4f}")
print(f"Augmented score norm: {tilde_s.norm(dim=1).mean():.4f}")
print(f"Increase ratio: {(tilde_s.norm(dim=1) / (s_theta.norm(dim=1) + 1e-6)).mean():.2f}x")
```

### 실험 3: Noisy classifier의 필요성

```python
# t=0 (깨끗한 이미지)
x_clean = torch.randn(1, 128)
y_class = 5

# t=large (매우 노이즈)
t_values = torch.tensor([10, 100, 500, 999])
alpha_bar = torch.cos(torch.pi * t_values / 1000 / 2) ** 2

for t in t_values:
    noise = torch.randn_like(x_clean)
    x_noisy = torch.sqrt(alpha_bar[t]) * x_clean + torch.sqrt(1 - alpha_bar[t]) * noise
    
    logits = classifier(x_noisy)
    confidence = torch.softmax(logits, dim=1)[0, y_class].item()
    print(f"t={t}: confidence in class {y_class} = {confidence:.4f}")
```

## 🔗 실전 활용

Stable Diffusion 이전 ImageNet classifier guidance (Dhariwal & Nichol 2021):
- ResNet-50을 noisy 이미지에서 fine-tune
- Inference: $w=1.0$ 정도에서 quality 향상 (FID 개선)
- 문제: CIFAR-10, ImageNet 등 강한 분류기 필요

## ⚖️ 가정과 한계

1. **분류기 품질**: $p_\phi(y|x_t)$가 정확해야 함. Noisy 영역에서 학습 필수
2. **Gradient 신뢰성**: High noise (t 큰 경우) 때 gradient 불안정
3. **분리된 학습**: Diffusion + classifier 두 개 모델 필요 → 비용
4. **Mode collapse**: 지나친 guidance 시 다양성 손실

## 📌 핵심 정리

- **Bayes' rule**: $\nabla \log p(x|y) = \nabla \log p(x) + \nabla \log p(y|x)$
- **Score augmentation**: $\tilde{s}_\theta = s_\theta + s \cdot \nabla \log p_\phi(y|x_t)$
- **Gradient-based guidance**: 별도 분류기의 confidence gradient 활용
- **한계**: 외부 classifier 학습 비용, noisy regime에서의 불안정성

## 🤔 생각해볼 문제

1. Classifier guidance에서 guidance scale $s$를 증가시키면 FID는 개선되지만 recall은 저하된다. 왜일까? (hint: mode-seeking vs mode-covering)

<details>
<summary>Hint</summary>
높은 $s$는 분류기 gradient를 크게 증폭하여, 고confidence 샘플들로 집중(sharpening)된다. 분포의 꼬리 부분이 무시되어 recall(다양성)이 떨어진다.
</details>

2. Noisy $x_t$ ($t$ large)에서 분류기를 학습하지 않으면 어떻게 될까?

<details>
<summary>Hint</summary>
분류기가 $x_t$에 과적합하지만, 생성 과정 대부분이 high-noise regime에 있어 신뢰할 수 없는 gradient를 제공한다. 결과: 생성 품질 급격히 저하.
</details>

3. $\nabla_{x_t} \log p(y|x_t)$를 계산할 때 backward pass의 계산량은?

<details>
<summary>Hint</summary>
분류기 foward + backward이므로 O(params). Diffusion step마다 수행하면 전체 생성 시간 2배 이상 증가 가능. (CFG에서는 무조건 step 3배, 하지만 분류기 overhead 없음)
</details>

---

<div align="center">

[◀ 이전](../ch4-ddim/04-probability-flow-ode.md) | [📚 README](../README.md) | [다음 ▶](./02-classifier-free-guidance.md)

</div>
