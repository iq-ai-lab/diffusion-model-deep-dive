Multimodal Diffusion Transformers (MM-DiT)

## 🎯 핵심 질문

Image와 text를 별개의 modality로 처리하면 정보 손실이 발생한다면, 하나의 sequence로 결합하여 joint self-attention으로 처리할 때 어떤 아키텍처 선택이 최적인가?

## 🔍 왜 Multimodal DiT인가?

**문제 (Stable Diffusion 1-2)**:
- Image latent과 text embedding을 **cross-attention**으로 결합
- Image→text, text→image 정보 흐름이 단방향 + 비대칭
- Text modality가 image 생성을 "제약"하지만, 깊은 상호작용 불가

**해결 (MM-DiT, Esser et al., 2024)**:
- Image patches + text tokens을 **하나의 sequence**로 concatenate
- 모든 토큰에 동일한 self-attention (joint processing)
- Modality-specific weight matrices: $W_Q^i, W_K^i, W_V^i$ (image) vs $W_Q^t, W_K^t, W_V^t$ (text)

**결과**:
- Cross-modal information flow가 자유로움
- SD3, Dall-E 3로 채택되어 text-to-image 품질 대폭 개선

## 📐 수학적 선행 조건

- Transformer attention mechanism (Ch5-01)
- Multi-head attention (Vaswani et al., 2017)
- Cross-attention vs self-attention (Ch5-04)
- Rectified flow (Ch7-02) — SD3는 RF + MM-DiT 조합

## 📖 직관적 이해

**Sequence 구성**:
$$\mathbf{x}_{\text{joint}} = [\mathbf{x}_{\text{img}}, \mathbf{x}_{\text{text}}] \in \mathbb{R}^{(N_{\text{img}} + N_{\text{text}}) \times d}$$

where:
- $\mathbf{x}_{\text{img}}$: image patch tokens (e.g., 64×64 / patch_size=2 = 1024 tokens)
- $\mathbf{x}_{\text{text}}$: text tokens from language model (e.g., 77 tokens, CLIP)

**Modality-specific projection**:
각 modality는 own set of linear projections를 가짐:
$$Q_{\text{img}} = W_Q^i \mathbf{x}_{\text{img}}, \quad K_{\text{img}}, V_{\text{img}} = W_K^i \mathbf{x}_{\text{img}}, W_V^i \mathbf{x}_{\text{img}}$$
$$Q_{\text{text}} = W_Q^t \mathbf{x}_{\text{text}}, \quad K_{\text{text}}, V_{\text{text}} = W_K^t \mathbf{x}_{\text{text}}, W_V^t \mathbf{x}_{\text{text}}$$

**Joint attention** (mixed query/key/value):
$$\text{Attn}(\mathbf{x}_{\text{joint}}) = \text{softmax}\left( \frac{[Q_{\text{img}}, Q_{\text{text}}] [K_{\text{img}}, K_{\text{text}}]^T}{\sqrt{d_k}} \right) [V_{\text{img}}, V_{\text{text}}]$$

모든 image token은 모든 text token을 attend 가능 (그 역도).

## ✏️ 엄밀한 정의

**MM-DiT Block**:

Input: $\mathbf{x}_{\text{joint}} = [\mathbf{x}_{\text{img}}; \mathbf{x}_{\text{text}}] \in \mathbb{R}^{(N_i + N_t) \times d}$

**Layer Norm + Modality-Specific Projection**:
$$\mathbf{q}_i = W_Q^i \text{LN}(\mathbf{x}_{\text{img}})$$
$$\mathbf{q}_t = W_Q^t \text{LN}(\mathbf{x}_{\text{text}})$$

Analogously for $K, V$.

**Joint Self-Attention**:
$$\mathbf{q}_{\text{joint}} = [\mathbf{q}_i; \mathbf{q}_t] \in \mathbb{R}^{(N_i + N_t) \times (h \cdot d_k)}$$
$$\mathbf{k}_{\text{joint}} = [\mathbf{k}_i; \mathbf{k}_t], \quad \mathbf{v}_{\text{joint}} = [\mathbf{v}_i; \mathbf{v}_t]$$

Attention scores:
$$A = \text{softmax}\left( \frac{\mathbf{q}_{\text{joint}} \mathbf{k}_{\text{joint}}^T}{\sqrt{d_k}} \right) \in \mathbb{R}^{(N_i + N_t) \times (N_i + N_t)}$$

Attention output:
$$\text{Attn}(\mathbf{x}_{\text{joint}}) = A \mathbf{v}_{\text{joint}} \in \mathbb{R}^{(N_i + N_t) \times d}$$

**Output projection** (shared across modalities):
$$\mathbf{y}_{\text{joint}} = W_O \text{Attn}(\mathbf{x}_{\text{joint}}) + \mathbf{x}_{\text{joint}}$$

**FFN + residual**:
$$\mathbf{x}_{\text{joint}}^{(\ell+1)} = \text{FFN}(\mathbf{y}_{\text{joint}}) + \mathbf{y}_{\text{joint}}$$

## 🔬 정리와 증명

**정리 (Information Flow in Joint Attention)**:

MM-DiT에서 joint self-attention을 사용하면, 각 modality는 다른 modality의 **완전한 문맥 정보**에 접근 가능하며, cross-attention (Ch5-04)에 비해 정보 손실이 없다.

*증명*:
1. Cross-attention (SD1-2 방식):
   $$\text{CrossAttn}(\mathbf{x}_{\text{img}}, \mathbf{x}_{\text{text}}) = \text{softmax}\left( \frac{Q_{\text{img}} K_{\text{text}}^T}{\sqrt{d_k}} \right) V_{\text{text}}$$
   
   Image query만 text와 interact → asymmetric information flow. Text는 image를 직접 observe 못함.

2. Joint self-attention (MM-DiT):
   $$\text{JointAttn}([\mathbf{x}_{\text{img}}, \mathbf{x}_{\text{text}}])$$
   
   Attention matrix $A \in \mathbb{R}^{(N_i + N_t) \times (N_i + N_t)}$ 는:
   $$A = \begin{bmatrix} A_{ii} & A_{it} \\ A_{ti} & A_{tt} \end{bmatrix}$$
   
   where:
   - $A_{it}$: image to text attention (image가 text 정보 습득)
   - $A_{ti}$: text to image attention (text가 image 정보 습득)
   - $A_{ii}, A_{tt}$: intra-modal self-attention

3. 각 modality의 output:
   $$\mathbf{y}_{\text{img}} = [A_{ii} | A_{it}] [\mathbf{v}_i; \mathbf{v}_t]$$
   $$\mathbf{y}_{\text{text}} = [A_{ti} | A_{tt}] [\mathbf{v}_i; \mathbf{v}_t]$$
   
   Both include full information from both modalities (through $\mathbf{v}_i, \mathbf{v}_t$).

4. Cross-attention 대비 이득:
   - Cross-attention: image는 text의 $V_{\text{text}}$만 습득 (query 제약)
   - Joint attention: image는 text의 모든 representation ($Q, K, V$) 간접 접근
   - Symmetry: text도 image의 정보 습득 → bidirectional flow

5. 정보론적 관점:
   $$I(\mathbf{x}_{\text{img}}; \mathbf{x}_{\text{text}} | \text{joint}) > I(\mathbf{x}_{\text{img}}; \mathbf{x}_{\text{text}} | \text{cross})$$
   
   where $I$ = mutual information.

$\square$

## 💻 NumPy / PyTorch 구현 검증

### 실험 1: Joint Attention 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ModalitySpecificAttention(nn.Module):
    """MM-DiT attention with modality-specific projections"""
    def __init__(self, d_hidden, n_heads, n_img_tokens, n_text_tokens):
        super().__init__()
        assert d_hidden % n_heads == 0
        self.d_k = d_hidden // n_heads
        self.n_heads = n_heads
        self.n_img = n_img_tokens
        self.n_text = n_text_tokens
        self.d_hidden = d_hidden
        
        # Image-specific projections
        self.W_Q_img = nn.Linear(d_hidden, d_hidden)
        self.W_K_img = nn.Linear(d_hidden, d_hidden)
        self.W_V_img = nn.Linear(d_hidden, d_hidden)
        
        # Text-specific projections
        self.W_Q_text = nn.Linear(d_hidden, d_hidden)
        self.W_K_text = nn.Linear(d_hidden, d_hidden)
        self.W_V_text = nn.Linear(d_hidden, d_hidden)
        
        # Shared output projection
        self.W_O = nn.Linear(d_hidden, d_hidden)
    
    def forward(self, x_img, x_text):
        # x_img: (batch, n_img, d_hidden)
        # x_text: (batch, n_text, d_hidden)
        batch = x_img.shape[0]
        
        # Modality-specific projections
        Q_img = self.W_Q_img(x_img).view(batch, self.n_img, self.n_heads, self.d_k).transpose(1, 2)
        K_img = self.W_K_img(x_img).view(batch, self.n_img, self.n_heads, self.d_k).transpose(1, 2)
        V_img = self.W_V_img(x_img).view(batch, self.n_img, self.n_heads, self.d_k).transpose(1, 2)
        
        Q_text = self.W_Q_text(x_text).view(batch, self.n_text, self.n_heads, self.d_k).transpose(1, 2)
        K_text = self.W_K_text(x_text).view(batch, self.n_text, self.n_heads, self.d_k).transpose(1, 2)
        V_text = self.W_V_text(x_text).view(batch, self.n_text, self.n_heads, self.d_k).transpose(1, 2)
        
        # Concatenate for joint attention
        Q_joint = torch.cat([Q_img, Q_text], dim=2)  # (batch, n_heads, n_img+n_text, d_k)
        K_joint = torch.cat([K_img, K_text], dim=2)
        V_joint = torch.cat([V_img, V_text], dim=2)
        
        # Attention
        scores = torch.matmul(Q_joint, K_joint.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply to values
        attn_out = torch.matmul(attn_weights, V_joint)  # (batch, n_heads, n_img+n_text, d_k)
        
        # Merge heads
        attn_out = attn_out.transpose(1, 2).contiguous()
        attn_out = attn_out.view(batch, self.n_img + self.n_text, self.d_hidden)
        
        # Output projection
        output = self.W_O(attn_out)
        
        # Split modalities for output
        y_img = output[:, :self.n_img, :]
        y_text = output[:, self.n_img:, :]
        
        return y_img, y_text

# Test
attn = ModalitySpecificAttention(d_hidden=256, n_heads=8, n_img_tokens=256, n_text_tokens=77)
x_img = torch.randn(2, 256, 256)  # batch=2, n_img=256, d_hidden=256
x_text = torch.randn(2, 77, 256)   # batch=2, n_text=77

y_img, y_text = attn(x_img, x_text)
print(f"Image output: {y_img.shape}")
print(f"Text output: {y_text.shape}")
print(f"Total params: {sum(p.numel() for p in attn.parameters()) / 1e6:.1f}M")
```

### 실험 2: Attention Map 시각화 (Joint vs Cross)

```python
import matplotlib.pyplot as plt

# Joint attention weights
attn_weights = attn_weights.detach()[0, 0, :, :]  # (batch=0, head=0)
# Shape: (n_img + n_text, n_img + n_text) = (256 + 77, 256 + 77) = (333, 333)

# Extract cross-modal interactions
img_to_text = attn_weights[:256, 256:]  # image queries → text keys (256, 77)
text_to_img = attn_weights[256:, :256]  # text queries → image keys (77, 256)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Image → Text
im0 = axes[0].imshow(img_to_text.cpu().numpy(), cmap='viridis', aspect='auto')
axes[0].set_title("Image → Text Attention")
axes[0].set_xlabel("Text token index")
axes[0].set_ylabel("Image patch index")
plt.colorbar(im0, ax=axes[0])

# Text → Image
im1 = axes[1].imshow(text_to_img.cpu().numpy(), cmap='viridis', aspect='auto')
axes[1].set_title("Text → Image Attention")
axes[1].set_xlabel("Image patch index")
axes[1].set_ylabel("Text token index")
plt.colorbar(im1, ax=axes[1])

plt.tight_layout()
plt.savefig("mm_dit_attention.png")
print("Saved attention maps to mm_dit_attention.png")

# Statistics
print(f"Image→Text attention mean: {img_to_text.mean():.4f}")
print(f"Text→Image attention mean: {text_to_img.mean():.4f}")
print(f"Image→Text attention entropy: {-(img_to_text * (img_to_text.log() + 1e-10)).sum(dim=-1).mean():.4f}")
```

### 실험 3: Modality-Specific vs Shared Weights 비교

```python
# MM-DiT: modality-specific Q, K, V
# 대안: shared Q, K, V (표준 transformer)

class SharedAttention(nn.Module):
    """Standard transformer (shared projections)"""
    def __init__(self, d_hidden, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_hidden, n_heads, batch_first=True)
    
    def forward(self, x_img, x_text):
        x_joint = torch.cat([x_img, x_text], dim=1)
        attn_out, _ = self.attn(x_joint, x_joint, x_joint)
        y_img = attn_out[:, :x_img.shape[1], :]
        y_text = attn_out[:, x_img.shape[1]:, :]
        return y_img, y_text

# Compare parameters
mm_dit_params = sum(p.numel() for p in attn.parameters())
shared_params = sum(p.numel() for p in SharedAttention(256, 8).parameters())

print(f"MM-DiT modality-specific: {mm_dit_params / 1e6:.3f}M params")
print(f"Standard shared attention: {shared_params / 1e6:.3f}M params")
print(f"Overhead: {(mm_dit_params / shared_params - 1) * 100:.1f}%")
print("\n→ Modality-specific weights add ~50% params")
print("→ Benefit: each modality learns specialized representations")
```

## 🔗 실전 활용

**Stable Diffusion 3 (Esser et al., 2024)**:
- MM-DiT backbone + Rectified Flow (Ch7-02)
- Dual text encoder: CLIP-L + OpenCLIP-bigG
- Joint training: image + text tokens

**코드 예시**:
```python
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium",
    torch_dtype=torch.float16
).to("cuda")

prompt = "A serene landscape with mountains and a lake at sunset"
image = pipe(
    prompt=prompt,
    num_inference_steps=50,
    guidance_scale=7.5,
    height=1024,
    width=1024
).images[0]

image.save("output_sd3.png")
```

## ⚖️ 가정과 한계

1. **Modality 사이의 token 수 불균형**: Image tokens >> text tokens (1024 vs 77). Attention complexity $\propto (1024 + 77)^2$ → 주로 이미지가 자기 자신에 attend
2. **Learnable positional encoding**: Image + text를 concatenate하면, 위치 정보 해석 모호 (image 끝과 text 시작 경계)
3. **Compute cost**: Cross-attention보다 $O((N_i + N_t)^2)$ complexity 증가 → inference 시간 증가
4. **Text-image 정렬 가정**: Joint sequence가 모든 조건화 시나리오에 최적이라는 보장 없음

## 📌 핵심 정리

1. **MM-DiT = joint sequence (image + text) + modality-specific Q, K, V + shared output**
2. Bidirectional information flow (text→image, image→text 동시)
3. Cross-attention 대비 **정보 손실 제거**, 상호작용 깊화
4. **추가 계산** ($O((N_i + N_t)^2)$) vs **품질 향상** (trade-off)

## 🤔 생각해볼 문제

1. **왜 Q, K, V만 modality-specific이고, output projection $W_O$는 shared인가?**
   <details>
   <summary>힌트</summary>
   Q, K, V는 각 modality의 "관점"을 학습 → 다를 수 있음. 하지만 output은 공동 representation space로 project → modality-agnostic. Shared $W_O$는 모델 크기 제약 + 유연한 정보 혼합 허용.
   </details>

2. **Image tokens가 text tokens보다 훨씬 많으면 (1024 vs 77), text 정보가 attention에서 "묻힐" 가능성은?**
   <details>
   <summary>힌트</summary>
   Attention weight matrix에서 text 행(row)의 norm이 image 행보다 작을 수 있음. 실제로 SD3는 text tokens에 특별한 가중치 또는 별도의 normalization 적용 가능. 또는 두 modality를 separate attention head로 처리하되 마지막 layer에서만 mixing.
   </details>

3. **MM-DiT에서 image와 text의 순서 (image-first vs text-first)가 결과에 영향을 미칠까?**
   <details>
   <summary>힌트</summary>
   Self-attention은 permutation-invariant on input order (이론적으로). 실제로는 positional encoding과 attention pattern의 미묘한 차이로 slight 영향 가능. 현재 표준: image 먼저, then text (image가 더 많은 token).
   </details>

---

<div align="center">

[◀ 이전](./03-dit.md) | [📚 README](../README.md) | [다음 ▶](./05-cascaded-diffusion.md)

</div>
