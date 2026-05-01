# Cross-Attention for Text Conditioning (Stable Diffusion)

## 🎯 핵심 질문

Diffusion UNet에 text conditioning을 통합하는 가장 효율적인 방법은 무엇인가? Cross-attention을 사용하면, image latent와 text embedding을 어떻게 상호작용시킬 수 있을까?

## 🔍 왜 Cross-Attention인가

초기 조건부 diffusion 방법들:
- **Concatenation**: 입력에 직접 조건 연결 (차원 증가, 계산 비용)
- **Spatial conditioning**: 조건을 spatial map으로 변환 (제한적, fine-grained 제어 어려움)
- **FiLM (Feature-wise Linear Modulation)**: scale + shift (표현력 제한)

**Cross-Attention** (Stable Diffusion, Imagen):
- Query: 이미지 feature map ($h_{\text{img}}$)
- Key, Value: 텍스트 embedding ($c_{\text{txt}}$)
- 결과: 각 이미지 위치가 관련 텍스트 토큰에 선택적으로 attend

이는 self-attention (이미지 내)와 cross-attention (텍스트로부터)을 결합하는 구조.

## 📐 수학적 선행 조건

- **Attention mechanism**: $\text{Attn}(Q, K, V) = \text{softmax}(QK^T / \sqrt{d}) V$
- **Transformer blocks**: Multi-head attention, layer normalization
- **Text encoding**: CLIP (256D, 77 tokens) 또는 T5 (768D, varying length)
- **Spatial features**: UNet bottleneck (64×64 또는 32×32 space)

## 📖 직관적 이해

**Self-Attention** (within image):
- 이미지 피처 간 관계 학습 (denoising에 필요)

**Cross-Attention** (image-text):
- 각 이미지 위치가 텍스트 어휘를 "look up"
- Query: "이 위치(position)에서 필요한 정보는?"
- Key/Value: "텍스트에서 어떤 부분이 관련?"
- Attention weights: 각 토큰의 contribution

예: "red dog on grass"
- 빨간색 영역: "red" token에 높은 attention
- 개 영역: "dog" token에 높은 attention
- 잔디 영역: "grass" token에 높은 attention

## ✏️ 엄밀한 정의

**Cross-Attention Block** (Stable Diffusion):

Query projection:
$$Q = W_Q h_{\text{img}} \in \mathbb{R}^{HW \times d}$$

Key, Value projection (텍스트로부터):
$$K = W_K c_{\text{txt}} \in \mathbb{R}^{L \times d}$$
$$V = W_V c_{\text{txt}} \in \mathbb{R}^{L \times d}$$

여기서:
- $h_{\text{img}} \in \mathbb{R}^{HW \times d_{\text{img}}}$: denoising UNet의 feature map (flattened spatial)
- $c_{\text{txt}} \in \mathbb{R}^{L \times d_{\text{txt}}}$: 텍스트 encoder 출력 (L tokens)
- $H, W$: 공간 해상도
- $d$: attention dimension (일반적으로 64-128)

Attention output:
$$\text{CrossAttn}(h_{\text{img}}, c_{\text{txt}}) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) V$$

결과: $\mathbb{R}^{HW \times d}$ 형태로, 다시 이미지 공간으로 reshape.

## 🔬 정리와 증명

**정리 (Cross-Attention의 조건부 Score 효과)**: Cross-attention을 통한 텍스트 conditioning은, implicit하게 조건부 score function $s_\theta(x_t, c)$를 근사한다.

**증명 스케치**:
UNet의 denoising residual 학습:
$$\epsilon_\theta(x_t, c) = \epsilon + s_\theta(x_t, c) \cdot \sqrt{1-\bar\alpha_t}$$

Cross-attention이 있는 경우, 각 spatial location의 feature는 텍스트 context에 의존한다:
$$h_{\text{img}}^{(l)} = \text{CrossAttn}(h_{\text{img}}^{(l-1)}, c_{\text{txt}}) + h_{\text{img}}^{(l-1)}$$

이는 implicit하게 조건부 조정을 수행한다:
$$\nabla_x \epsilon_\theta(x_t, c) \approx s_\theta(x_t, c)$$

또한 classifier-free guidance 관점에서, dropout된 embedding (null vector)과의 차이:
$$\nabla \log p(c|x_t) \approx \epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \text{null})$$

이는 cross-attention 가중치의 차이로 구현된다. $\square$

## 💻 NumPy / PyTorch 구현 검증

### 실험 1: Cross-Attention 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, d_model=64, n_heads=8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(768, d_model)  # 텍스트 차원 (T5/CLIP)
        self.W_v = nn.Linear(768, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x_spatial, c_text):
        """
        x_spatial: (B, N, d_model) - image features (N=H*W)
        c_text: (B, L, 768) - text embeddings (L=seq_len)
        """
        B, N, D = x_spatial.shape
        L = c_text.shape[1]
        
        # Project
        Q = self.W_q(x_spatial)  # (B, N, D)
        K = self.W_k(c_text)      # (B, L, D)
        V = self.W_v(c_text)      # (B, L, D)
        
        # Reshape for multi-head
        Q = Q.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)  # (B, h, N, d_h)
        K = K.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)  # (B, h, L, d_h)
        V = V.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)  # (B, h, L, d_h)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, h, N, L)
        attn_weights = F.softmax(scores, dim=-1)  # (B, h, N, L)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)  # (B, h, N, d_h)
        
        # Reshape back
        context = context.transpose(1, 2).contiguous()  # (B, N, h, d_h)
        context = context.view(B, N, D)  # (B, N, D)
        
        # Output projection
        output = self.W_o(context)
        
        return output, attn_weights

# Test
cross_attn = CrossAttention(d_model=64, n_heads=8)
x_spatial = torch.randn(2, 256, 64)  # (B=2, N=16*16, D=64)
c_text = torch.randn(2, 77, 768)     # (B=2, L=77 tokens, 768)

output, attn = cross_attn(x_spatial, c_text)
print(f"Cross-Attention output shape: {output.shape}")
print(f"Attention weights shape: {attn.shape}")
print(f"Attention is normalized: {attn[0,0,:,:].sum(dim=-1).mean():.4f} (should be 1.0)")
```

### 실험 2: Attention Map 시각화

```python
import numpy as np

def visualize_attention_map(attn_weights, text_tokens, spatial_h=16, spatial_w=16):
    """
    attn_weights: (n_heads, N, L)
    text_tokens: list of str
    """
    # Average over heads
    attn_avg = attn_weights.mean(dim=0)  # (N, L)
    attn_map = attn_avg.view(spatial_h, spatial_w, -1)  # (H, W, L)
    
    print("Attention Map for Key Tokens:")
    important_tokens = [0, 5, 10]  # E.g., "a", "red", "dog"
    
    for token_idx in important_tokens:
        token_map = attn_map[:, :, token_idx]
        max_val = token_map.max().item()
        print(f"  Token {token_idx} ('{text_tokens[token_idx]}'): "
              f"max_attention={max_val:.4f}, "
              f"mean={token_map.mean():.4f}")

# Mock test
text_tokens = ["", "a", "red", "dog", "on", "grass", "."]  # CLIP tokenization
attn_weights = torch.randn(8, 256, 77)

visualize_attention_map(attn_weights, text_tokens)
```

### 실험 3: Conditioning Dropout (null embedding)

```python
def embedding_with_dropout(c_text, dropout_prob=0.1):
    """
    During training: randomly replace embeddings with null token
    During inference: use for unconditioned branch (CFG)
    """
    if dropout_prob > 0 and torch.rand(1).item() < dropout_prob:
        # Replace with learned null embedding or zero
        c_text_dropped = torch.zeros_like(c_text)
        return c_text_dropped
    return c_text

# Training simulation
c_text = torch.randn(4, 77, 768)
for step in range(100):
    c_dropped = embedding_with_dropout(c_text, dropout_prob=0.1)
    # Forward with c_dropped
    if step % 20 == 0:
        dropped_frac = (c_dropped.abs().sum(dim=(1,2)) < 1e-5).float().mean().item()
        print(f"Step {step}: {dropped_frac:.1%} samples have null conditioning")
```

### 실험 4: Text Encoder Comparison (CLIP vs T5)

```python
# Stable Diffusion uses CLIP text encoder
class CLIPTextEncoder:
    def __init__(self):
        self.max_tokens = 77
        self.embedding_dim = 768
    
    def encode(self, prompt):
        # Simplified: in reality uses CLIP tokenizer + transformer
        return torch.randn(1, self.max_tokens, self.embedding_dim)

# Imagen uses T5
class T5TextEncoder:
    def __init__(self):
        self.max_tokens = 256  # variable, usually longer
        self.embedding_dim = 768
    
    def encode(self, prompt):
        return torch.randn(1, self.max_tokens, self.embedding_dim)

# Comparison
prompts = ["a red dog", "a very detailed painting of a cat wearing sunglasses"]

for prompt in prompts:
    clip_emb = CLIPTextEncoder().encode(prompt)
    t5_emb = T5TextEncoder().encode(prompt)
    
    print(f"\nPrompt: '{prompt}'")
    print(f"  CLIP: shape {clip_emb.shape}, max_tokens=77 (fixed)")
    print(f"  T5:   shape {t5_emb.shape}, max_tokens=256 (variable)")
```

## 🔗 실전 활용

**Stable Diffusion (CompVis, 2022)**:
- Text encoder: CLIP ViT-L (768D, 77 tokens)
- UNet: 여러 resolution에서 cross-attention blocks
- Latent: 64×64, 32×32 등에서 cross-attn

**Imagen (Saharia et al., 2022)**:
- Text encoder: Dual (CLIP + T5)
- T5: 더 긴 시퀀스, 더 세밀한 의미 포착
- Super-resolution 단계: 추가 cross-attention

**SD3 (Stability AI, 2024)**:
- Text: CLIP + T5 + Llama ensemble
- MMDiT (Multimodal DiT): 이미지/텍스트 토큰을 동시 처리

## ⚖️ 가정과 한계

1. **Text encoder 고정**: 보통 cross-attention만 fine-tune. 텍스트 encoder 학습은 추가 비용
2. **토큰 길이 제약**: CLIP 77 tokens 제한. 긴 prompt는 truncate 또는 분리 처리
3. **Attention map 해석**: 완벽한 1:1 correspondence 아님. 특히 복잡한 조합(색상+객체)에서
4. **계산 복잡도**: Cross-attention은 self-attention보다 KV 캐싱 어려움

## 📌 핵심 정리

- **Query**: $Q = W_Q h_{\text{img}}$ (이미지 feature)
- **Key, Value**: $K, V = W_{K,V} c_{\text{txt}}$ (텍스트 embedding)
- **Attention**: $\text{Attn} = \text{softmax}(QK^T/\sqrt{d}) V$
- **Conditioning dropout**: CFG를 위해 null embedding 사용
- **Text encoders**: CLIP (강하고 compact), T5 (의미 풍부, 길음)

## 🤔 생각해볼 문제

1. Cross-attention에서 query와 key-value의 차원이 다를 수 있는데 (64 vs 768), 왜 projection하는가?

<details>
<summary>Hint</summary>
원본 text embedding은 768D (CLIP 기준)인데, UNet 계산 비용 제어를 위해 더 낮은 차원 (64-128)으로 project한다. 또한 각 헤드별로 차원을 분할 (multi-head attention)하므로, 내부 일관성이 필요하다.
</details>

2. Attention weights $\text{softmax}(QK^T/\sqrt{d})$를 시각화했을 때, 간단한 prompt ("a dog")에서는 명확한 localization이 보이지만, 복잡한 prompt ("a dog wearing blue jacket sitting on a red carpet")에서는 흐릿해진다. 왜?

<details>
<summary>Hint</summary>
복잡한 prompt에서는 여러 속성(색상, 의류, 자세, 배경)이 얽혀 있어, 각 토큰이 이미지의 다양한 위치에 영향을 미친다. 또한 공간적 관계("sitting on")가 명확한 attention map으로 표현되기 어렵다. 이는 cross-attention의 본질적 한계.
</details>

3. Stable Diffusion에서 negative prompt를 사용할 때 (다음 섹션), cross-attention 가중치는 어떻게 변할까?

<details>
<summary>Hint</summary>
Negative prompt의 embedding도 cross-attention에 입력되지만, guidance에서 subtracted된다. 결과적으로 negative prompt의 attention weights 영역은 보수되고 (낮은 활성화), positive prompt의 attention은 그 영역에서 증폭된다.
</details>

---

<div align="center">

[◀ 이전](./03-cfg-tradeoff.md) | [📚 README](../README.md) | [다음 ▶](./05-negative-prompt.md)

</div>
