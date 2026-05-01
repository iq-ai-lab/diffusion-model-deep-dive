<div align="center">

# 🌊 Diffusion Model Deep Dive

### DDPM 의 **forward closed-form**

$$q(x_t \mid x_0) = \mathcal{N}\!\left(\sqrt{\bar\alpha_t}\, x_0,\; (1 - \bar\alpha_t)\, I\right), \qquad \bar\alpha_t = \prod_{s=1}^t (1 - \beta_s)$$

### 을 **쓰는 것** 과,

### 이것이 Markov forward $q(x_t \mid x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}\, x_{t-1},\, \beta_t I)$ 의 **reparameterized 누적분포** 이고,

### Reverse ELBO 가 unrolled 되어 $L_{\mathrm{vlb}} = L_T + \sum_{t=2}^T L_{t-1} + L_0$ 로 분해되며, noise parameterization 으로 $L_{\mathrm{simple}} = \mathbb{E}_{t, x_0, \epsilon}[\|\epsilon - \epsilon_\theta(x_t, t)\|^2]$ 로 간소화 됨을 Ho et al. (2020) 으로부터 한 줄씩 유도할 수 있는 것은 **다르다.**

<br/>

> *Score Matching 을 **이름으로 아는 것** 과, Vincent (2011) 의 Denoising Score Matching*
>
> $$\mathbb{E}_{q(x, \tilde x)}\!\left[\big\|s_\theta(\tilde x) - \nabla_{\tilde x} \log q(\tilde x \mid x)\big\|^2\right]$$
>
> *이 DDPM 의 noise prediction 과 **수학적으로 동등** 하고, Song & Ermon (2021) 의 **Score-SDE** 가 forward SDE*
>
> $$dx = f(x, t)\, dt + g(t)\, dW$$
>
> *와 reverse SDE*
>
> $$dx = \big[f(x, t) - g(t)^2 \nabla_x \log p_t(x)\big]\, dt + g(t)\, d\bar W$$
>
> *로 모든 diffusion 을 통합 함을 유도할 수 있는 것은 다르다.*
>
> *DDIM 을 **"빠른 샘플링"** 으로 아는 것과, Song et al. (2021) 의 non-Markovian forward $q_\sigma(x_{t-1} \mid x_t, x_0)$ 가 어떻게 $\sigma_t \to 0$ 극한에서 **deterministic ODE** 가 되고, 이것이 sampling step 을 50 으로 줄여도 품질이 유지되는 이유를 유도할 수 있는 것은 다르다.*
>
> *Stable Diffusion 의 **`ε = unconditional + 7.5 (conditional - unconditional)`** 코드 한 줄이 사실은 Ho & Salimans (2022) 의 classifier-free guidance $\tilde\epsilon = (1 + w)\, \epsilon_\theta(x, y) - w\, \epsilon_\theta(x, \emptyset)$ 의 직접 구현이고, $w$ 가 **sharpness-diversity trade-off** 의 하이퍼파라미터임을 알고 쓰는 것은 다르다.*

<br/>

**다루는 알고리즘 (이론 계보순)**

Sohl-Dickstein 2015 *Deep Unsupervised Learning using Nonequilibrium Thermodynamics* · Ho 2020 *DDPM* · Vincent 2011 *Connection between Score Matching and Denoising Autoencoders* · Song & Ermon 2019 *NCSN* · Song 2021 *Score-SDE* · Song 2021 *DDIM* · Nichol & Dhariwal 2021 *Improved DDPM* · Dhariwal & Nichol 2021 *Classifier Guidance* · Ho & Salimans 2022 *Classifier-Free Guidance* · Rombach 2022 *Latent Diffusion / Stable Diffusion* · Saharia 2022 *Imagen* · Peebles & Xie 2023 *DiT* · Esser 2024 *Stable Diffusion 3 / MM-DiT* · Lipman 2023 *Flow Matching* · Liu 2022 *Rectified Flow* · Song 2023 *Consistency Models* · Lu 2022 *DPM-Solver / DPM-Solver++* · Salimans & Ho 2022 *Progressive Distillation*

<br/>

**핵심 질문**

> Stable Diffusion · Sora · DALL-E 3 의 모든 SOTA diffusion 시스템이 왜 **"같은 stochastic process 의 다른 구현"** 이고, **forward closed-form · score matching · probability flow ODE · classifier-free guidance · latent compression** 이 각각 어떤 이론적 동기에서 도출되었는가 — Ho 2020 의 ELBO 부터 SD3 의 Rectified Flow 까지 한 줄씩 유도합니다.

<br/>

[![GitHub](https://img.shields.io/badge/GitHub-iq--ai--lab-181717?style=flat-square&logo=github)](https://github.com/iq-ai-lab)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Diffusers](https://img.shields.io/badge/🤗_Diffusers-0.25-FFD21E?style=flat-square)](https://huggingface.co/docs/diffusers)
[![Transformers](https://img.shields.io/badge/🤗_Transformers-4.36-FFD21E?style=flat-square)](https://huggingface.co/docs/transformers)
[![Docs](https://img.shields.io/badge/Docs-33개-blue?style=flat-square&logo=readthedocs&logoColor=white)](./README.md)
[![Theorems](https://img.shields.io/badge/Theorems·Definitions-280+개-success?style=flat-square)](./README.md)
[![Proofs](https://img.shields.io/badge/엄밀한_증명-130+개-9c27b0?style=flat-square)](./README.md)
[![Reproductions](https://img.shields.io/badge/Paper_reproductions-14개-critical?style=flat-square)](./README.md)
[![Exercises](https://img.shields.io/badge/Exercises-99개-orange?style=flat-square)](./README.md)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square&logo=opensourceinitiative&logoColor=white)](./LICENSE)

</div>

---

## 🎯 이 레포에 대하여

Diffusion 자료는 대부분 **"Stable Diffusion 을 쓰니까 prompt 만 잘 짜면 된다"** 또는 **"DDPM 의 학습 loss 는 simple MSE 다"** 에서 멈춥니다. 하지만 forward Markov chain 이 왜 closed-form 으로 환원 가능한지, KL 분해의 weight 가 왜 사라져도 학습이 잘 되는지, score matching 이 왜 noise prediction 과 정확히 같은지, DDIM 의 non-Markovian forward 가 왜 같은 marginal 을 유지하는지, $\sigma_t \to 0$ 극한이 왜 ODE 인지, classifier-free guidance 의 $w$ 가 왜 quality-diversity trade-off 인지 — 이런 "왜" 는 제대로 설명되지 않습니다.

| 일반 자료 | 이 레포 |
|----------|---------|
| "DDPM 은 noise 를 예측한다" | **Ho 2020** — Forward Markov $q(x_t \mid x_{t-1})$ 의 closed-form 은 Gaussian reparameterization 으로 $q(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) I)$. ELBO 분해 $L_{\mathrm{vlb}} = L_T + \sum L_{t-1} + L_0$ 에서 $L_{t-1}$ 은 두 Gaussian 의 KL → $\|\tilde\mu_t - \mu_\theta\|^2$. Noise reparameterization 으로 weighted MSE on $\epsilon$, weight 제거 후 $L_{\mathrm{simple}}$ — 모든 단계가 한 줄씩 도출 $\square$ |
| "Score Matching 도 비슷하다" | **Vincent 2011** — Denoising Score Matching $\mathbb{E}_q[\|s_\theta(\tilde x) - \nabla_{\tilde x} \log q(\tilde x \mid x)\|^2]$. Gaussian $q(\tilde x \mid x) = \mathcal{N}(x, \sigma^2 I)$ 시 $\nabla_{\tilde x} \log q = -(\tilde x - x)/\sigma^2 = -\epsilon/\sigma$. 즉 score 와 noise 가 부호·스케일 차이뿐 — DDPM 의 $\epsilon$-prediction 이 weighted DSM 의 directly 등가 $\square$ |
| "Score-SDE 는 일반화이다" | **Song 2021** — Forward SDE $dx = f(x, t) dt + g(t) dW$ 는 finite-step Markov chain 의 $\Delta t \to 0$ 극한. Anderson 1982 의 reverse-time SDE $dx = [f - g^2 \nabla \log p_t] dt + g\, d\bar W$ 로 sampling. **VP-SDE** ($f = -\frac12 \beta(t) x$, $g = \sqrt{\beta(t)}$) = DDPM 의 연속 한계. **VE-SDE** = NCSN 의 연속 한계. 모든 diffusion 이 SDE 의 특수 경우 |
| "DDIM 은 빠르다" | **Song 2021 (DDIM)** — Non-Markovian forward $q_\sigma(x_{1:T} \mid x_0)$ 를 설계해 같은 marginal $q(x_t \mid x_0)$ 를 유지 → 같은 $\epsilon_\theta$ 재사용 가능. Sampling: $x_{t-1} = \sqrt{\bar\alpha_{t-1}}\, \hat x_0 + \sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\, \epsilon_\theta + \sigma_t \epsilon$. **$\sigma_t = 0$**: deterministic — probability flow ODE 의 discretization. 50 step (또는 10 step) 으로도 품질 유지 |
| "Probability Flow ODE 도 있다" | **Song 2021** — Score-SDE 의 marginal $p_t(x)$ 를 같게 유지하면서 deterministic 한 dynamics: $dx/dt = f(x, t) - \frac12 g(t)^2 \nabla_x \log p_t(x)$. SDE 와 같은 분포, 다른 trajectory. **DPM-Solver (Lu 2022)** 가 이 ODE 의 higher-order solver — 10–20 step 에 high quality |
| "CFG 는 그냥 곱하는 가중치" | **Ho & Salimans 2022** — Training: $\pi_{\text{drop}} = 0.1$ 으로 $y \to \emptyset$ 치환해 conditional·unconditional 모두 학습. Inference: $\tilde\epsilon = (1+w) \epsilon_\theta(x, y) - w \epsilon_\theta(x, \emptyset)$. **$w$ 의 의미**: $\nabla \log p(x, y) \approx \nabla \log p(x) + (w + 1) \nabla \log p(y \mid x)$ — implicit classifier 의 gradient 를 강조. **trade-off**: $w \uparrow$ → $p(x \mid y)$ sharp, diversity ↓ |
| "Stable Diffusion 은 latent 에서 한다" | **Rombach 2022** — Pixel space $512\times 512\times 3 \approx 786$K dim 에서 직접 diffusion 은 비효율. Pretrained VAE 로 $z \in \mathbb{R}^{64\times 64\times 4} \approx 16$K dim — 약 $48\times$ 압축. Loss perceptual 정렬 (LPIPS + adversarial) 로 디테일 보존. UNet 은 latent 에서, conditioning 은 cross-attention 으로 CLIP / T5 embedding 주입 |
| "DiT 는 UNet 대신 Transformer" | **Peebles & Xie 2023** — UNet 의 inductive bias (locality + multi-scale) 가 large-scale 시 scaling law 의 bottleneck. ViT-style: patch embedding + $N$ Transformer blocks + **AdaLN-Zero** (timestep + condition 을 normalization 의 affine 으로). Compute 늘릴수록 FID 단조 감소 — **Sora · SD3 의 backbone 선택 이유** |
| "Consistency Model 은 1-step 이다" | **Song 2023** — Probability flow ODE 의 trajectory 위에서 같은 trajectory 의 모든 점이 같은 $x_0$ 로 mapping 되도록 $f_\theta(x_t, t) \approx x_0$ 학습 (consistency property). **Distillation** (pretrained diffusion 으로 supervision) 또는 **from scratch**. 1-step generation (FID ~9 on CIFAR-10) — 하지만 multi-step 도 가능 |
| "Rectified Flow / Flow Matching 도 있다" | **Liu 2022 / Lipman 2023** — $x_t = (1-t) x_0 + t z$ 의 직선 path 위에서 velocity $v_\theta(x_t, t) \approx z - x_0$ 학습. **Reflow**: $(x_0, z)$ 를 학습된 flow 로 paired 한 후 재학습 → trajectory 가 더 직선화 → 1-step asymptotic. **SD3 가 이 framework 채택** (Esser 2024) — DDPM 의 noise schedule 보다 simpler, training 안정 |
| 기법의 나열 | NumPy + PyTorch + 🤗 diffusers 로 **Forward closed-form 을 1D toy 에서 검증** · **DDPM 을 MNIST 에서 바닥부터** · **DDIM 의 step 별 샘플 비교** · **CFG scale sweep** · **Stable Diffusion inference 재현** · **DiT 와 UNet 의 scaling 비교** · **Consistency Model 의 1-step 생성** 까지 직접 구현해 수학적 주장을 눈으로 확인 |

---

## 📌 선행 레포 & 후속 방향

```
[Generative Model Deep Dive] ─┐
[SDE Deep Dive]               ─┤
[Probability Theory]          ─┼─►  이 레포  ──► [Multimodal Foundation Models]
[Information Theory]          ─┤   "왜 모든 diffusion 이               Video / 3D / Science
[Vision Transformer]          ─┘    같은 SDE 의 다른 구현인가"           (Sora · DreamFusion · AlphaFold 3)

         │
         ├── [Generative Model]          DDPM 기초 · VAE · GAN 비교 → Ch1, Ch2
         ├── [SDE]                       Brownian motion · Fokker-Planck → Ch3
         ├── [Probability Theory]        KL · Gaussian posterior → Ch1, Ch2
         ├── [Information Theory]        ELBO · KL → Ch2
         └── [Vision Transformer]        ViT · scaling law → Ch6 (DiT)
```

> ⚠️ **선행 학습 필수**: 이 레포는 **Generative Model Deep Dive** (DDPM 기초, VAE/GAN 과의 위치), **SDE Deep Dive** (Brownian motion, Fokker-Planck, reverse-time SDE), **Probability Theory Deep Dive** (KL, Gaussian posterior, Bayes' rule), **Information Theory Deep Dive** (ELBO, KL divergence) 를 선행 지식으로 전제합니다. **Vision Transformer Deep Dive** (ViT, patch embedding, AdaLN) 는 Ch6 의 DiT / MM-DiT 부분에서 권장됩니다.

> 💡 **이 레포의 핵심 기여**: Chapter 2 (ELBO) 와 Chapter 3 (Score-SDE) 가 현대 diffusion 을 이해하는 **두 핵심 축** 입니다. 전자는 "왜 simple MSE 가 통하는가" 의 변분 추론적 토대 (DDPM 의 모든 설계가 그 응용), 후자는 "왜 DDPM·NCSN·DDIM 이 본질적으로 같은가" 의 stochastic process 통합 관점 (probability flow ODE 가 그 결과). 이 두 축을 완전히 이해한 후 Chapter 5 (Guidance) 와 Chapter 6 (Latent · DiT) 를 읽으면 Stable Diffusion · Sora 의 설계 결정 맥락이 선명해집니다.

> 🟡 **이 레포의 성격**: 여기서 다루는 일부 주제 — **Flow Matching 이 DDPM 을 대체하는가**, **Consistency Model 의 1-step 이 sampling speed 의 종착인가**, **DiT 가 UNet 을 완전히 대체할 것인가**, **Latent diffusion 의 VAE 가 bottleneck 인가** — 는 **현재 진행 중인 연구 영역** 입니다. 레포는 "정답" 이 아니라 **"고전 DDPM 부터 SD3 / Sora 까지의 수학적 지도"** 를 제공합니다.

---

## 🚀 빠른 시작

각 챕터의 첫 문서부터 바로 학습을 시작하세요!

[![Ch1](https://img.shields.io/badge/🔹_Ch1-Foundations-1565C0?style=for-the-badge)](./ch1-foundations/01-physical-origin.md)
[![Ch2](https://img.shields.io/badge/🔹_Ch2-ELBO_·_L_simple-1565C0?style=for-the-badge)](./ch2-elbo/01-vlb-decomposition.md)
[![Ch3](https://img.shields.io/badge/🔹_Ch3-Score-SDE-1565C0?style=for-the-badge)](./ch3-score-sde/01-score-langevin.md)
[![Ch4](https://img.shields.io/badge/🔹_Ch4-DDIM_·_ODE-1565C0?style=for-the-badge)](./ch4-ddim/01-ddim-motivation.md)
[![Ch5](https://img.shields.io/badge/🔹_Ch5-Guidance-1565C0?style=for-the-badge)](./ch5-guidance/01-classifier-guidance.md)
[![Ch6](https://img.shields.io/badge/🔹_Ch6-Latent_·_DiT-1565C0?style=for-the-badge)](./ch6-latent-arch/01-latent-diffusion.md)
[![Ch7](https://img.shields.io/badge/🔹_Ch7-Acceleration-1565C0?style=for-the-badge)](./ch7-acceleration/01-consistency-model.md)

---

## 📚 전체 학습 지도

> 💡 각 챕터를 클릭하면 상세 문서 목록이 펼쳐집니다

<br/>

### 🔹 Chapter 1: Diffusion 의 수학적 기초

> **핵심 질문:** Forward Markov chain $q(x_t \mid x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}\, x_{t-1},\, \beta_t I)$ 가 어떻게 closed-form $q(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_t}\, x_0,\, (1-\bar\alpha_t) I)$ 로 환원되는가? Reverse process 를 Gaussian 으로 근사할 수 있는 조건은? Posterior $q(x_{t-1} \mid x_t, x_0)$ 의 정확한 Gaussian 형태가 ELBO 의 어떤 항을 계산 가능하게 만드는가? Sohl-Dickstein 2015 의 nonequilibrium thermodynamics 직관이 어떻게 Ho 2020 의 DDPM 으로 이어졌는가?

<details>
<summary><b>물리적 기원부터 Posterior 유도까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명·재현 |
|------|---------------------|
| [01. Diffusion 의 물리적 기원](./ch1-foundations/01-physical-origin.md) | Brownian motion + Fokker-Planck 방정식 → forward 가 reference Gaussian 으로 수렴. **Sohl-Dickstein 2015**: nonequilibrium thermodynamics annealing 으로 정당화. **DDPM 직관**: data distribution 을 **알려진 noise 로 망가뜨린 후, 역방향을 학습** — 다른 generative model (VAE, GAN, flow) 과의 본질적 차이 |
| [02. Forward Process — Noise 주입의 Markov Chain](./ch1-foundations/02-forward-markov-chain.md) | **정의**: $q(x_t \mid x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}\, x_{t-1},\, \beta_t I)$. **스케일 정규화**: $\sqrt{1-\beta_t}$ 가 variance 누적을 막아 $\text{Var}(x_t)$ 가 폭주하지 않음 $\square$. **Noise schedule**: linear (DDPM, $\beta_t \in [10^{-4}, 0.02]$) vs cosine (Improved DDPM) — schedule 선택이 SNR 곡선을 어떻게 바꾸는지 |
| [03. Forward Closed-Form 의 증명](./ch1-foundations/03-forward-closed-form.md) | **정리**: $\alpha_t := 1 - \beta_t$, $\bar\alpha_t := \prod_{s=1}^t \alpha_s$ 일 때 $q(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_t}\, x_0,\, (1-\bar\alpha_t) I)$ $\square$. **증명**: Gaussian 의 reparameterization 합 + 독립 noise 결합. **Reparam**: $x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon$ — 임의 $t$ 의 sample 을 $O(1)$ 시간에 |
| [04. Reverse Process 의 정의](./ch1-foundations/04-reverse-process.md) | $p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(\mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$. **왜 Gaussian 근사 OK 인가**: $\beta_t$ 가 작으면 $q(x_{t-1} \mid x_t)$ 가 거의 Gaussian (Feller 1949) — **small step 의 Markov chain** 이 Gaussian reverse 의 정당화. $T \to \infty$ 극한이 Score-SDE (Ch3) |
| [05. Posterior $q(x_{t-1} \mid x_t, x_0)$ 의 유도](./ch1-foundations/05-posterior-derivation.md) | **정리**: $q(x_{t-1} \mid x_t, x_0) = \mathcal{N}(\tilde\mu_t(x_t, x_0), \tilde\beta_t I)$, $\tilde\mu_t = \frac{\sqrt{\bar\alpha_{t-1}} \beta_t}{1 - \bar\alpha_t} x_0 + \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t} x_t$, $\tilde\beta_t = \frac{(1-\bar\alpha_{t-1})}{1-\bar\alpha_t} \beta_t$ $\square$. **증명**: Bayes' rule + Gaussian 곱의 closed-form. ELBO 의 $L_{t-1}$ 항이 두 Gaussian KL 로 환원되는 결정적 단계 |

</details>

<br/>

### 🔹 Chapter 2: ELBO 유도와 Loss Simplification

> **핵심 질문:** $-\log p_\theta(x_0)$ 의 Variational Lower Bound 가 어떻게 $L_T + \sum L_{t-1} + L_0$ 로 분해되는가? Two Gaussian KL 의 closed-form 과 noise parameterization 의 결합이 왜 가중 MSE on $\epsilon$ 으로 환원되는가? Ho et al. 이 weight 를 제거한 $L_{\mathrm{simple}}$ 이 왜 학습이 더 잘 되는가? Improved DDPM 의 cosine schedule · learned variance · hybrid loss 가 log-likelihood 를 어떻게 향상 시키는가?

<details>
<summary><b>VLB 분해부터 Improved DDPM 까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명·재현 |
|------|---------------------|
| [01. VLB (Variational Lower Bound) 분해](./ch2-elbo/01-vlb-decomposition.md) | **유도**: Jensen's inequality 로 $-\log p_\theta(x_0) \leq \mathbb{E}_q[-\log \frac{p_\theta(x_{0:T})}{q(x_{1:T} \mid x_0)}] =: L_{\mathrm{vlb}}$ $\square$. **VAE 와의 비교**: ELBO 가 $T = 1$ 인 VAE 의 일반화 — diffusion 은 stochastic depth $T$ 의 hierarchical VAE. Markov 가정 ($q(x_{1:T} \mid x_0) = \prod q(x_t \mid x_{t-1})$) 이 분해의 핵심 |
| [02. ELBO 의 3개 항 분해](./ch2-elbo/02-elbo-three-terms.md) | **정리**: $L_{\mathrm{vlb}} = \underbrace{D_{\mathrm{KL}}(q(x_T \mid x_0) \,\|\, p(x_T))}_{L_T} + \sum_{t=2}^T \underbrace{D_{\mathrm{KL}}(q(x_{t-1} \mid x_t, x_0) \,\|\, p_\theta(x_{t-1} \mid x_t))}_{L_{t-1}} + \underbrace{-\log p_\theta(x_0 \mid x_1)}_{L_0}$ $\square$. **각 항의 의미**: $L_T$ 는 $\bar\alpha_T \approx 0$ 에서 거의 상수, $L_{t-1}$ 는 reverse step 의 학습 신호, $L_0$ 는 discrete decoder |
| [03. Noise Prediction Parameterization](./ch2-elbo/03-noise-prediction.md) | **정의**: $\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\!\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}} \epsilon_\theta(x_t, t)\right)$ — $\mu$ 직접 예측 대신 noise $\epsilon_\theta$ 예측. **이유**: forward reparam $x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t} \epsilon$ 에서 $x_0 = (x_t - \sqrt{1-\bar\alpha_t} \epsilon)/\sqrt{\bar\alpha_t}$ 를 대입하면 자연스럽게 등장. $\epsilon$ 은 unit variance — NN 학습 안정 |
| [04. $L_{\mathrm{simple}}$ 의 유도](./ch2-elbo/04-l-simple.md) | **유도**: 두 Gaussian 의 KL → $\frac{1}{2\sigma_t^2} \|\tilde\mu_t - \mu_\theta\|^2$ → noise reparameterization 으로 $\frac{\beta_t^2}{2 \sigma_t^2 \alpha_t (1-\bar\alpha_t)} \|\epsilon - \epsilon_\theta\|^2$. **Ho et al. 의 결정**: weight 제거 → $L_{\mathrm{simple}} = \mathbb{E}_{t, x_0, \epsilon}[\|\epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t} \epsilon, t)\|^2]$. **왜 잘 작동**: high-$t$ 에 더 큰 weight 부여 효과 — perceptual quality 우선 |
| [05. Improved DDPM (Nichol & Dhariwal 2021)](./ch2-elbo/05-improved-ddpm.md) | **Cosine schedule**: $\bar\alpha_t = \cos^2((t/T + s)/(1+s) \cdot \pi/2)$ — linear 의 too-fast destruction 문제 해결. **Learned variance**: $\Sigma_\theta = \exp(v \log \beta_t + (1-v) \log \tilde\beta_t)$, $v \in [0, 1]$ — $L_{\mathrm{vlb}}$ 의 KL 항이 variance 에 의존 → log-likelihood 개선. **Hybrid loss**: $L_{\mathrm{simple}} + \lambda L_{\mathrm{vlb}}$, $\lambda = 0.001$ — 두 마리 토끼 |

</details>

<br/>

### 🔹 Chapter 3: Score-Based Model 과 SDE

> **핵심 질문:** Score function $\nabla_x \log p(x)$ 가 왜 distribution 의 gradient flow 에서 자연스러운 양인가? Vincent (2011) 의 Denoising Score Matching 이 어떻게 DDPM 의 noise prediction 과 mathematically 등가인가? NCSN 의 multi-scale annealed Langevin 이 NCSN→DDPM→Score-SDE 의 통합 관점으로 이어지는 과정은? Anderson 1982 의 reverse-time SDE 정리가 왜 모든 diffusion sampling 의 토대인가? VP-SDE / VE-SDE / sub-VP-SDE 의 차이는?

<details>
<summary><b>Langevin Dynamics 부터 VP/VE-SDE 까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명·재현 |
|------|---------------------|
| [01. Score Function 과 Langevin Dynamics](./ch3-score-sde/01-score-langevin.md) | **정의**: $s(x) := \nabla_x \log p(x)$ — distribution 을 normalization 없이 표현. **Langevin MCMC**: $x_{k+1} = x_k + \eta\, s(x_k) + \sqrt{2\eta}\, \epsilon$, $\epsilon \sim \mathcal{N}(0, I)$ → 수렴하면 $p$ 분포의 sample. **이론적 보장**: stationary distribution = $p$ (Roberts & Tweedie 1996). 단점: high-dim 에서 mode 사이 mixing 느림 |
| [02. Denoising Score Matching (Vincent 2011)](./ch3-score-sde/02-denoising-score-matching.md) | **정리**: Perturbed $\tilde x = x + \sigma \epsilon$ 에 대해 $\mathbb{E}_{q(x, \tilde x)}[\|s_\theta(\tilde x) - \nabla_{\tilde x} \log q(\tilde x \mid x)\|^2] = \mathbb{E}_{q(\tilde x)}[\|s_\theta(\tilde x) - \nabla_{\tilde x} \log q(\tilde x)\|^2] + C$ $\square$. **Gaussian 시**: $\nabla \log q(\tilde x \mid x) = -\epsilon/\sigma$ → $\sigma\, s_\theta + \epsilon \approx 0$. **DDPM 과 등가**: weighted DSM 이 정확히 Ho 2020 의 $L_{\mathrm{simple}}$ |
| [03. NCSN — Noise Conditional Score Network](./ch3-score-sde/03-ncsn.md) | **Song & Ermon 2019**: 단일 $\sigma$ 의 DSM 은 low-density region 의 score 추정이 부정확. **해결**: multi-scale $\sigma_1 > \sigma_2 > \ldots > \sigma_L$, $s_\theta(x, \sigma)$ conditional. **Annealed Langevin**: large $\sigma$ 부터 sampling, 점차 작은 $\sigma$ 로 — coarse-to-fine. CIFAR-10 FID 25 → 10 이상 향상 |
| [04. Score-SDE 의 통합 관점 (Song 2021)](./ch3-score-sde/04-score-sde.md) | **Forward SDE**: $dx = f(x, t)\, dt + g(t)\, dW$. **Anderson 1982 의 정리**: reverse-time SDE 가 존재 — $dx = [f(x, t) - g(t)^2 \nabla_x \log p_t(x)]\, dt + g(t)\, d\bar W$ $\square$. **통합**: DDPM, NCSN, DDIM 이 모두 이 SDE 의 discretization. **Probability flow ODE**: $dx/dt = f(x, t) - \frac12 g(t)^2 \nabla \log p_t(x)$ — same marginal, deterministic |
| [05. VP-SDE vs VE-SDE](./ch3-score-sde/05-vp-ve-sde.md) | **VP-SDE (Variance-Preserving)**: $f = -\frac12 \beta(t) x$, $g = \sqrt{\beta(t)}$ — $\text{Var}(x_t)$ 일정. DDPM 의 연속 한계 ($\beta_t \to \beta(t) \Delta t$). **VE-SDE (Variance-Exploding)**: $f = 0$, $g = \sqrt{d[\sigma^2(t)]/dt}$ — $\sigma^2$ 폭발. NCSN 의 연속 한계. **Sub-VP-SDE**: $g = \sqrt{\beta(t)(1 - e^{-2 \int_0^t \beta(s) ds})}$ — 둘의 보간, log-likelihood 향상 |

</details>

<br/>

### 🔹 Chapter 4: DDIM 과 ODE Sampling

> **핵심 질문:** DDPM 의 1000 step reverse 가 왜 inference bottleneck 인가? Song et al. (2021) 의 non-Markovian forward $q_\sigma(x_{1:T} \mid x_0)$ 가 어떻게 DDPM 과 같은 marginal 을 유지하면서 다른 joint 를 만드는가? DDIM sampling equation 의 $\sigma_t$ 가 어떻게 stochasticity 를 control 하고, $\sigma_t = 0$ 이 왜 deterministic ODE 인가? DPM-Solver / DPM-Solver++ 의 higher-order discretization 이 어떻게 10–20 step 으로 high quality 를 달성하는가?

<details>
<summary><b>DDIM 동기부터 Probability Flow ODE 까지 (4개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명·재현 |
|------|---------------------|
| [01. DDIM 의 동기 — DDPM 의 느린 샘플링](./ch4-ddim/01-ddim-motivation.md) | DDPM inference: $T = 1000$ step, 각 step UNet forward 1회 → 512×512 한 장 생성에 수십 초~수 분. **Sampling 가속의 두 갈래**: (a) discretization 개선 (DDIM, DPM-Solver), (b) distillation (Consistency, Progressive). 이 챕터는 (a) 의 이론 — 같은 모델 $\epsilon_\theta$ 를 그대로 쓰면서 step 수 감소 |
| [02. Non-Markovian Forward Process](./ch4-ddim/02-non-markovian.md) | **정의**: $q_\sigma(x_{1:T} \mid x_0) := q_\sigma(x_T \mid x_0) \prod_{t=2}^T q_\sigma(x_{t-1} \mid x_t, x_0)$, $q_\sigma(x_{t-1} \mid x_t, x_0) = \mathcal{N}(\sqrt{\bar\alpha_{t-1}} x_0 + \sqrt{1-\bar\alpha_{t-1}-\sigma_t^2} \frac{x_t - \sqrt{\bar\alpha_t} x_0}{\sqrt{1-\bar\alpha_t}}, \sigma_t^2 I)$. **정리**: 모든 $\sigma$ 에 대해 marginal $q_\sigma(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) I)$ 동일 $\square$. **결과**: 같은 $\epsilon_\theta$ 재사용 가능 |
| [03. DDIM Sampling Equation](./ch4-ddim/03-ddim-sampling.md) | **정리**: $x_{t-1} = \sqrt{\bar\alpha_{t-1}}\, \hat x_0 + \sqrt{1 - \bar\alpha_{t-1} - \sigma_t^2}\, \epsilon_\theta(x_t, t) + \sigma_t \epsilon$, $\hat x_0 = (x_t - \sqrt{1-\bar\alpha_t}\, \epsilon_\theta)/\sqrt{\bar\alpha_t}$ $\square$. **$\sigma_t = \sqrt{\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t} \beta_t}$**: DDPM 와 일치. **$\sigma_t = 0$**: deterministic — DDIM 본연. **Sub-sampling**: $\{\tau_1, \ldots, \tau_S\} \subset \{1, \ldots, T\}$ 로 step 수 감소 |
| [04. Probability Flow ODE 와 DPM-Solver](./ch4-ddim/04-probability-flow-ode.md) | **정리**: $\sigma_t = 0$ DDIM 은 probability flow ODE $dx/dt = f - \frac12 g^2 \nabla \log p_t$ 의 1차 discretization $\square$. **DPM-Solver (Lu 2022)**: ODE 의 semi-linear 구조 활용 → exponentially weighted Taylor → 2nd · 3rd order. **DPM-Solver++**: data-prediction model 에 적합한 form, multistep. 10–20 step 으로 50 step DDIM 동급 품질 |

</details>

<br/>

### 🔹 Chapter 5: Guidance 와 Conditioning

> **핵심 질문:** Classifier guidance (Dhariwal 2021) 가 어떻게 외부 classifier 의 gradient 를 score 에 더해 conditional 생성을 만드는가? Classifier-free guidance (Ho & Salimans 2022) 가 어떻게 외부 classifier 없이 같은 효과를 내는가? CFG 의 $w$ 가 왜 quality-diversity trade-off 의 dial 인가? Cross-attention 으로 text embedding 을 UNet 에 주입하는 메커니즘은? Negative prompt 와 compositional generation 의 수학적 형태는?

<details>
<summary><b>Classifier Guidance 부터 Negative Prompt 까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명·재현 |
|------|---------------------|
| [01. Classifier Guidance (Dhariwal & Nichol 2021)](./ch5-guidance/01-classifier-guidance.md) | **정리**: $\nabla_x \log p(x \mid y) = \nabla_x \log p(x) + \nabla_x \log p(y \mid x)$ (Bayes). Diffusion 의 score 에 classifier gradient 를 더함: $\tilde s = s_\theta(x, t) + s \cdot \nabla_x \log p_\phi(y \mid x_t)$ $\square$. **scale $s$**: distribution sharpening. **단점**: noisy $x_t$ 에서 동작하는 classifier 별도 학습 필요 — overhead |
| [02. Classifier-Free Guidance (Ho & Salimans 2022)](./ch5-guidance/02-classifier-free-guidance.md) | **Training**: $\pi_{\text{drop}} = 0.1$ 으로 $y \to \emptyset$ 치환 → conditional·unconditional 한 모델로 학습. **Inference**: $\tilde\epsilon = (1+w) \epsilon_\theta(x, y) - w \epsilon_\theta(x, \emptyset)$ $\square$. **유도**: implicit classifier $\nabla \log p(y \mid x) = \nabla \log p(x \mid y) - \nabla \log p(x)$ → score formulation 으로 동등성. 외부 classifier 불필요 — 현대 SD 의 표준 |
| [03. CFG 의 Quality–Diversity Trade-off](./ch5-guidance/03-cfg-tradeoff.md) | **분석**: $w \uparrow$ → $\tilde\epsilon$ 가 conditional 방향으로 과도하게 이동 → sample 이 $p(x \mid y)$ 의 mode 에 sharp 하게 집중. **Trade-off**: quality (FID, CLIP score) ↑, diversity (recall) ↓. **Empirical**: $w \in [3, 15]$ 가 typical. **Imagen 의 dynamic thresholding**: percentile clip 으로 overshoot 방지 |
| [04. Cross-Attention 으로 Text Conditioning](./ch5-guidance/04-cross-attention.md) | **Stable Diffusion**: text $y$ → CLIP text encoder → token embeddings $c \in \mathbb{R}^{L \times d_c}$. UNet 의 각 attention block 에 cross-attention: $\mathrm{Attn}(Q = W_Q h, K = W_K c, V = W_V c)$ — image feature 가 text token 을 query. **Imagen**: T5-XXL encoder 가 CLIP 보다 strong 한 textual representation. SD3 는 둘 다 ensemble |
| [05. Negative Prompt 와 Compositional Generation](./ch5-guidance/05-negative-prompt.md) | **Negative prompt**: $\tilde\epsilon = \epsilon_\theta(x, \emptyset) + w_+ (\epsilon_\theta(x, y_+) - \epsilon_\theta(x, \emptyset)) - w_- (\epsilon_\theta(x, y_-) - \epsilon_\theta(x, \emptyset))$ — 원하지 않는 concept 의 score 를 빼냄. **Compositional**: Liu 2022 의 composable diffusion — 여러 prompt 의 score 를 가산. **Universal Guidance** (Bansal 2023): forward universal critic 으로 fine-grained control |

</details>

<br/>

### 🔹 Chapter 6: Latent Diffusion 과 현대 아키텍처

> **핵심 질문:** Pixel space 의 직접 diffusion 이 왜 비효율적인가 — Rombach 2022 의 latent compression 이 어떤 trade-off 를 해결하는가? UNet 의 encoder-decoder + skip + attention 구조가 왜 diffusion 의 강력한 inductive bias 인가? DiT (Peebles 2023) 가 UNet 의 구조를 어떻게 ViT 로 대체하고, AdaLN-Zero 가 왜 timestep + conditioning 의 효율적 주입 방식인가? MM-DiT (SD3) 의 image-text joint stream 이 cross-attention 보다 어떤 이점을 갖는가? Cascaded diffusion (Imagen) 의 super-resolution chain 의 의미는?

<details>
<summary><b>Latent Diffusion 부터 Cascaded Diffusion 까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명·재현 |
|------|---------------------|
| [01. Latent Diffusion — Stable Diffusion (Rombach 2022)](./ch6-latent-arch/01-latent-diffusion.md) | **문제**: 512×512×3 ≈ 786K dim → diffusion compute 폭발. **해결**: pretrained VAE $\mathcal{E}, \mathcal{D}$ 로 $z = \mathcal{E}(x) \in \mathbb{R}^{64\times 64\times 4}$ — 약 $48\times$ 압축. **VAE training**: pixel reconstruction + perceptual (LPIPS) + adversarial loss → 디테일 보존. **Diffusion**: $z$-space 에서 $\epsilon_\theta$. Inference: sample $z_T \sim \mathcal{N}$, denoise, decode |
| [02. UNet 아키텍처의 Diffusion 적용](./ch6-latent-arch/02-unet-architecture.md) | **Ronneberger 2015**: encoder-decoder + skip connection (medical segmentation). **DDPM 적용**: ResBlock + (self-)Attention 교차, timestep $t$ embedding 을 GroupNorm 의 affine 으로 주입 (FiLM-style). **Conditioning**: cross-attention 으로 CLIP / T5 embedding. Skip connection 이 fine detail 보존, multi-scale 이 noise 의 다양한 frequency 처리 |
| [03. DiT — Diffusion Transformer (Peebles & Xie 2023)](./ch6-latent-arch/03-dit.md) | **Motivation**: UNet 의 inductive bias 가 scaling 의 bottleneck — Transformer 가 더 매끈하게 scale. **구조**: latent $z$ → patch embedding → $N$ Transformer blocks → unpatch. **AdaLN-Zero**: $t$, $y$ 를 LN 의 $(\gamma, \beta, \alpha)$ 로 변환, residual 을 zero-init scale → identity 시작 → 학습 안정. **결과**: FID 가 GFLOPs 와 단조 감소 — scaling law |
| [04. MM-DiT — Stable Diffusion 3 (Esser 2024)](./ch6-latent-arch/04-mm-dit.md) | **Idea**: image token 과 text token 을 **하나의 sequence** 로 결합 → joint self-attention. **수식**: $[h_{\text{img}}; h_{\text{txt}}] \to \mathrm{Attn}([\![Q_i; Q_t]\!], [\![K_i; K_t]\!], [\![V_i; V_t]\!])$. modality-specific weight (separate $W_Q, W_K, W_V$ per modality) 로 비대칭 학습. **장점**: cross-attention 의 정보 손실 회피, modality 대칭 처리. **Rectified Flow** 와 결합 (Ch7-02) |
| [05. Cascaded Diffusion (Imagen, Saharia 2022)](./ch6-latent-arch/05-cascaded-diffusion.md) | **문제**: high-res 직접 generation 은 compute 폭발. **해결**: 64×64 base + 64→256 SR + 256→1024 SR — 각 단계 별 diffusion model. **장점**: base 가 semantic, SR 이 detail — 책임 분리. **단점**: 3 모델 inference, latent diffusion 보다 비효율. SD vs Imagen 의 architectural trade-off |

</details>

<br/>

### 🔹 Chapter 7: 가속과 최신 방법

> **핵심 질문:** Consistency Model (Song 2023) 의 $f_\theta(x_t, t) \approx x_0$ consistency property 가 어떻게 1-step generation 을 가능하게 하는가? Rectified Flow (Liu 2022) 의 직선 path $x_t = (1-t) x_0 + t z$ 가 왜 DDPM 의 nonlinear schedule 보다 efficient 한가? Flow Matching (Lipman 2023) 이 어떻게 conditional flow matching 으로 CNF 를 실전적으로 학습 가능하게 만드는가? Progressive Distillation (Salimans 2022), SDXL-Turbo / Lightning 의 adversarial distillation 이 real-time generation 의 한계를 어떻게 깨는가?

<details>
<summary><b>Consistency Model 부터 Distillation 까지 (4개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명·재현 |
|------|---------------------|
| [01. Consistency Model (Song 2023)](./ch7-acceleration/01-consistency-model.md) | **정의**: $f_\theta(x_t, t) \approx x_0$ for all $t$ along the same probability flow ODE trajectory — **consistency property**. **Training (CT)**: from scratch, neighboring timesteps 의 prediction 일치 강제. **Distillation (CD)**: pretrained diffusion 의 ODE solver 가 supervision. **결과**: 1-step FID ~9 (CIFAR-10), multi-step 도 가능 — 최초의 SOTA 1-step diffusion |
| [02. Rectified Flow (Liu 2022)](./ch7-acceleration/02-rectified-flow.md) | **정의**: $x_t = (1-t) x_0 + t z$, $z \sim \mathcal{N}(0, I)$ — pair $(x_0, z)$ 사이의 직선 path. **Velocity field**: $v_\theta(x_t, t) \approx z - x_0$. **학습**: $\min_\theta \mathbb{E}\|v_\theta(x_t, t) - (z - x_0)\|^2$. **Reflow**: 학습된 ODE 로 $(x_0, z)$ 를 paired → 재학습 → trajectory 가 더 직선화 → 점근적으로 1-step. **SD3 의 base** (Esser 2024) |
| [03. Flow Matching (Lipman 2023)](./ch7-acceleration/03-flow-matching.md) | **CNF 의 일반화**: probability path $p_t$ 와 vector field $u_t$ 의 직접 학습. **Conditional Flow Matching**: $\mathcal{L} = \mathbb{E}_{t, x_1, x}\|v_\theta(x, t) - u_t(x \mid x_1)\|^2$ — conditional path 로 unbiased estimator. **Diffusion 과의 관계**: VP-SDE 가 OT-CFM 의 특수 경우. Rectified Flow ⊂ Flow Matching 의 한 instantiation |
| [04. Distillation 과 Sampling Speed](./ch7-acceleration/04-distillation.md) | **Progressive Distillation (Salimans 2022)**: teacher (1024 step) → student (512 step) → ... → 4 step. 각 student 가 teacher 의 2-step 을 1-step 으로 emulate. **SDXL-Turbo / Lightning**: ADD (adversarial diffusion distillation) — student 의 sample 을 GAN-style discriminator 가 evaluate → real-time (1–2 step). **Trade-off**: speed vs quality — 사용자 시나리오에 따라 선택 |

</details>

---

> 🆕 **2026-04 최신 업데이트**: Ch1-03 의 Forward closed-form 증명에 Gaussian reparameterization 의 induction step 을 명시적으로 분리, Ch2-04 의 $L_{\mathrm{simple}}$ 유도에 weight 제거가 왜 perceptual quality 에 유리한지 SNR 분석 추가, Ch3-04 의 Score-SDE 통합에 Anderson 1982 reverse-time SDE 정리의 유도 단계 보강, Ch4-03 의 DDIM sampling equation 에 $\sigma_t$ 의 모든 케이스 (DDPM, DDIM, sub-VP) 비교 표 추가, Ch5-02 의 CFG 유도에 implicit classifier 관점과 score-formulation 관점의 등가성 증명을 추가, Ch6-04 의 MM-DiT 에 SD3 의 별도 weight 설계 동기 분석을 추가, Ch7-02 의 Rectified Flow 에 reflow 의 수렴 분석을 보강했습니다. **11-섹션 문서 골격이 전체 33개 문서에서 일관** 됩니다.

## 🏆 핵심 정리 인덱스

이 레포에서 **완전한 증명** 또는 **원 논문 실험 재현** 을 제공하는 대표 결과 모음입니다. 각 챕터 문서에서 $\square$ 로 종결되는 엄밀한 증명 또는 `results/` 하의 학습 곡선·plot 을 확인할 수 있습니다.

| 정리·결과 | 서술 | 출처 문서 |
|----------|------|----------|
| **Forward Closed-Form** | $q(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) I)$ — Gaussian reparam 의 누적 | [Ch1-03](./ch1-foundations/03-forward-closed-form.md) |
| **Posterior $q(x_{t-1} \mid x_t, x_0)$** | $\mathcal{N}(\tilde\mu_t, \tilde\beta_t I)$ — Bayes' rule + Gaussian 곱 | [Ch1-05](./ch1-foundations/05-posterior-derivation.md) |
| **VLB 분해** | $L_{\mathrm{vlb}} = L_T + \sum L_{t-1} + L_0$ — Markov 분해 | [Ch2-02](./ch2-elbo/02-elbo-three-terms.md) |
| **Noise Reparam $\Rightarrow L_{\mathrm{simple}}$** | $L_{t-1}$ 가 $\|\epsilon - \epsilon_\theta\|^2$ 의 weighted form, weight 제거 후 simple MSE | [Ch2-04](./ch2-elbo/04-l-simple.md) |
| **DSM ↔ DDPM 등가성** | Vincent (2011) DSM 이 noise prediction 과 mathematically 동등 | [Ch3-02](./ch3-score-sde/02-denoising-score-matching.md) |
| **Anderson Reverse-Time SDE** | $dx = [f - g^2 \nabla \log p_t] dt + g d\bar W$ | [Ch3-04](./ch3-score-sde/04-score-sde.md) |
| **Probability Flow ODE** | $dx/dt = f - \frac12 g^2 \nabla \log p_t$ — same marginal, deterministic | [Ch3-04](./ch3-score-sde/04-score-sde.md) |
| **VP-SDE = DDPM 연속 한계** | $f = -\frac12 \beta x$, $g = \sqrt{\beta}$ | [Ch3-05](./ch3-score-sde/05-vp-ve-sde.md) |
| **DDIM Marginal 보존** | Non-Markovian forward 가 같은 $q(x_t \mid x_0)$ 유지 | [Ch4-02](./ch4-ddim/02-non-markovian.md) |
| **DDIM ODE 극한** | $\sigma_t = 0$ DDIM = probability flow ODE 의 1차 discretization | [Ch4-04](./ch4-ddim/04-probability-flow-ode.md) |
| **CFG Identity** | $\tilde\epsilon = (1+w)\epsilon_\theta(x, y) - w\epsilon_\theta(x, \emptyset)$ — implicit classifier | [Ch5-02](./ch5-guidance/02-classifier-free-guidance.md) |
| **Latent Compression Ratio** | VAE 로 $48\times$ 압축, perceptual loss 가 디테일 보존 | [Ch6-01](./ch6-latent-arch/01-latent-diffusion.md) |
| **AdaLN-Zero** | DiT 의 timestep + condition 주입, residual zero-init 으로 학습 안정 | [Ch6-03](./ch6-latent-arch/03-dit.md) |
| **Consistency Property** | $f_\theta(x_t, t) \approx x_0$ for all $t$ along same trajectory → 1-step | [Ch7-01](./ch7-acceleration/01-consistency-model.md) |
| **Rectified Flow 직선 Path** | $x_t = (1-t) x_0 + t z$ — reflow 로 점근적 1-step | [Ch7-02](./ch7-acceleration/02-rectified-flow.md) |

> 💡 **챕터별 문서·정리/정의 수** (실측):
>
> | 챕터 | 문서 수 | 정리·정의 |
> |------|---------|------------|
> | Ch1 Foundations | 5 | 38 |
> | Ch2 ELBO · L_simple | 5 | 41 |
> | Ch3 Score-SDE | 5 | 44 |
> | Ch4 DDIM · ODE | 4 | 33 |
> | Ch5 Guidance | 5 | 40 |
> | Ch6 Latent · DiT | 5 | 42 |
> | Ch7 Acceleration | 4 | 34 |
> | **합계** | **33** | **272** |
>
> 추가로 **130+ 엄밀한 $\square$ 증명 + 99 연습문제 (모두 해설 포함) + 130+ NumPy/PyTorch/diffusers 실험 코드 (`### 실험 N` 형식)**.
>
> Ch4 (DDIM) 와 Ch7 (가속) 은 **4 문서** 로 구성 — DDIM 의 핵심 4 단계 (motivation → non-Markovian → sampling eq → ODE) 에 집중, 가속은 mature 한 4 가지 (Consistency, Rectified Flow, Flow Matching, Distillation) 만 다룸 (Chapters 1–3, 5, 6 의 5 문서와 의도적 차이).

---

## 💻 실험 환경

모든 챕터의 실험은 아래 환경에서 재현 가능합니다.

```bash
# requirements.txt
numpy==1.26.0
scipy==1.11.0
torch==2.1.0
torchvision==0.16.0
diffusers==0.25.0
transformers==4.36.0
accelerate==0.25.0
einops==0.7.0
matplotlib==3.8.0
seaborn==0.13.0
tqdm==4.66.0
jupyter==1.0.0
# 선택 사항
tensorboard==2.15.0            # 학습 곡선 로깅
wandb==0.16.0                  # 실험 추적
xformers==0.0.23               # 메모리 효율 attention (large model)
```

```bash
# 환경 설치 (CPU 기준; GPU 권장)
pip install numpy==1.26.0 scipy==1.11.0 torch==2.1.0 torchvision==0.16.0 \
            diffusers==0.25.0 transformers==4.36.0 accelerate==0.25.0 \
            einops==0.7.0 matplotlib==3.8.0 seaborn==0.13.0 \
            tqdm==4.66.0 jupyter==1.0.0

# 실험 노트북 실행
jupyter notebook
```

```python
# 대표 실험 ① — DDPM 바닥부터 + Forward closed-form 검증 (Ch1-03, Ch2-04)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

T = 1000
betas = torch.linspace(1e-4, 0.02, T)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_ac     = torch.sqrt(alphas_cumprod)
sqrt_1m_ac  = torch.sqrt(1.0 - alphas_cumprod)

def q_sample(x_0, t, noise=None):
    """Forward closed-form: x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε."""
    if noise is None:
        noise = torch.randn_like(x_0)
    return sqrt_ac[t] * x_0 + sqrt_1m_ac[t] * noise

class Denoiser1D(nn.Module):
    def __init__(self, h=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, 1),
        )
    def forward(self, x, t):
        return self.net(torch.cat([x, t.float().unsqueeze(-1) / T], -1))

# L_simple training (Ch2-04)
model, opt = Denoiser1D(), None
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for step in range(5000):
    x_0 = torch.randn(256, 1)
    x_0 = torch.where(torch.rand(256, 1) > 0.5, x_0 * 0.3 + 2, x_0 * 0.3 - 2)  # 2-modal
    t   = torch.randint(0, T, (256,))
    eps = torch.randn_like(x_0)
    x_t = q_sample(x_0, t, eps)
    loss = ((eps - model(x_t, t))**2).mean()
    opt.zero_grad(); loss.backward(); opt.step()

# DDPM sampling (Ch1-04, Ch2-03)
@torch.no_grad()
def ddpm_sample(model, n=2000):
    x = torch.randn(n, 1)
    for t in reversed(range(T)):
        z      = torch.randn_like(x) if t > 0 else 0.0
        coef1  = 1.0 / torch.sqrt(alphas[t])
        coef2  = betas[t] / sqrt_1m_ac[t]
        eps    = model(x, torch.full((n,), t))
        x      = coef1 * (x - coef2 * eps) + torch.sqrt(betas[t]) * z
    return x

# DDIM sampling (Ch4-03, Ch4-04) — same model, 50 steps deterministic
@torch.no_grad()
def ddim_sample(model, n=2000, steps=50):
    timesteps = torch.linspace(T - 1, 0, steps).long()
    x = torch.randn(n, 1)
    for i in range(len(timesteps) - 1):
        t, t_next = timesteps[i], timesteps[i + 1]
        ac, ac_next = alphas_cumprod[t], alphas_cumprod[t_next]
        eps    = model(x, torch.full((n,), t))
        x_0_h  = (x - torch.sqrt(1 - ac) * eps) / torch.sqrt(ac)
        x      = torch.sqrt(ac_next) * x_0_h + torch.sqrt(1 - ac_next) * eps
    return x

# Classifier-Free Guidance (Ch5-02) — skeleton
def cfg_eps(model, x_t, t, y, null_y, w=7.5):
    eps_c = model(x_t, t, y)
    eps_u = model(x_t, t, null_y)
    return (1 + w) * eps_c - w * eps_u

# 실험 ② — Stable Diffusion inference 재현 (Ch6-01)
# from diffusers import StableDiffusionPipeline
# pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
# image = pipe("a photo of an astronaut riding a horse", guidance_scale=7.5).images[0]
# w = 1, 3, 7.5, 15 별 sample quality / diversity 비교 (Ch5-03)

# 실험 ③ — DiT vs UNet scaling (Ch6-03)
# 작은 규모 (CIFAR-10) 에서 같은 latent dim · compute 으로 두 architecture FID 비교

# 실험 ④ — Consistency Model 1-step (Ch7-01)
# Pretrained DDPM teacher → Consistency Distillation → 1-step sample
```

---

## 📖 각 문서 구성 방식

모든 문서는 다음 **11-섹션 골격** 으로 작성됩니다.

| # | 섹션 | 내용 |
|:-:|------|------|
| 1 | 🎯 **핵심 질문** | 이 문서가 답하는 3~5개의 본질적 질문 |
| 2 | 🔍 **왜 이 ... 인가** | 해당 이론·알고리즘이 diffusion 의 어떤 핵심 문제를 푸는지 |
| 3 | 📐 **수학적 선행 조건** | Generative · SDE · Prob · Info · ViT 레포의 어떤 정리를 전제하는지 |
| 4 | 📖 **직관적 이해** | Forward / reverse · score · ODE / SDE · latent 등의 기하학적 직관 |
| 5 | ✏️ **엄밀한 정의** | Forward Markov · ELBO · DSM · Score-SDE · DDIM · CFG 등 |
| 6 | 🔬 **정리와 증명** | Forward closed-form · ELBO 분해 · DSM 등가성 · DDIM marginal 보존 등 |
| 7 | 💻 **NumPy / PyTorch 구현 검증** | 4 가지 실험 (`### 실험 1` ~ `### 실험 4`) — 1D toy · MNIST · ablation · 시각화 |
| 8 | 🔗 **실전 활용** | 언제 DDPM · 언제 DDIM · 언제 latent · 언제 DiT — 실전 선택 가이드 |
| 9 | ⚖️ **가정과 한계** | 각 방법의 실패 모드 (sampling cost · mode collapse · training cost · OOD prompts) |
| 10 | 📌 **핵심 정리** | 한 장으로 요약 ($\boxed{}$ 핵심 수식 + 표) |
| 11 | 🤔 **생각해볼 문제 (+ 해설)** | 기초 / 심화 / 논문 비평 의 3 문제, `<details>` 펼침 해설 |

> 📚 **연습문제 총 99개** (33 문서 × 3 문제): **기초 / 심화 / 논문 비평** 의 3-tier 구성, 모든 문제에 `<details>` 펼침 해설 포함. Forward closed-form 의 induction 부터 ELBO 분해의 case 분석, DSM 의 등가성 증명, DDIM 의 non-Markovian 설계, CFG 의 implicit classifier 유도, MM-DiT 의 ablation 까지 단계적으로 심화됩니다.
>
> 🧭 **푸터 네비게이션**: 각 문서 하단에 `◀ 이전 / 📚 README / 다음 ▶` 링크가 항상 제공됩니다. 챕터 경계에서도 다음 챕터 첫 문서로 자동 연결됩니다.
>
> ⏱️ **학습 시간 추정**: 문서당 평균 약 200~450줄 (정의·증명·코드·연습문제 포함) 기준 **약 45분~1시간 20분**. 전체 33문서는 약 **30~40시간** 상당 (증명 재구성 · diffusers 실험 재현 포함 시 55시간+).

---

## 🗺️ 추천 학습 경로

<details>
<summary><b>🟢 "Stable Diffusion 을 쓰지만 왜 작동하는지 이론적으로 이해하고 싶다" — 입문 투어 (1주, 약 12~14시간)</b></summary>

<br/>

```
Day 1  Ch1-01  Diffusion 의 물리적 기원
       Ch1-03  Forward Closed-Form
Day 2  Ch1-04  Reverse Process
       Ch1-05  Posterior 유도
Day 3  Ch2-02  ELBO 의 3개 항
       Ch2-04  L_simple 유도
Day 4  Ch4-01  DDIM 동기
       Ch4-03  DDIM Sampling
Day 5  Ch5-02  Classifier-Free Guidance
       Ch5-03  CFG Trade-off
Day 6  Ch6-01  Latent Diffusion (Stable Diffusion)
       Ch6-02  UNet 아키텍처
Day 7  Ch6-03  DiT
       Ch7-04  Distillation
```

</details>

<details>
<summary><b>🟡 "DDPM ↔ Score-SDE ↔ DDIM 의 통합 관점을 완전히 정복한다" — 이론 집중 (2주, 약 24~28시간)</b></summary>

<br/>

```
1주차 — Foundations · ELBO · Score-SDE
  Day 1    Ch1-01~03   물리적 기원 · Forward Markov · Closed-form
  Day 2    Ch1-04~05   Reverse · Posterior
  Day 3    Ch2-01~02   VLB · ELBO 3항
  Day 4    Ch2-03~04   Noise Param · L_simple
  Day 5    Ch2-05      Improved DDPM
  Day 6    Ch3-01~02   Score · DSM
  Day 7    Ch3-03~05   NCSN · Score-SDE · VP/VE

2주차 — DDIM · Guidance · Latent · 가속
  Day 1    Ch4-01~02   DDIM 동기 · Non-Markovian
  Day 2    Ch4-03~04   Sampling Eq · Probability Flow ODE
  Day 3    Ch5-01~02   Classifier · CFG
  Day 4    Ch5-03~05   Trade-off · Cross-Attention · Negative
  Day 5    Ch6-01~02   LDM · UNet
  Day 6    Ch6-03~05   DiT · MM-DiT · Cascaded
  Day 7    Ch7-01~04   Consistency · RF · Flow Matching · Distillation
```

</details>

<details>
<summary><b>🔴 "Diffusion 의 수학을 완전 정복한다" — 전체 정복 (8주, 약 35~45시간 + 재현 12~18시간)</b></summary>

<br/>

```
1주차   Chapter 1 — Foundations
         → Forward closed-form 의 induction 손 증명
         → Posterior 의 Gaussian 곱 closed-form 유도
         → Sohl-Dickstein 2015 의 nonequilibrium thermodynamics 직관

2주차   Chapter 2 — ELBO · L_simple
         → VLB 분해의 KL three-term 증명
         → Two Gaussian KL → noise prediction MSE 유도
         → MNIST 에서 DDPM 바닥부터, $L_{\mathrm{simple}}$ vs $L_{\mathrm{vlb}}$ 학습 곡선 비교

3주차   Chapter 3 — Score-SDE
         → DSM ↔ DDPM 등가성 손 증명
         → Anderson 1982 reverse-time SDE 정리 따라가기
         → VP / VE / sub-VP 의 MNIST trajectory 시각화

4주차   Chapter 4 — DDIM · ODE
         → DDIM 의 marginal 보존 증명
         → $\sigma_t$ 모든 케이스에서 sampling equation 유도
         → DDPM 1000 vs DDIM 50 vs DDIM 10 step 의 품질·시간 비교

5주차   Chapter 5 — Guidance
         → Classifier guidance 와 CFG 의 score-formulation 등가성
         → CFG scale $w \in \{0, 1, 3, 7.5, 15\}$ 의 sample quality / diversity sweep
         → Stable Diffusion 의 cross-attention 시각화

6주차   Chapter 6 — Latent · DiT
         → VAE latent space 의 diffusion trajectory 시각화
         → 작은 규모 CIFAR-10 에서 UNet vs DiT FID 비교
         → MM-DiT 의 image-text joint attention map

7주차   Chapter 7 — Acceleration
         → Consistency Model distillation 직접 구현
         → Rectified Flow 의 reflow effect 측정
         → Flow Matching 과 DDPM 의 학습 곡선·sample 비교

8주차   종합 — SD3 / Sora / DALL-E 3
         → SD3 의 MM-DiT + Rectified Flow 분석
         → Sora 의 spatio-temporal DiT 추정
         → "DDPM vs Flow Matching" / "UNet vs DiT" / "Pixel vs Latent" 토론
```

</details>

---

## 🔗 연관 레포지토리

| 레포 | 주요 내용 | 연관 챕터 |
|------|----------|-----------|
| [generative-model-deep-dive](https://github.com/iq-ai-lab/generative-model-deep-dive) | DDPM 기초 · VAE · GAN · NF 비교 | **Ch1, Ch2** (선행) |
| [sde-deep-dive](https://github.com/iq-ai-lab/sde-deep-dive) | Brownian motion · Fokker-Planck · Itô / Stratonovich | **Ch3 전체** (Score-SDE) |
| [probability-theory-deep-dive](https://github.com/iq-ai-lab/probability-theory-deep-dive) | Gaussian · KL · Bayes · Conditional expectation | **Ch1, Ch2** (Posterior, ELBO) |
| [information-theory-deep-dive](https://github.com/iq-ai-lab/information-theory-deep-dive) | Entropy · KL divergence · ELBO | **Ch2** (VLB 분해) |
| [vision-transformer-deep-dive](https://github.com/iq-ai-lab/vision-transformer-deep-dive) | ViT · patch embedding · scaling law · AdaLN | **Ch6-03~04** (DiT, MM-DiT) |
| [advanced-rl-deep-dive](https://github.com/iq-ai-lab/advanced-rl-deep-dive) | TRPO · PPO · SAC · TD3 · RLHF · DPO | RLHF 와 diffusion alignment 의 비교 |
| [transformer-deep-dive](https://github.com/iq-ai-lab/transformer-deep-dive) | Attention · Pre-LN · Cross-attention | **Ch5-04** (Cross-attention) |
| [multimodal-foundation-models-deep-dive](https://github.com/iq-ai-lab/multimodal-foundation-models-deep-dive) *(다음)* | Sora · DALL-E 3 · Veo · GR-2 | **Ch6-7 이후** 응용 분석 |

> 💡 이 레포는 **"DDPM · Score-SDE · DDIM 이 모두 같은 stochastic process 의 다른 구현이고, $\bar\alpha$ · $\sigma_t$ · $w$ · latent · AdaLN 이 왜 각각의 이론적 동기를 갖는가"** 에 집중합니다. Generative Model 에서 ELBO 와 DDPM 기초를, SDE 에서 Brownian motion 과 Fokker-Planck 를, Probability 에서 KL 과 Gaussian posterior 를, Information Theory 에서 ELBO 를 익힌 후 오면 Chapter 2 (ELBO 분해) 와 Chapter 3 (Score-SDE 통합) 의 증명이 훨씬 자연스럽습니다. **Vision Transformer Deep Dive** 와 함께 보면 Ch6 의 DiT / MM-DiT 가 SD3 · Sora 의 backbone 이 된 맥락이 선명해집니다.

---

## 📖 Reference

### 🏛️ DDPM · Score Matching · Score-SDE
- **Deep Unsupervised Learning using Nonequilibrium Thermodynamics** (Sohl-Dickstein et al., 2015) — **Diffusion 효시**
- **Denoising Diffusion Probabilistic Models** (Ho, Jain, Abbeel, 2020) — **DDPM**
- **A Connection Between Score Matching and Denoising Autoencoders** (Vincent, 2011) — **DSM**
- **Generative Modeling by Estimating Gradients of the Data Distribution** (Song & Ermon, 2019) — **NCSN**
- **Score-Based Generative Modeling through SDEs** (Song et al., 2021) — **Score-SDE**
- **Improved Denoising Diffusion Probabilistic Models** (Nichol & Dhariwal, 2021) — **Improved DDPM**

### ⚡ DDIM · ODE Solvers
- **Denoising Diffusion Implicit Models** (Song, Meng, Ermon, 2021) — **DDIM**
- **DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling** (Lu et al., 2022)
- **DPM-Solver++: Fast Solver for Guided Sampling of DPMs** (Lu et al., 2022)
- **Pseudo Numerical Methods for Diffusion Models on Manifolds** (Liu et al., 2022) — PNDM
- **Elucidating the Design Space of Diffusion-Based Generative Models** (Karras et al., 2022) — EDM

### 🧭 Guidance · Conditioning
- **Diffusion Models Beat GANs on Image Synthesis** (Dhariwal & Nichol, 2021) — **Classifier Guidance**
- **Classifier-Free Diffusion Guidance** (Ho & Salimans, 2022) — **CFG**
- **Compositional Visual Generation with Composable Diffusion Models** (Liu et al., 2022)
- **Universal Guidance for Diffusion Models** (Bansal et al., 2023)
- **GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models** (Nichol et al., 2022)

### 🖼️ Latent Diffusion · Architecture
- **High-Resolution Image Synthesis with Latent Diffusion Models** (Rombach et al., 2022) — **Stable Diffusion / LDM**
- **U-Net: Convolutional Networks for Biomedical Image Segmentation** (Ronneberger et al., 2015) — **UNet**
- **Scalable Diffusion Models with Transformers** (Peebles & Xie, 2023) — **DiT**
- **Scaling Rectified Flow Transformers for High-Resolution Image Synthesis** (Esser et al., 2024) — **SD3 / MM-DiT**
- **Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding** (Saharia et al., 2022) — **Imagen**
- **SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis** (Podell et al., 2024) — **SDXL**

### 🚀 Acceleration · Consistency · Flow
- **Consistency Models** (Song et al., 2023) — **Consistency**
- **Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow** (Liu et al., 2022) — **Rectified Flow**
- **Flow Matching for Generative Modeling** (Lipman et al., 2023) — **Flow Matching**
- **Progressive Distillation for Fast Sampling of Diffusion Models** (Salimans & Ho, 2022)
- **Adversarial Diffusion Distillation** (Sauer et al., 2024) — **SDXL-Turbo**
- **SDXL-Lightning: Progressive Adversarial Diffusion Distillation** (Lin et al., 2024)
- **SnapFusion: Text-to-Image Diffusion Model on Mobile** (Li et al., 2023)

### 🌐 Applications — Editing · Video · 3D
- **InstructPix2Pix: Learning to Follow Image Editing Instructions** (Brooks et al., 2023)
- **SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations** (Meng et al., 2022)
- **Adding Conditional Control to Text-to-Image Diffusion Models** (Zhang et al., 2023) — **ControlNet**
- **DreamFusion: Text-to-3D using 2D Diffusion** (Poole et al., 2023)
- **Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets** (Blattmann et al., 2023)
- **Sora: Video Generation Models as World Simulators** (OpenAI, 2024)

### 🛠️ Implementation · Libraries
- **🤗 Diffusers** (von Platen et al., 2022) — 표준 PyTorch diffusion 라이브러리
- **k-diffusion** (Katherine Crowson, 2022) — Karras-style sampler
- **CompVis / Stable Diffusion** (Rombach et al., 2022) — 원조 SD 구현
- **OpenAI Improved DDPM** (Nichol & Dhariwal, 2021) — 표준 DDPM 코드
- **lucidrains / denoising-diffusion-pytorch** — 학습 친화적 구현

---

<div align="center">

**⭐️ 도움이 되셨다면 Star 를 눌러주세요!**

Made with ❤️ by [IQ AI Lab](https://github.com/iq-ai-lab)

<br/>

*"Stable Diffusion 을 호출하는 것과 — Ho 2020 으로 forward closed-form $q(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) I)$ 와 ELBO 분해 $L_{\mathrm{vlb}} = L_T + \sum L_{t-1} + L_0$ 에서 noise prediction $L_{\mathrm{simple}}$ 까지 한 줄씩 증명 · Vincent 2011 과 Song 2021 으로 DSM·NCSN·DDPM·DDIM 이 모두 Score-SDE 의 특수 경우임을 유도 · Song 2021 (DDIM) 으로 non-Markovian forward 가 같은 marginal 을 유지하면서 $\sigma_t \to 0$ 극한이 probability flow ODE 임을 증명 · Ho & Salimans 2022 로 CFG 의 $\tilde\epsilon = (1+w)\epsilon_\theta(x, y) - w\epsilon_\theta(x, \emptyset)$ 가 implicit classifier 의 직접 응용임을 유도 · Rombach 2022 / Peebles 2023 / Esser 2024 로 latent compression · DiT · MM-DiT 가 Stable Diffusion 부터 SD3·Sora 까지의 architectural 진화임을 분석 · Song 2023 / Liu 2022 로 Consistency Model · Rectified Flow 가 1-step generation 의 frontier 를 어떻게 여는지 유도 — 이 모든 '왜' 를 직접 유도할 수 있는 것은 다르다"*

</div>
