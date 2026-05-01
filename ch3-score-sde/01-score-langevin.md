# 1. Score 함수와 Langevin MCMC

## 🎯 핵심 질문
Score 함수 $s(x) = \nabla_x \log p(x)$는 무엇이고, 이것이 Langevin MCMC에서 어떻게 샘플링을 수행하는가? 왜 고차원에서 mode mixing이 어려운가?

## 🔍 왜 Score인가?

데이터 분포 $p(x)$를 직접 모델링하기는 어렵지만, 그 **로그 확률의 그래디언트(score)**는 $x$가 높은 확률 영역으로 움직여야 할 방향을 가리킨다. 이는 분자 동역학(molecular dynamics)의 포텐셜 그래디언트처럼 작동한다.

## 📐 수학적 선행 조건
- 그래디언트: $\nabla_x f(x)$ 정의와 chain rule
- Brownian motion: $W_t$ 기본 성질, $dW$의 분산 = $dt$
- 확률 밀도: $\int p(x)dx = 1$, $\int p(x) \nabla_x \log p(x) dx = 0$
- Fokker-Planck 방정식 기초

## 📖 직관적 이해

한 입자가 에너지 지형(potential landscape)에서 움직인다고 생각하자. 에너지 $U(x) = -\log p(x)$이면, 입자는 다음 두 가지에 의해 움직인다:
1. **그래디언트 스텝**: $-\nabla_x U(x) = \nabla_x \log p(x) = s(x)$ (내리막)
2. **노이즈**: 열(temperature)에서 나오는 랜덤 섭동

이 둘이 결합되면 정상 분포(stationary distribution)가 $p(x)$가 되도록 설계할 수 있다.

## ✏️ 엄밀한 정의

**정의 3.1 (Score 함수)**  
데이터 분포 $p(x)$에 대해, score 함수는 
$$s(x) := \nabla_x \log p(x)$$
로 정의한다. 여기서 $p(x) > 0$을 가정한다.

**정의 3.2 (Langevin 동역학)**  
learning rate $\eta > 0$, 반복 $k = 0, 1, 2, \ldots$에 대해
$$x_{k+1} = x_k + \eta s(x_k) + \sqrt{2\eta} \epsilon_k, \quad \epsilon_k \sim \mathcal{N}(0, I)$$
를 Langevin 알고리즘이라 한다.

## 🔬 정리와 증명

**정리 3.1 (Langevin의 정상 분포)**  
위 알고리즘에서 $\eta \to 0$일 때, 반복의 극한 분포는 $p(x)$로 수렴한다 (Roberts & Tweedie 1996).

증명 스케치: Fokker-Planck 방정식 또는 연속시간 극한에서
$$dx = s(x)dt + \sqrt{2}dW$$
의 정상 분포를 구하면, density $\pi(x)$는
$$0 = \nabla \cdot (\pi s) + \nabla^2 \pi = \nabla \cdot (\pi s + \nabla \pi)$$
을 만족한다. $\nabla \log \pi = s$이면 우변이 0이 되므로, $\pi = p$이다. $\square$

**정리 3.2 (수렴 속도)**  
고차원에서 spectral gap이 지수적으로 작아지면, mode mixing time이 지수적으로 증가한다.

## 💻 NumPy / PyTorch 구현 검증

### 실험 1: 1D Gaussian에서의 Langevin 샘플링
```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
def score_1d(x, mu=0, sigma=1):
    """Score of N(mu, sigma^2)"""
    return -(x - mu) / sigma**2

# Langevin 반복
x_samples = [0.0]
eta = 0.01
for _ in range(10000):
    x = x_samples[-1]
    s = score_1d(x, mu=2.0, sigma=1.0)
    x_new = x + eta * s + np.sqrt(2 * eta) * np.random.randn()
    x_samples.append(x_new)

x_samples = np.array(x_samples[1000:])  # Burn-in
print(f"샘플 평균: {x_samples.mean():.3f} (참값 2.0)")
print(f"샘플 표준편차: {x_samples.std():.3f} (참값 1.0)")
```

### 실험 2: 2D Gaussian Mixture (Mode Mixing 관찰)
```python
# 2D mixture: 0.5*N([5,5], I) + 0.5*N([-5,-5], I)
def score_2d_mixture(x, mu1=np.array([5., 5.]), mu2=np.array([-5., -5.])):
    p1 = np.exp(-0.5 * np.sum((x - mu1)**2))
    p2 = np.exp(-0.5 * np.sum((x - mu2)**2))
    s1 = -(x - mu1)
    s2 = -(x - mu2)
    return (p1 * s1 + p2 * s2) / (p1 + p2)

x = np.array([5.0, 5.0])
eta = 0.01
samples = [x.copy()]

for _ in range(50000):
    s = score_2d_mixture(x)
    x = x + eta * s + np.sqrt(2 * eta) * np.random.randn(2)
    samples.append(x.copy())

samples = np.array(samples[5000:])
# Mode 사이를 얼마나 자주 방문했는가? 느린 mode mixing 관찰
```

### 실험 3: 고차원 시뮬레이션 (Dimensionality 효과)
```python
def langevin_in_d_dims(d=10, steps=10000, eta=0.01):
    x = np.zeros(d)
    samples = []
    for _ in range(steps):
        s = -x  # N(0, I)의 score
        x = x + eta * s + np.sqrt(2 * eta) * np.random.randn(d)
        samples.append(x.copy())
    return np.array(samples[1000:])

for d in [2, 5, 10, 50]:
    samples = langevin_in_d_dims(d)
    print(f"d={d}: norm = {np.linalg.norm(samples.mean(axis=0)):.3f}")
```

## 🔗 실전 활용

**Score 기반 생성 모델의 기초**
- Diffusion model은 고정된 $t$에서 $s_\theta(x, t) \approx \nabla_x \log p_t(x)$를 학습
- 학습한 score로 역방향 SDE/ODE를 풀어 샘플링

**물리적 해석**
- Langevin 동역학은 overdamped molecular dynamics의 이산화
- $\eta$: time step (물리에서 $dt$)
- 노이즈: 온도에 의한 thermal fluctuation

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| $p(x) > 0$ everywhere | 0 확률 영역에서 정의 불명 |
| Score를 정확히 알 수 있음 | 실제로는 $s_\theta$로 근사 필요 |
| $\eta$ 충분히 작음 | 큰 $\eta$에서 bias 발생, 작으면 느림 |
| Ergodicity | 고차원 다중 mode에서 mixing time 지수적 |

## 📌 핵심 정리

**Score 함수 $s(x) = \nabla_x \log p(x)$는 고확률 영역으로 향하는 방향을 가리킨다. Langevin 알고리즘 $x_{k+1} = x_k + \eta s(x_k) + \sqrt{2\eta}\epsilon_k$의 정상 분포는 $p(x)$이다. 그러나 고차원에서 mode mixing이 느려지는 것이 핵심 한계이며, 이를 해결하기 위해 multi-scale noise 전략이 등장한다.**

## 🤔 생각해볼 문제

### 문제 1
Langevin 알고리즘이 정상 분포에 수렴하려면 왜 노이즈 항 $\sqrt{2\eta}\epsilon$이 정확히 이 형태여야 하는가? 더 큰 또는 작은 노이즈는?

<details>
<summary>해설</summary>

Fokker-Planck 방정식 $\frac{\partial \rho}{\partial t} = -\nabla \cdot (\rho s) + \nabla^2 \rho$에서, 정상 상태 $\nabla \cdot (\rho s + \nabla \rho) = 0$은 $\rho = p$일 때 만족된다. 더 큰 노이즈는 과도한 mixing으로 mode를 놓치기 쉽고, 더 작은 노이즈는 특정 mode에 갇힌다.
</details>

### 문제 2
고차원에서 두 개의 잘 분리된 mode를 가진 분포를 생각하자. Mode 사이의 "에너지 장벽"이 높을수록 switching 빈도가 어떻게 변하는가?

<details>
<summary>해설</summary>

에너지 장벽이 높을수록 switching 빈도는 지수적으로 감소한다 ($\propto e^{-\Delta E/\eta}$). 이것이 "mode mixing 문제"의 핵심이며, noise를 시간에 따라 변화시키는 것(annealing)이 해결책이다.
</details>

### 문제 3
만약 $p(x)$를 모르고 estimated score $\hat{s}_\theta(x)$를 사용하면 어떤 오차가 생기는가?

<details>
<summary>해설</summary>

Score 오차는 분포의 편향(bias)을 야기한다. $\mathbb{E}[x_{k+1}|x_k] = x_k + \eta(\hat{s}_\theta(x_k) - s(x_k)) + O(\eta^2)$이므로, 오차가 누적되면 정상 분포가 $p$에서 벗어난다. 이를 최소화하기 위해 score matching이 등장한다.
</details>

---

<div align="center">

[◀ 이전](../ch2-elbo/05-improved-ddpm.md) | [📚 README](../README.md) | [다음 ▶](./02-denoising-score-matching.md)

</div>
