# M4R

**A Variational Approach to American Option Pricing via Physics-Informed Neural Networks**

Python library developed as part of my final dissertation project at Imperial College London. The library includes neural network and classical methods for pricing American options under Black-Scholes and Heston dynamics.

Further work may be done on this repository beyond the submission deadline of Monday 8 June 2026. Please refer to the commit **FIRST SUBMISSION** with commit id `2cf8354` and commit hash `2cf8354a602b2a4f49f96185d84396ee3459611d` for the version of the repository at submission.

## Packages

- **`utility`** — shared building blocks: the `PINN`/`ModelConfig` training framework (`utility.model`), the `Sampler` (`utility.sampler`), path simulators (`utility.simulate`), continuation-value estimation (`utility.continuation_prob`) and plotting helpers (`utility.plot`).
- **`black_scholes`** — Black-Scholes pricing: closed-form (`closed`), binomial tree (`tree`), PINN solvers (`pinn`) and Sobolev-trained networks (`sobolev`).
- **`heston`** — Heston stochastic-volatility pricing: closed-form (`closed`), tree (`tree`) and PINN solvers (`pinn`).
- **`config`** — experiment configurations (`ModelConfig` instances) and plot settings for each model/dimension combination.

## Installation

From the project root:

```bash
pip install -e .
```
