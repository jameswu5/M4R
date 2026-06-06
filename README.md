# M4R

Python library developed as part of my final dissertation project at Imperial College London. The library includes neural network and classical methods for pricing American options under Black-Scholes and Heston dynamics.

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
