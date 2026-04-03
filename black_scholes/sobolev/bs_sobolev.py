"""Sobolev-regularised network for pricing an American put under Black-Scholes."""

import torch
import torch.nn.functional as F

from utility.model import PINN


class BlackScholesSobolev(PINN):
    def __init__(self, model_config, seed):
        super().__init__(model_config, seed)
        self.history = {
            'loss': [],
            'pde_loss': [],
            'J2_loss': [],
            'J3_loss': [],
            'J4_loss': [],
        }

    def set_params(self, K, r, sigma, T, S_min, S_max):
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.S_min = S_min
        self.S_max = S_max

    def train(self, batch_size, epochs, early_stopping, anneal_freq=500, alpha=0.9):
        """
        Train with Sobolev regularity losses J2, J3, J4 and automatic loss
        reweighting via learning rate annealing.

        Args:
            batch_size (int): Number of samples per training batch.
            epochs (int): Maximum number of training epochs.
            early_stopping (EarlyStopping): Early stopping mechanism.
            anneal_freq (int): Frequency of loss reweighting updates.
            alpha (float): EMA smoothing factor for loss reweighting.
        """
        self.loss_weights = {
            'pde': 0.25,
            'J2':  0.25,
            'J3':  0.25,
            'J4':  0.25,
        }

        # Fixed validation batches for early stopping
        val_t1, val_t2 = self.__sample_time_pairs(batch_size)
        val_S = self.__sample_S(batch_size)

        for i in range(epochs):
            t1, t2 = self.__sample_time_pairs(batch_size)
            S = self.__sample_S(batch_size)

            pde_loss = self.__pde_loss(t1, S)
            J2, J3, J4 = self.__sobolev_loss(t1, t2, S)

            if i > 2000 and i % anneal_freq == 0:
                unweighted = {'pde': pde_loss, 'J2': J2, 'J3': J3, 'J4': J4}
                self.__anneal_weights(unweighted, alpha)

            loss = self.__process_loss(pde_loss, J2, J3, J4)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # Validation loss (unweighted sum for early stopping)
            val_pde = self.__pde_loss(val_t1, val_S)
            val_J2, val_J3, val_J4 = self.__sobolev_loss(val_t1, val_t2, val_S)
            val_loss = val_pde + val_J2 + val_J3 + val_J4

            if i % 500 == 0:
                weight_str = "  ".join(
                    f"{k}={v:.3f}" for k, v in self.loss_weights.items())
                print(f"Iter {i:>6} | Train: {loss.item():.4e} "
                      f"| Val: {val_loss.item():.4e} | Weights: {weight_str}")

            if early_stopping and early_stopping.step(val_loss.item(), self.model):
                print(f"Early stopping at epoch {i}")
                early_stopping.restore(self.model)
                break

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def __process_loss(self, pde_loss, J2, J3, J4, update_dict=True):
        w_pde = self.loss_weights['pde'] * pde_loss
        w_J2  = self.loss_weights['J2']  * J2
        w_J3  = self.loss_weights['J3']  * J3
        w_J4  = self.loss_weights['J4']  * J4
        loss = w_pde + w_J2 + w_J3 + w_J4
        if update_dict:
            self.history['loss'].append(loss.item())
            self.history['pde_loss'].append(w_pde.item())
            self.history['J2_loss'].append(w_J2.item())
            self.history['J3_loss'].append(w_J3.item())
            self.history['J4_loss'].append(w_J4.item())
        return loss

    def __anneal_weights(self, unweighted_losses: dict, alpha: float):
        params = list(self.model.parameters())
        total_loss = sum(self.loss_weights[k] * v for k, v in unweighted_losses.items())
        total_grads = torch.autograd.grad(
            total_loss, params, retain_graph=True, create_graph=True, allow_unused=True
        )
        peak_grad = max(g.abs().max().item() for g in total_grads if g is not None)

        new_weights = {}
        for name, loss in unweighted_losses.items():
            weighted_loss = self.loss_weights[name] * loss
            grads = torch.autograd.grad(
                weighted_loss, params, retain_graph=True, create_graph=True, allow_unused=True
            )
            grad_tensors = [g for g in grads if g is not None]
            mean_grad = (
                sum(g.abs().mean().item() for g in grad_tensors) / len(grad_tensors)
            ) if grad_tensors else 1.0
            lambda_hat = peak_grad / (mean_grad + 1e-8)
            new_weights[name] = alpha * self.loss_weights[name] + (1.0 - alpha) * lambda_hat

        total = sum(new_weights.values())
        self.loss_weights = {k: v / total for k, v in new_weights.items()}

    def __sample_time_pairs(self, batch_size):
        """Sample pairs (t1, t2) in [0, T] with |t1 - t2| >= 0.01."""
        t1, t2, _, _ = self.sampler.uniform_pair(
            0, self.T, batch_size, 1, epsilon=0.01, boundary=False
        )
        return t1, t2

    def __sample_S(self, batch_size):
        return self.sampler.uniform(self.S_min, self.S_max, (batch_size, 1))

    def __bs_residual(self, t, S):
        """
        Black-Scholes PDE residual (forward-time convention):
            L[f] = -f_t - rS f_S - 0.5 sigma^2 S^2 f_SS + r f

        For the European BS PDE  f_t + rS f_S + 0.5 sigma^2 S^2 f_SS - r f = 0,
        we have L[f] = 0, so the sign convention is correct for forward time
        where t = 0 is today and t = T is expiry.
        """
        f = self.model(t, S)
        f_t, f_S = torch.autograd.grad(
            f, (t, S), grad_outputs=torch.ones_like(f), create_graph=True
        )
        f_SS = torch.autograd.grad(
            f_S, S, grad_outputs=torch.ones_like(f_S), create_graph=True
        )[0]
        residual = (
            -f_t
            - self.r * S * f_S
            - 0.5 * self.sigma ** 2 * S ** 2 * f_SS
            + self.r * f
        )
        return residual, f

    def __pde_loss(self, t, S):
        """
        American put complementarity loss.
        """
        t = t.detach().requires_grad_(True)
        S = S.detach().requires_grad_(True)

        residual, v = self.__bs_residual(t, S)
        g = F.relu(self.K - S)

        complementarity = torch.min(residual, v - g)
        pde_loss = torch.mean(complementarity ** 2)

        return pde_loss

    def __sobolev_loss(self, t1, t2, S_interior):
        """
        Three Sobolev regularity terms for the 1-D Black-Scholes problem,
        where u = v - g and g(S) = max(K - S, 0).

        J2  — H^1 norm of u(T, ·) on the interior (terminal condition)
        J3  — H^{3/4} in time, L^2 in space, on the lateral boundary
        J4  — H^{1/4} in time, L^2 in space, for the normal derivative
              ∂u/∂S on the lateral boundary
        """
        S_interior = S_interior.detach().requires_grad_(True)
        batch = S_interior.shape[0]
        ones = torch.ones(batch, 1)

        # ------------------------------------------------------------------
        # J2: H^1 norm of u(T, S) = v(T, S) - g(S) at t = T
        # ------------------------------------------------------------------
        t_terminal = ones * self.T
        v_T = self.model(t_terminal, S_interior)
        g_T = F.relu(self.K - S_interior)
        u_T = v_T - g_T

        J2_L2 = torch.mean(u_T ** 2)
        grad_u_T = torch.autograd.grad(
            u_T, S_interior, grad_outputs=torch.ones_like(u_T), create_graph=True
        )[0]
        J2_H1 = torch.mean(grad_u_T ** 2)
        J2 = J2_L2 + J2_H1

        # ------------------------------------------------------------------
        # Boundary evaluations for J3 and J4
        # ------------------------------------------------------------------
        # Need requires_grad for computing spatial derivatives at boundaries
        x1 = (ones * self.S_min).requires_grad_(True)   # S_min boundary
        x2 = (ones * self.S_max).requires_grad_(True)   # S_max boundary

        v_t1_x1 = self.model(t1, x1)
        v_t1_x2 = self.model(t1, x2)
        v_t2_x1 = self.model(t2, x1)
        v_t2_x2 = self.model(t2, x2)

        g_x1 = F.relu(self.K - x1)
        g_x2 = F.relu(self.K - x2)

        u_t1_x1 = v_t1_x1 - g_x1
        u_t1_x2 = v_t1_x2 - g_x2
        u_t2_x1 = v_t2_x1 - g_x1
        u_t2_x2 = v_t2_x2 - g_x2

        dt = torch.abs(t1 - t2)

        # ------------------------------------------------------------------
        # J3: H^{3/4} fractional norm in time on {S_min, S_max}
        #     Gagliardo seminorm: |u(t1,x) - u(t2,x)|^2 / |t1-t2|^{1+2s}
        #     with s = 3/4  =>  exponent = 2.5
        # ------------------------------------------------------------------
        denom_t_J3 = (dt ** 2.5).clamp_min(1e-8)

        time_frac_x1 = torch.mean(((u_t1_x1 - u_t2_x1) ** 2) / denom_t_J3)
        time_frac_x2 = torch.mean(((u_t1_x2 - u_t2_x2) ** 2) / denom_t_J3)

        J3_L2 = 0.5 * (torch.mean(u_t1_x1 ** 2) + torch.mean(u_t1_x2 ** 2))
        J3 = J3_L2 + time_frac_x1 + time_frac_x2

        # ------------------------------------------------------------------
        # J4: H^{1/4} fractional norm in time for the normal derivative
        #     ∂u/∂S on {S_min, S_max}
        #     s = 1/4  =>  exponent = 1.5
        # ------------------------------------------------------------------
        dv_dS_t1_x1 = torch.autograd.grad(
            v_t1_x1, x1, grad_outputs=torch.ones_like(v_t1_x1), create_graph=True
        )[0]
        dv_dS_t1_x2 = torch.autograd.grad(
            v_t1_x2, x2, grad_outputs=torch.ones_like(v_t1_x2), create_graph=True
        )[0]
        dv_dS_t2_x1 = torch.autograd.grad(
            v_t2_x1, x1, grad_outputs=torch.ones_like(v_t2_x1), create_graph=True
        )[0]
        dv_dS_t2_x2 = torch.autograd.grad(
            v_t2_x2, x2, grad_outputs=torch.ones_like(v_t2_x2), create_graph=True
        )[0]

        # g(S) = relu(K - S).  dg/dS = -1 if S < K, else 0.
        # At fixed boundary points this is a constant, so the fractional
        # differences of du/dS = dv/dS - dg/dS reduce to differences of dv/dS.
        dg_dS_x1 = -1.0 if self.S_min < self.K else 0.0
        dg_dS_x2 = -1.0 if self.S_max < self.K else 0.0

        du_dS_t1_x1 = dv_dS_t1_x1 - dg_dS_x1
        du_dS_t1_x2 = dv_dS_t1_x2 - dg_dS_x2
        du_dS_t2_x1 = dv_dS_t2_x1 - dg_dS_x1
        du_dS_t2_x2 = dv_dS_t2_x2 - dg_dS_x2

        denom_t_J4 = (dt ** 1.5).clamp_min(1e-8)

        time_frac_x1_J4 = torch.mean(
            ((du_dS_t1_x1 - du_dS_t2_x1) ** 2) / denom_t_J4
        )
        time_frac_x2_J4 = torch.mean(
            ((du_dS_t1_x2 - du_dS_t2_x2) ** 2) / denom_t_J4
        )

        spatial_denom_J4 = (self.S_max - self.S_min) ** 2
        spatial_frac_J4 = torch.mean(
            (du_dS_t1_x2 - du_dS_t1_x1) ** 2 / spatial_denom_J4
        )

        J4_L2 = 0.5 * (
            torch.mean(du_dS_t1_x1 ** 2) + torch.mean(du_dS_t1_x2 ** 2)
        )
        J4 = J4_L2 + time_frac_x1_J4 + time_frac_x2_J4 + spatial_frac_J4

        return J2, J3, J4