import torch


def compute_derivatives_nd(model, t, S):
    t = t.requires_grad_(True)
    S = S.requires_grad_(True)
    v = model(t, S)

    v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]  # shape (N, 1)
    v_S = torch.autograd.grad(v, S, grad_outputs=torch.ones_like(v), create_graph=True)[0]  # shape (N, n_assets)

    n_samples, n_assets = S.shape

    # Allocate Hessian tensor: shape (n_samples, n_assets, n_assets)
    H = torch.zeros((n_samples, n_assets, n_assets), dtype=v.dtype, device=v.device)

    for i in range(n_assets):
        vi = v_S[:, i].unsqueeze(-1)
        vi_S = torch.autograd.grad(vi, S, grad_outputs=torch.ones_like(vi), create_graph=True)[0]  # shape (N, n_assets)
        H[:, i, :] = vi_S

    # Enforce symmetry of Hessian
    H = 0.5 * (H + H.transpose(1, 2))

    # Now the H tensor is of form H[b, i, j] = d^2 v(b) / dS_i dS_j

    return v, v_t, v_S, H


def pde_residual_nd(v, v_t, v_S, H, S, r, Sigma):
    """
    Compute PDE residual for N-dimensional Black-Scholes operator.

    Parameters
    ----------
    v: (B, 1)           model value
    v_t: (B, 1)         time derivative
    v_S: (B, N)         first derivatives wrt S
    H: (B, N, N)        Hessian per batch
    S: (B, N)           asset prices (S1,...,SN)
    r: float            risk-free rate
    Sigma: (N, N) or (N,)
        If (N, N): full covariance matrix
        If (N,): diagonal variances (cross terms zero)
    """
    if not torch.is_tensor(Sigma):
        Sigma = torch.as_tensor(Sigma, dtype=S.dtype, device=S.device)
    else:
        Sigma = Sigma.to(dtype=S.dtype, device=S.device)

    B, N = S.shape

    drift = r * torch.sum(S * v_S, dim=1, keepdim=True)  # shape (B, 1)

    S_outer = S.unsqueeze(2) * S.unsqueeze(1)  # shape (B, N, N)

    # Broadcast Sigma to batch if needed:
    if Sigma.dim() == 1:
        Sigma_mat = torch.diag(Sigma).unsqueeze(0).expand(B, N, N)  # (B, N, N)
    elif Sigma.dim() == 2:
        Sigma_mat = Sigma.unsqueeze(0).expand(B, N, N)  # (B, N, N)
    else:
        raise ValueError("Sigma must be either (N,) or (N,N)")

    elements = Sigma_mat * S_outer * H  # shape (B, N, N)
    diffusion = 0.5 * torch.sum(elements, dim=(1, 2), keepdim=True)  # shape (B, 1)

    residual = -v_t - drift - diffusion + r * v  # shape (B, 1)
    return residual


def heston_residual(model, t, S, V, **kwargs):
    """
    Computes the Heston infinitesimal generator Lf and PDE residual applied to a neural network f.

    Parameters
    ----------
    model : torch.nn.Module
        Neural network approximating f(t, S, V)
    t, S, V : torch.Tensor
        Inputs with requires_grad=True
    kwargs : dict
        Must include:
            r      : risk-free rate
            kappa  : mean reversion speed
            theta  : long-run variance
            sigma  : vol of vol
            rho    : correlation

    Returns
    -------
    torch.Tensor
        Residual evaluated at (t, S, V)
    """

    r = kwargs["r"]
    kappa = kwargs["kappa"]
    theta = kwargs["theta"]
    sigma = kwargs["sigma"]
    rho = kwargs["rho"]

    t = t.requires_grad_(True)
    S = S.requires_grad_(True)
    V = V.requires_grad_(True)

    f = model(t, S, V)

    f_t, f_S, f_V = torch.autograd.grad(
        f, (t, S, V), grad_outputs=torch.ones_like(f), create_graph=True, retain_graph=True
    )

    f_SS = torch.autograd.grad(
        f_S, S, grad_outputs=torch.ones_like(f_S), create_graph=True, retain_graph=True
    )[0]

    f_VV = torch.autograd.grad(
        f_V, V, grad_outputs=torch.ones_like(f_V), create_graph=True, retain_graph=True
    )[0]

    f_SV = torch.autograd.grad(
        f_S,
        V,
        grad_outputs=torch.ones_like(f_S),
        create_graph=True,
        retain_graph=True
    )[0]

    Lf = (
        r * S * f_S
        + kappa * (theta - V) * f_V
        + 0.5 * (
            S**2 * V * f_SS
            + 2.0 * rho * sigma * S * V * f_SV
            + sigma**2 * V * f_VV
        )
    )

    residual = -f_t - Lf + r * f

    return residual
