import torch


def bs_residual(model, t, S, r, sigma):
    f = model(t, S)
    f_t, f_S = torch.autograd.grad(
        f, (t, S), grad_outputs=torch.ones_like(f), create_graph=True
    )
    f_SS = torch.autograd.grad(
        f_S, S, grad_outputs=torch.ones_like(f_S), create_graph=True
    )[0]

    return -f_t - r * S * f_S - 0.5 * sigma**2 * S**2 * f_SS + r * f


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


def heston_residual_nd(model, t, S, V, **kwargs):
    """
    Computes PDE residual for multi-asset Heston with common variance factor.
    """

    r = kwargs["r"]
    kappa = kwargs["kappa"]
    theta = kwargs["theta"]
    sigma_bar = kwargs["sigma_bar"]        # vol of variance
    sigma = kwargs["sigma"]                # (n,), vol of each asset
    Sigma = kwargs["Sigma"]                # (n,n), correlation matrix
    rho = kwargs["rho"]                    # (n,) correlation S_i - V

    # Enable autograd
    t.requires_grad_(True)
    S.requires_grad_(True)
    V.requires_grad_(True)

    # Forward pass
    u = model(t, S, V)

    # First derivatives
    u_t = torch.autograd.grad(
        u, t, torch.ones_like(u),
        create_graph=True
    )[0]

    u_S = torch.autograd.grad(
        u, S, torch.ones_like(u),
        create_graph=True
    )[0]                      # (N,n)

    u_V = torch.autograd.grad(
        u, V, torch.ones_like(u),
        create_graph=True
    )[0]

    N, n = S.shape

    # Second derivatives S_i S_k
    u_SS = []
    for i in range(n):
        grad_i = torch.autograd.grad(
            u_S[:, i:i+1], S,
            torch.ones_like(u_S[:, i:i+1]),
            create_graph=True
        )[0]
        u_SS.append(grad_i.unsqueeze(1))

    u_SS = torch.cat(u_SS, dim=1)   # (N,n,n)

    # Second derivative V V
    u_VV = torch.autograd.grad(
        u_V, V, torch.ones_like(u_V),
        create_graph=True
    )[0]

    # Cross derivatives S_i V
    u_SV = []
    for i in range(n):
        grad_i = torch.autograd.grad(
            u_S[:, i:i+1], V,
            torch.ones_like(u_S[:, i:i+1]),
            create_graph=True
        )[0]
        u_SV.append(grad_i)

    u_SV = torch.cat(u_SV, dim=1)   # (N,n)

    # Generator Lu

    # Drift terms
    L = r * torch.sum(S * u_S, dim=1, keepdim=True)
    L += kappa * (theta - V) * u_V

    # Asset-asset diffusion
    diffusion_SS = 0
    for i in range(n):
        for k in range(n):
            coeff = (
                V
                * sigma[i] * sigma[k]
                * Sigma[i, k]
                * S[:, i:i+1] * S[:, k:k+1]
            )
            diffusion_SS += coeff * u_SS[:, i, k:k+1]

    L += 0.5 * diffusion_SS

    # Variance diffusion
    L += 0.5 * sigma_bar**2 * V * u_VV

    # Cross S-V terms
    cross = 0
    for i in range(n):
        coeff = (
            V
            * sigma[i]
            * sigma_bar
            * rho[i]
            * S[:, i:i+1]
        )
        cross += coeff * u_SV[:, i:i+1]

    L += cross

    # PDE residual
    residual = -u_t - L + r * u

    return residual


def test_compute_derivatives_nd():
    def simple_model(t, S):
        return t**2 + torch.sum(S**2, dim=1, keepdim=True)  # shape (B,1)

    B, N = 5, 3
    t = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
    S = torch.tensor([[1.0, 2.0, 3.0],
                      [2.0, 3.0, 4.0],
                      [3.0, 4.0, 5.0],
                      [4.0, 5.0, 6.0],
                      [5.0, 6.0, 7.0]])

    v, v_t, v_S, H = compute_derivatives_nd(simple_model, t, S)

    assert torch.allclose(v_t, 2*t)
    assert torch.allclose(v_S, 2*S)
    for i in range(N):
        for j in range(N):
            expected = 2.0 if i == j else 0.0
            assert torch.allclose(H[:, i, j], torch.full((B,), expected))

    print("test_compute_derivatives_nd passed")


def test_pde_residual_nd():
    # Test 1: single asset, linear payoff
    B = 3
    N = 1
    S = torch.tensor([[1.0], [2.0], [3.0]])
    v = S.clone()
    v_t = torch.zeros_like(v)
    v_S = torch.ones_like(S)
    H = torch.zeros((B, N, N))
    r = 0.05
    Sigma = torch.tensor([0.0])  # no diffusion

    residual = pde_residual_nd(v, v_t, v_S, H, S, r, Sigma)
    expected = -v_t - r*S*v_S + r*v
    assert torch.allclose(residual, expected, atol=1e-6), "Test 1 failed"

    # Test 2: single asset, quadratic payoff
    v = S**2
    v_t = torch.zeros_like(v)
    v_S = 2*S
    H = 2*torch.ones((B, N, N))
    Sigma = torch.tensor([0.2**2])
    residual = pde_residual_nd(v, v_t, v_S, H, S, r, Sigma)

    # Compute expected manually using same formulas
    S_outer = S.unsqueeze(2) * S.unsqueeze(1)
    Sigma_mat = torch.diag(Sigma).unsqueeze(0).expand(B, N, N)
    diffusion = 0.5 * torch.sum(Sigma_mat * S_outer * H, dim=(1, 2), keepdim=True)
    drift = r * torch.sum(S * v_S, dim=1, keepdim=True)
    expected = -v_t - drift - diffusion + r*v
    assert torch.allclose(residual, expected, atol=1e-6), "Test 2 failed"

    # Test 3: two assets, linear payoff, diagonal Sigma
    B = 2
    N = 2
    S = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    v = S.sum(dim=1, keepdim=True)  # v = S1 + S2
    v_t = torch.zeros_like(v)
    v_S = torch.ones_like(S)
    H = torch.zeros((B, N, N))
    r = 0.03
    Sigma = torch.tensor([0.1**2, 0.2**2])
    residual = pde_residual_nd(v, v_t, v_S, H, S, r, Sigma)
    drift = r * (S*v_S).sum(dim=1, keepdim=True)
    diffusion = torch.zeros_like(v)  # H = 0 => diffusion = 0
    expected = -v_t - drift - diffusion + r*v
    assert torch.allclose(residual, expected, atol=1e-6), "Test 3 failed"

    print("test_pde_residual_nd passed")


if __name__ == "__main__":
    test_compute_derivatives_nd()
    test_pde_residual_nd()
