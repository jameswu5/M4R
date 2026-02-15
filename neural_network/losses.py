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


def heston_residual_nd(model, t, S, V, **kwargs):
    device = S.device
    dtype = S.dtype
    r = torch.as_tensor(kwargs["r"], device=device, dtype=dtype)
    kappa = torch.as_tensor(kwargs["kappa"], device=device, dtype=dtype)
    theta = torch.as_tensor(kwargs["theta"], device=device, dtype=dtype)
    sigma = torch.as_tensor(kwargs["sigma"], device=device, dtype=dtype)
    rho_sv = torch.as_tensor(kwargs["rho_sv"], device=device, dtype=dtype)
    rho_ss = torch.as_tensor(kwargs["rho_ss"], device=device, dtype=dtype)
    rho_vv = torch.as_tensor(kwargs["rho_vv"], device=device, dtype=dtype)

    t.requires_grad_(True)
    S.requires_grad_(True)
    V.requires_grad_(True)

    S_list = [S[:, i].unsqueeze(1) for i in range(S.shape[1])]
    V_list = [V[:, i].unsqueeze(1) for i in range(V.shape[1])]

    u = model(t, *S_list, *V_list)

    # First derivatives
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_S = torch.autograd.grad(u, S, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_V = torch.autograd.grad(u, V, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    N, d = S.shape

    # Second derivatives
    u_SS = torch.stack([torch.autograd.grad(u_S[:, i], S, grad_outputs=torch.ones_like(u_S[:, i]), create_graph=True)[0] for i in range(d)], dim=1)
    u_VV = torch.stack([torch.autograd.grad(u_V[:, i], V, grad_outputs=torch.ones_like(u_V[:, i]), create_graph=True)[0] for i in range(d)], dim=1)
    u_SV = torch.stack([torch.autograd.grad(u_S[:, i], V, grad_outputs=torch.ones_like(u_S[:, i]), create_graph=True)[0] for i in range(d)], dim=1)

    # Compute Operator L
    # Drift
    L_drift = torch.sum(r * S * u_S, dim=1, keepdim=True) + torch.sum(kappa * (theta - V) * u_V, dim=1, keepdim=True)

    # Diffusion
    sqrtV = torch.sqrt(V)
    vol_matrix = sqrtV.unsqueeze(2) * sqrtV.unsqueeze(1)

    # Asset-Asset Diffusion
    S_ij = S.unsqueeze(2) * S.unsqueeze(1)
    L_ss = 0.5 * rho_ss * vol_matrix * S_ij * u_SS

    # Vol-Vol Diffusion
    sig_ij = sigma.unsqueeze(1) * sigma.unsqueeze(0)
    L_vv = 0.5 * rho_vv * sig_ij * vol_matrix * u_VV

    # Mixed Spot-Vol (diagonal)
    u_SV_diag = torch.diagonal(u_SV, dim1=1, dim2=2)
    L_sv = rho_sv * sigma * V * S * u_SV_diag

    # Sum all second-order components
    L_diffusion = L_ss.view(N, -1).sum(dim=1, keepdim=True) + L_vv.view(N, -1).sum(dim=1, keepdim=True) + L_sv.sum(dim=1, keepdim=True)

    residual = -u_t - (L_drift + L_diffusion) + r * u

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
