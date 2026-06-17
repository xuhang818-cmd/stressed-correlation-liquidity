"""
dcc_garch.py
------------
Engle (2002) DCC-GARCH, two steps:

  Step 1  For each asset i fit a univariate GARCH(1,1):
              sigma2_t = omega + alpha * eps_{t-1}^2 + beta * sigma2_{t-1}
          Standardize: z_{i,t} = eps_{i,t} / sigma_{i,t}.
          (Standardizing strips out volatility so step 2 is purely co-movement.)

  Step 2  Let Qbar = unconditional corr of the standardized residuals z.
              Q_t = (1 - a - b) * Qbar + a * (z_{t-1} z_{t-1}') + b * Q_{t-1}
              R_t = diag(Q_t)^{-1/2} Q_t diag(Q_t)^{-1/2}   (valid corr matrix)
          (a, b) estimated by maximizing the Gaussian DCC log-likelihood,
          a, b >= 0 and a + b < 1 for stationarity.
"""
import numpy as np
import pandas as pd
from arch import arch_model
from scipy.optimize import minimize


def fit_univariate_garch(returns: pd.DataFrame):
    """Return standardized residuals z (T x N) and conditional vols (T x N)."""
    z, vol = {}, {}
    for col in returns.columns:
        am = arch_model(returns[col] * 100, mean="Constant", vol="GARCH", p=1, q=1, dist="normal")
        res = am.fit(disp="off")
        z[col] = res.std_resid                      # scale-invariant
        vol[col] = res.conditional_volatility / 100.0
    Z = pd.DataFrame(z).dropna()
    V = pd.DataFrame(vol).reindex(Z.index)
    return Z, V


def _dcc_negloglik(theta, z, Qbar):
    a, b = theta
    if a < 0 or b < 0 or a + b >= 0.99999:
        return 1e12
    T = z.shape[0]
    Q = Qbar.copy()
    ll = 0.0
    for t in range(T):
        if t > 0:
            outer = np.outer(z[t - 1], z[t - 1])
            Q = (1 - a - b) * Qbar + a * outer + b * Q
        d = np.sqrt(np.diag(Q))
        R = Q / np.outer(d, d)
        R = (R + R.T) / 2.0
        sign, logdet = np.linalg.slogdet(R)
        if sign <= 0:
            return 1e12
        zt = z[t]
        ll += logdet + zt @ np.linalg.solve(R, zt)
    return 0.5 * ll


def fit_dcc(Z: pd.DataFrame):
    """Estimate (a, b) and return them plus the average pairwise correlation series."""
    z = Z.values
    Qbar = np.corrcoef(z.T)
    opt = minimize(_dcc_negloglik, x0=[0.02, 0.95], args=(z, Qbar),
                   method="L-BFGS-B", bounds=[(1e-6, 0.5), (1e-6, 0.999)])
    a, b = opt.x

    # Second pass: rebuild R_t and record the mean off-diagonal correlation.
    T, N = z.shape
    Q = Qbar.copy()
    iu = np.triu_indices(N, k=1)
    avg_corr = np.empty(T)
    for t in range(T):
        if t > 0:
            outer = np.outer(z[t - 1], z[t - 1])
            Q = (1 - a - b) * Qbar + a * outer + b * Q
        d = np.sqrt(np.diag(Q))
        R = Q / np.outer(d, d)
        avg_corr[t] = R[iu].mean()
    return {"a": a, "b": b,
            "avg_corr": pd.Series(avg_corr, index=Z.index, name="avg_corr")}
