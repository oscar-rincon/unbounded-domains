
import sympy as sp
import numpy as np


def source_term_inf(xs_inf, ys_inf, alpha_s_inf, beta_s_inf, eps_s_inf):


    u = sp.exp(
        -alpha_s_inf*(xs_inf**2 + ys_inf**2)
    ) * sp.cos(beta_s_inf*ys_inf)

    k = 1 + 2/(1 + sp.exp(-ys_inf/eps_s_inf))

    # -------------------------
    # Poisson source term
    # -------------------------
    ux = sp.diff(u, xs_inf)
    uy = sp.diff(u, ys_inf)

    f = -(
        sp.diff(k*ux, xs_inf)
        +
        sp.diff(k*uy, ys_inf)
    )

    f = sp.simplify(f)

    f_numpy = sp.lambdify(
        (xs_inf, ys_inf, alpha_s_inf, beta_s_inf, eps_s_inf),
        f,
        "numpy",
    )

    return f_numpy
 
def analytical_solution_inf(xs_inf, ys_inf, alpha_s_inf, beta_s_inf):
    """
    Analytical solution u(x,y).
    """
    return np.exp(-alpha_s_inf * (xs_inf**2 + ys_inf**2)) * np.cos(beta_s_inf * ys_inf)

def coefficient_inf(xs_inf, ys_inf, epsilon):
    """
    Variable coefficient k(y).
    """
    return 1 + 2 / (1 + np.exp(-ys_inf / epsilon))