
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import time
from matplotlib.patches import Rectangle
from functools import partial   
from matplotlib.gridspec import GridSpec
#gaussian_kde
from scipy.stats import gaussian_kde

from infinite import analytical_solution_inf, coefficient_inf, source_term_inf, generate_dataset_inf, evaluate_model_inf


def set_seed(seed=42):
    # Python's built-in random module
    
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    
    # Numpy's random module
    np.random.seed(seed)
    
    # PyTorch seed for CPU
    torch.manual_seed(seed)
    
    # PyTorch seed for all GPU devices (if using CUDA)
    torch.cuda.manual_seed_all(seed)
    
    # Make sure to disable CuDNN's non-deterministic optimizations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

 

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, hidden_units, activation_function):
        """
        Initializes a more general neural network model.

        Args:
            input_size (int): The size of the input layer.
            output_size (int): The size of the output layer.
            hidden_layers (int): The number of hidden layers.
            hidden_units (int): The number of units in each hidden layer.
            activation_function (nn.Module): The activation function to use in the hidden layers.
        """
        super(MLP, self).__init__()
        self.linear_in = nn.Linear(input_size, hidden_units)
        self.linear_out = nn.Linear(hidden_units, output_size)
        self.layers = nn.ModuleList([nn.Linear(hidden_units, hidden_units) for _ in range(hidden_layers)])
        self.act = activation_function

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor of the network.
        """
        x = self.linear_in(x)
        for layer in self.layers:
            x = self.act(layer(x))
        x = self.linear_out(x)
        return x    
 
class CoefficientNet(nn.Module):

    def __init__(self,
                 hidden_layers=3,
                 hidden_units=25,
                 activation=nn.Tanh()):
        super().__init__()

        layers = [
            nn.Linear(1, hidden_units),
            activation
        ]

        for _ in range(hidden_layers-1):
            layers += [
                nn.Linear(hidden_units, hidden_units),
                activation
            ]

        layers.append(nn.Linear(hidden_units,1))

        self.net = nn.Sequential(*layers)

    def forward(self, X):

        y = X[:,1:2]

        phi = self.net(y)

        k = 1.0 + 2.0*torch.sigmoid(phi)

        return k

def derivative(dy: torch.Tensor, x: torch.Tensor, order: int = 1) -> torch.Tensor:
    """
    Computes the derivative of a given tensor 'dy' with respect to another tensor 'x',
    up to a specified order.

    Args:
        dy (torch.Tensor): The tensor whose derivative is to be computed.
        x (torch.Tensor): The tensor with respect to which the derivative is to be computed.
        order (int, optional): The order of the derivative to compute. Defaults to 1, which
                               means a first-order derivative. Higher orders result in higher-order
                               derivatives.

    Returns:
        torch.Tensor: The computed derivative of 'dy' with respect to 'x', of the specified order.
    """
    for i in range(order):
        dy = torch.autograd.grad(
            dy, x, grad_outputs=torch.ones_like(dy), create_graph=True, retain_graph=True
        )[0]
    return dy  

def init_weights(m):
    """
    Initializes the weights and biases of a linear layer in the neural network using Xavier normalization.

    Args:
        m: The module or layer to initialize. If the module is of type nn.Linear, its weights and biases
           will be initialized.
    """
    if type(m) == nn.Linear:
        torch.manual_seed(42)  # fix inside
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.0)


def pde_loss_inf(
    model_u,
    model_k,
    X,
    F
):

    # --------------------------------------------------
    # Predictions
    # --------------------------------------------------

    u = model_u(X)
    k = model_k(X)
    #k = coefficient_torch(X)
    # --------------------------------------------------
    # grad(u)
    # --------------------------------------------------

    grad_u = torch.autograd.grad(
        u,
        X,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
    )[0]

    ux = grad_u[:, 0:1]
    uy = grad_u[:, 1:2]

    # --------------------------------------------------
    # Fluxes
    # --------------------------------------------------

    qx = k * ux
    qy = k * uy

    grad_qx = torch.autograd.grad(
        qx,
        X,
        grad_outputs=torch.ones_like(qx),
        create_graph=True,
    )[0]

    grad_qy = torch.autograd.grad(
        qy,
        X,
        grad_outputs=torch.ones_like(qy),
        create_graph=True,
    )[0]

    div = (
        grad_qx[:, 0:1]
        + grad_qy[:, 1:2]
    )

    residual = - div - F

    # --------------------------------------------------
    # PDE loss
    # --------------------------------------------------

    loss_pde = torch.mean(residual**2)
 
    return loss_pde



def observation_loss_u(
    model_u,
    X,
    U_true,
    criterion
):

    pred = model_u(X)

    mse = criterion(pred, U_true)

    return mse


def observation_loss_k(
    model_k,
    X,
    K_true,
    criterion):

    pred = model_k(X)

    mse = criterion(pred, K_true)

    return mse


def build_models(
    device,
    hidden_layers=3,
    hidden_units=25,
    activation=nn.Tanh(),
):
    model_u = MLP(
        input_size=2,
        output_size=1,
        hidden_layers=hidden_layers,
        hidden_units=hidden_units,
        activation_function=activation,
    ).to(device).double()

    model_k = CoefficientNet(
        hidden_layers=hidden_layers,
        hidden_units=hidden_units,
        activation=nn.Sigmoid(),
    ).to(device).double()

    model_u.apply(init_weights)
    model_k.apply(init_weights)

    return model_u, model_k

def train_dual_network(
    model_u,
    model_k,
    X_obs,
    U_obs,
    X_obs_k,
    K_obs,
    X_pde,
    F_pde,
    adam_lr=1e-3,
    adam_iters=1000,
    lbfgs_iters=2000,
    verbose=False,
    print_every=100, 
    adaptive_weights=True,
    alpha=0.1,
    update_every=100,
    ratio_threshold=10.0,       
):

    criterion = nn.MSELoss()

    parameters = (
        list(model_u.parameters())
        + list(model_k.parameters())
    )

    optimizer_adam = optim.Adam(
        parameters,
        lr=adam_lr,
    )

    optimizer_lbfgs = optim.LBFGS(
        parameters,
        lr=1.0,
        max_iter=lbfgs_iters,
        max_eval=lbfgs_iters,
        history_size=100,
        line_search_fn="strong_wolfe",
    )

    history = {
        "total": [],
        "u": [],
        "k": [],
        "pde": [],
        "lambda_u": [],
        "lambda_k": [],
        "lambda_pde": [],
    }

    # --------------------------------------------------
    # Adaptive weights
    # --------------------------------------------------

    lambda_u = 1.0
    lambda_k = 1.0
    lambda_pde = 1.0

    # --------------------------------------------------
    # Helper: compute losses
    # --------------------------------------------------

    def compute_losses():

        loss_u = observation_loss_u(
            model_u,
            X_obs,
            U_obs,
            criterion,
        )

        loss_k = observation_loss_k(
            model_k,
            X_obs_k,
            K_obs,
            criterion,
        )

        loss_pde = pde_loss_inf(
            model_u,
            model_k,
            X_pde,
            F_pde,
        )

        loss_u = torch.nan_to_num(loss_u)
        loss_k = torch.nan_to_num(loss_k)
        loss_pde = torch.nan_to_num(loss_pde)

        total = (
            lambda_u * loss_u
            + lambda_k * loss_k
            + lambda_pde * loss_pde
        )

        return total, loss_u, loss_k, loss_pde

    def update_loss_weights():

        nonlocal lambda_u, lambda_k, lambda_pde

        if len(history["u"]) < update_every:
            return

        V = np.array([
            np.mean(history["u"][-update_every:]),
            np.mean(history["k"][-update_every:]),
            np.mean(history["pde"][-update_every:])
        ])

        ratio = V.max() / (V.min() + 1e-12)

        if ratio <= ratio_threshold:
            return

        R = (V - V.min()) / (V.max() - V.min() + 1e-12)

        lambdas = 1.0 + alpha * R

        # Fastest task gets weight 1
        fastest = np.argmin(V)
        lambdas[fastest] = 1.0

        lambda_u, lambda_k, lambda_pde = lambdas

        if verbose:

            print(
                f"Updated weights -> "
                f"λu={lambda_u:.2f}, "
                f"λk={lambda_k:.2f}, "
                f"λpde={lambda_pde:.2f}"
            )

    # --------------------------------------------------
    # Helper: save history
    # --------------------------------------------------

    def save_history(total, loss_u, loss_k, loss_pde):

        history["total"].append(total.item())
        history["u"].append(loss_u.item())
        history["k"].append(loss_k.item())
        history["pde"].append(loss_pde.item())
        history["lambda_u"].append(lambda_u)
        history["lambda_k"].append(lambda_k)
        history["lambda_pde"].append(lambda_pde)

    # --------------------------------------------------
    # Adam
    # --------------------------------------------------
    if verbose:
        print("\n====================================")
        print("Training with Adam")
        print("====================================")

    model_u.train()
    model_k.train()

    for epoch in range(adam_iters):

        optimizer_adam.zero_grad()

        total, loss_u, loss_k, loss_pde = compute_losses()

        total.backward()
        optimizer_adam.step()

        save_history(total, loss_u, loss_k, loss_pde)

        if adaptive_weights and (epoch + 1) % update_every == 0:
            update_loss_weights()

        if verbose and epoch % print_every == 0:

            print(
                f"Adam {epoch:5d} | "
                f"Total={total.item():.3e} | "
                f"ObsU={loss_u.item():.3e} | "
                f"ObsK={loss_k.item():.3e} | "
                f"PDE={loss_pde.item():.3e}"
            )

    # --------------------------------------------------
    # L-BFGS
    # --------------------------------------------------

    if verbose:
        print("\n====================================")
        print("Training with L-BFGS")
        print("====================================")

    state = {"iter": 0}

    def closure():

        optimizer_lbfgs.zero_grad()

        total, loss_u, loss_k, loss_pde = compute_losses()

        total.backward()

        save_history(total, loss_u, loss_k, loss_pde)
     

        state["iter"] += 1

        if adaptive_weights and (state["iter"]) % update_every == 0:
            update_loss_weights()

        if verbose and state["iter"] % print_every == 0:

            print(
                f"L-BFGS {state['iter']:5d} | "
                f"Total={total.item():.3e} | "
                f"ObsU={loss_u.item():.3e} | "
                f"ObsK={loss_k.item():.3e} | "
                f"PDE={loss_pde.item():.3e}"
            )


        return total

    optimizer_lbfgs.step(closure)

    return history




def run_experiment_inf(
    hidden_layers=4,
    hidden_units=50,
    activation=nn.Tanh(),
    n_obs_u=100,
    n_obs_k=100,
    n_pde=10_000,
    alpha=0.5,
    beta=5.0,
    epsilon=1.0,
    adam_lr=1e-3,
    adam_iters=1000,
    lbfgs_iters=2000,
    device="cpu",
):
    """
    Run a single training experiment on the infinite-domain problem.

    Returns
    -------
    err_u : float
        Relative L2 error of the solution.
    err_k : float
        Relative L2 error of the coefficient.
    """

    # --------------------------------------------------
    # Build models
    # --------------------------------------------------
    model_u, model_k = build_models(
        device=device,
        hidden_layers=hidden_layers,
        hidden_units=hidden_units,
        activation=activation,
    )

    # --------------------------------------------------
    # Generate dataset
    # --------------------------------------------------
    (
        X_obs,
        U_obs,
        X_obs_k,
        K_obs,
        X_pde,
        F_pde,
        _,
        _,
        _,
    ) = generate_dataset_inf(
        alpha=alpha,
        beta=beta,
        epsilon=epsilon,
        n_obs_u=n_obs_u,
        n_obs_k=n_obs_k,
        n_pde=n_pde,
        device=device,
        plot=False,
    )

    # --------------------------------------------------
    # Train
    # --------------------------------------------------
    train_dual_network(
        model_u=model_u,
        model_k=model_k,
        X_obs=X_obs,
        U_obs=U_obs,
        X_obs_k=X_obs_k,
        K_obs=K_obs,
        X_pde=X_pde,
        F_pde=F_pde,
        adam_lr=adam_lr,
        adam_iters=adam_iters,
        lbfgs_iters=lbfgs_iters,
    )

    # --------------------------------------------------
    # Evaluate
    # --------------------------------------------------
    err_u, err_k = evaluate_model_inf(
        model_u=model_u,
        model_k=model_k,
        analytical_solution=analytical_solution_inf,
        coefficient=coefficient_inf,
        alpha=alpha,
        beta=beta,
        epsilon=epsilon,
        device=device,
    )

    return err_u, err_k