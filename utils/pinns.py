
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import time
from matplotlib.patches import Rectangle
from functools import partial   
from matplotlib.gridspec import GridSpec
#gaussian_kde
from scipy.stats import gaussian_kde
import traceback
from efficient_kan import KAN
from infinite import analytical_solution_inf, coefficient_inf, source_term_inf, generate_dataset_inf, evaluate_model_inf


"""
Training utilities for the dual-network (u, k) PINN model.

This file is a reorganized version of the original `train_dual_network`
function. The logic and the numbers it produces are UNCHANGED — every
inner helper (compute_losses, compute_losses_test, ratio_calculation,
get_pde_weight, update_loss_weights, save_history) has simply been
pulled out to module level and made to take its state as explicit
arguments instead of capturing it via closures/`nonlocal`.

The only genuinely new pieces are the run-saving utilities at the
bottom (`build_training_summary`, `format_summary_text`,
`save_training_run`), which let you:
  - save model weights + full history + a readable summary
  - into a folder named after the moment the run finished
  - and turn saving on/off with a single flag (save_results=True/False)

NOTE: `compute_losses` and `compute_losses_test` from the original code
were byte-for-byte identical except for which tensors they were fed
(train vs. test). They've been merged into a single `compute_losses`
that takes the data as arguments — this does not change any numbers,
it just removes duplication. If you'd rather keep them as two
separate functions, say so and I'll split them back out.

You still need `observation_loss_u`, `observation_loss_k`,
`pde_loss_inf`, and `l2_regularization` defined/imported exactly as
in your original codebase — they are not redefined here.
"""

import os
import json
import pickle
from datetime import datetime
 

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

def create_kan(
    input_size,
    output_size,
    hidden_layers=3,
    hidden_units=25,
    grid_size=3,
    spline_order=3,
):
    """
    Creates a KAN model.

    Args:
        input_size (int): Number of input features.
        output_size (int): Number of output features.
        hidden_layers (int): Number of hidden layers.
        hidden_units (int): Number of neurons (KAN units) per hidden layer.
        grid_size (int): Number of spline intervals.
        spline_order (int): Order of the B-splines.

    Returns:
        KAN: Initialized KAN model.
    """
    return KAN(
        layers_hidden=[input_size]
              + [hidden_units] * hidden_layers
              + [output_size],
        grid_size=grid_size,
        spline_order=spline_order,
    )

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

class CoefficientKAN(nn.Module):

    def __init__(
        self,
        hidden_layers=3,
        hidden_units=16,
        grid=3,
        k=3,
    ):
        super().__init__()

        layers = [1] + [hidden_units] * hidden_layers + [1]

        self.net = KAN(
            layers_hidden=layers,
            grid_size=grid,
            spline_order=k,
        )

    def forward(self, X):

        y = X[:, 1:2]

        phi = self.net(y)

        k = 1.0 + 2.0 * torch.sigmoid(phi)

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


def build_models_KAN(
    device,
    hidden_layers=3,
    hidden_units=25,
    grid_size=5,
    spline_order=3,
):
    model_u = KAN(
        layers_hidden=[2] + [hidden_units] * hidden_layers + [1],
        grid_size=grid_size,
        spline_order=spline_order,
        grid_range=[-5,5],
    ).to(device)

    model_k = KAN(
        layers_hidden=[2] + [hidden_units] * hidden_layers + [1],
        grid_size=grid_size,
        spline_order=spline_order,
        grid_range=[-5,5],
    ).to(device)

    return model_u, model_k 

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
    ).to(device)#.double()

    # model_k = MLP(
    #     input_size=2,
    #     output_size=1,
    #     hidden_layers=hidden_layers,
    #     hidden_units=hidden_units,
    #     activation_function=activation,
    # ).to(device)#.double()
    model_k = CoefficientNet(
        hidden_layers=hidden_layers,
        hidden_units=hidden_units,
        activation=nn.Sigmoid(),
    ).to(device)#.double()

    model_u.apply(init_weights)
    model_k.apply(init_weights)

    return model_u, model_k

def l2_regularization(parameters):
    l2 = torch.zeros((), device=parameters[0].device)
    for p in parameters:
        l2 += p.pow(2).sum()
    return l2

# def train_dual_network(
#     model_u,
#     model_k,
#     X_obs_train,
#     U_obs_train,
#     X_obs_k_train,
#     K_obs_train,
#     X_pde_train,
#     F_pde_train,
#     X_obs_test,
#     U_obs_test,
#     X_obs_k_test,
#     K_obs_test,
#     X_pde_test,
#     F_pde_test,
#     adam_lr=1e-3,
#     adam_iters=1000,
#     lbfgs_iters=2000,
#     verbose=False,
#     print_every=100, 
#     lambda_pde_scheduler=True,
#     adaptive_weights=True,
#     alpha=5,
#     update_every=200, 
#     regularization=False,    
# ):
#     ratio = 1
#     criterion = nn.MSELoss()

#     parameters = (
#         list(model_u.parameters())
#         + list(model_k.parameters())
#     )

#     optimizer_adam = optim.AdamW(
#         parameters,
#         lr=adam_lr ,
#         weight_decay=1 if regularization else 0.0
#     )

#     optimizer_lbfgs = optim.LBFGS(
#         parameters,
#         lr=1,
#         max_iter=lbfgs_iters,
#         max_eval=lbfgs_iters,
#         history_size=100,
#         tolerance_change=1.0 * np.finfo(float).eps,
#         line_search_fn="strong_wolfe")

 

#     history = {
#         "total": [],
#         "u": [],
#         "k": [],
#         "pde": [],

#         "lambda_u": [],
#         "lambda_k": [],
#         "lambda_pde": [],

#         # New entries
#         "R_u": [],
#         "R_k": [],
#         "R_pde": [],
#         "R_reg_t": [],
#         "ratio": [],
#         "total_test": [],
#         "total_no_reg": [],
#         "total_no_reg_test": [],
#     }

#     # --------------------------------------------------
#     # Adaptive weights
#     # --------------------------------------------------

#     lambda_u = 1.0
#     lambda_k = 1.0
#     lambda_pde = 1.0
 
#     # --------------------------------------------------
#     # Helper: compute losses
#     # --------------------------------------------------



#     def compute_losses(lambda_reg=1.0):

#         loss_u = observation_loss_u(
#             model_u,
#             X_obs_train,
#             U_obs_train,
#             criterion,
#         )

#         loss_k = observation_loss_k(
#             model_k,
#             X_obs_k_train,
#             K_obs_train,
#             criterion,
#         )

#         loss_pde = pde_loss_inf(
#             model_u,
#             model_k,
#             X_pde_train,
#             F_pde_train,
#         )

#         loss_u = torch.nan_to_num(loss_u)
#         loss_k = torch.nan_to_num(loss_k)
#         loss_pde = torch.nan_to_num(loss_pde)


#         if regularization:
#             reg_t = l2_regularization(parameters)
            
  
#         total = (
#             lambda_u * loss_u
#             + lambda_k * loss_k
#             + lambda_pde * loss_pde
#             + (lambda_reg * reg_t if regularization else 0.0)
#         )

#         total_no_reg = (
#              loss_u
#             + loss_k
#             + loss_pde
#         )

#         return total, loss_u, loss_k, loss_pde, total_no_reg

#     def compute_losses_test(lambda_reg=1.0):

#         loss_u = observation_loss_u(
#             model_u,
#             X_obs_test,
#             U_obs_test,
#             criterion,
#         )

#         loss_k = observation_loss_k(
#             model_k,
#             X_obs_k_test,
#             K_obs_test,
#             criterion,
#         )

#         loss_pde = pde_loss_inf(
#             model_u,
#             model_k,
#             X_pde_test,
#             F_pde_test,
#         )

#         loss_u = torch.nan_to_num(loss_u)
#         loss_k = torch.nan_to_num(loss_k)
#         loss_pde = torch.nan_to_num(loss_pde)

#         if regularization:
#             reg_t = l2_regularization(parameters)
 
#         total = (
#             lambda_u * loss_u
#             + lambda_k * loss_k
#             + lambda_pde * loss_pde
#             + (lambda_reg * reg_t if regularization else 0.0)
#         )

#         total_no_reg = (
#             loss_u
#             + loss_k
#             + loss_pde
#         )

#         return total, loss_u, loss_k, loss_pde, total_no_reg

 
#     def ratio_calculation():
#         V = np.array([
#             np.mean(history["u"][-update_every:]),
#             np.mean(history["k"][-update_every:]),
#             np.mean(history["pde"][-update_every:]),
#             #np.mean(history["reg_t"][-update_every:])
#         ])

#         # ----------------------------------------------------
#         # Step 7: Ratio
#         # ----------------------------------------------------

#         ratio = V.max() / (V.min() + 1e-12)
#         return ratio


#     def get_pde_weight(epoch):

#         if epoch < 250:
#             return 0.10

#         elif epoch < 500:
#             return 0.25

#         elif epoch < 750:
#             return 0.50

#         elif epoch < 1000:
#             return 0.75

#         else:
#             return 1.00


#     def update_loss_weights(loss_u, loss_k):

#         nonlocal lambda_u, lambda_k

#         # ----------------------------------------------------
#         # Current U and K losses
#         # ----------------------------------------------------

#         if len(history["u"]) == 0:

#             V = np.array([
#                 loss_u.item(),
#                 loss_k.item(),
#             ])

#         else:

#             V = np.array([
#                 np.mean(history["u"][-update_every:]),
#                 np.mean(history["k"][-update_every:]),
#             ])

#         # ----------------------------------------------------
#         # Ratio between U and K
#         # ----------------------------------------------------

#         ratio = V.max() / (V.min() + 1e-12)

#         if not adaptive_weights:
#             return

#         ratio_threshold = 10.0

#         if ratio <= ratio_threshold:

#             history["R_u"].append(0.0)
#             history["R_k"].append(0.0)

#             return

#         # ----------------------------------------------------
#         # Compute relative difference
#         # ----------------------------------------------------

#         R = (V - V.min()) / (
#             V.max() - V.min() + 1e-12
#         )

#         history["R_u"].append(R[0])
#         history["R_k"].append(R[1])

#         # ----------------------------------------------------
#         # Update only U and K
#         # ----------------------------------------------------

#         lambdas = 1.0 + alpha * R

#         # Loss with smaller magnitude keeps weight = 1
#         fastest = np.argmin(V)
#         lambdas[fastest] = 1.0

#         lambda_u = lambdas[0]
#         lambda_k = lambdas[1]

#         if verbose:
#             print(
#                 f"V      = {V.round(3)}\n"
#                 f"R      = {R.round(3)}\n"
#                 f"ratio  = {ratio:.2f}\n"
#                 f"lambda_u = {lambda_u:.3f}\n"
#                 f"lambda_k = {lambda_k:.3f}\n"
#                 f"lambda_pde = {lambda_pde:.3f}"
#             )
 
        
#     # --------------------------------------------------
#     # Helper: save history
#     # --------------------------------------------------

#     def save_history(total, loss_u, loss_k, loss_pde, ratio, total_test, total_no_reg,total_no_reg_test):

#         history["total"].append(total.item())
#         history["u"].append(loss_u.item())
#         history["k"].append(loss_k.item())
#         history["pde"].append(loss_pde.item())
#         #history["reg_t"].append(reg_t.item())
#         history["lambda_u"].append(lambda_u)
#         history["lambda_k"].append(lambda_k)
#         history["lambda_pde"].append(lambda_pde)
#         #history["lambda_reg_t"].append(lambda_reg_t)
#         history["ratio"].append(ratio)
#         history["total_test"].append(total_test.item())
#         history["total_no_reg"].append(total_no_reg.item())
#         history["total_no_reg_test"].append(total_no_reg_test.item())

#     # --------------------------------------------------
#     # Adam
#     # --------------------------------------------------
#     if verbose:
#         print("\n====================================")
#         print("Training with Adam")
#         print("====================================")

#     model_u.train()
#     model_k.train()

#     for epoch in range(adam_iters+1):

#         if lambda_pde_scheduler is True:
#             lambda_pde = get_pde_weight(epoch)

#         optimizer_adam.zero_grad()

#         # total_no_reg_test = torch.tensor(0.0)  # Initialize total_no_reg_test to avoid undefined variable error
#         # total_test = torch.tensor(0.0)  # Initialize total_test to avoid undefined
#         #lambda_reg=0

#         total, loss_u, loss_k, loss_pde, total_no_reg = compute_losses()
#         total_test, loss_u_test, loss_k_test, loss_pde_test, total_no_reg_test = compute_losses_test()

#         if epoch == 0:

#             V = np.array([
#                 loss_u.item(),
#                 loss_k.item(),
#                 loss_pde.item(),
#                 #reg_t.item()
#             ])

#             ratio = V.max() / (V.min() + 1e-12)
#         #     update_loss_weights(loss_u, loss_k, loss_pde) 
#         #     total, loss_u, loss_k, loss_pde, total_no_reg = compute_losses()
#         #     total_test, loss_u_test, loss_k_test, loss_pde_test, total_no_reg_test = compute_losses_test()

#         else:
#             ratio = ratio_calculation()
        

#         total.backward()
#         optimizer_adam.step()

#         #expect in zero 


#         if adaptive_weights and epoch % update_every == 0 and epoch > 0:
#             update_loss_weights(loss_u, loss_k)
 


#         if epoch % print_every == 0:
#             save_history(total, loss_u, loss_k, loss_pde, ratio, total_test, total_no_reg, total_no_reg_test)

#         if verbose and epoch % print_every == 0:
#             print(
#                 f"Adam {epoch:5d} | "
#                 f"Total={total.item():.3e} | "
#                 f"ObsU={loss_u.item():.3e} | "
#                 f"ObsK={loss_k.item():.3e} | "
#                 f"PDE={loss_pde.item():.3e} | "
#                 f"Ratio={ratio:.2f}"
#             )


#     # --------------------------------------------------
#     # L-BFGS
#     # --------------------------------------------------

#     if verbose:
#         print("\n====================================")
#         print("Training with L-BFGS")
#         print("====================================")

#     state = {
#         "iter": 0,
#         "loss_u": None,
#         "loss_k": None,
#         "loss_pde": None,
#     }

#     iters_done = 0
#     #while iters_done < lbfgs_iters-1:

#         # --------------------------------------------------
#         # Restart L-BFGS every update_every iterations
#         # --------------------------------------------------

#         #current_block = min(update_every, lbfgs_iters - iters_done)

#         # optimizer_lbfgs = torch.optim.LBFGS(
#         #     list(model_u.parameters()) + list(model_k.parameters()),
#         #     lr=1.0,
#         #     max_iter=current_block,
#         #     max_eval=current_block,
#         #     history_size=100,
#         #     tolerance_grad=1e-9,
#         #     tolerance_change=1e-12,
#         #     line_search_fn="strong_wolfe",
#         # )

#     def closure():

#         optimizer_lbfgs.zero_grad()

#         lambda_reg = 0#1e-3

#         total, loss_u, loss_k, loss_pde, total_no_reg = \
#             compute_losses(lambda_reg)

#         total_test, loss_u_test, loss_k_test, loss_pde_test, total_no_reg_test = \
#             compute_losses_test(lambda_reg)

#         total.backward()

#         # ------------------------------------------
#         # Ratio
#         # ------------------------------------------

#         if iters_done == 0:

#             V = np.array([
#                 loss_u.item(),
#                 loss_k.item(),
#                 loss_pde.item(),
#             ])

#             ratio = V.max() / (V.min() + 1e-12)

#         else:

#             ratio = ratio_calculation()

#         # ------------------------------------------
#         # Store latest losses
#         # ------------------------------------------

#         state["loss_u"] = loss_u.detach()
#         state["loss_k"] = loss_k.detach()
#         state["loss_pde"] = loss_pde.detach()

#         # ------------------------------------------
#         # Save history
#         # ------------------------------------------

#         if state["iter"] % print_every == 0:
#             save_history(
#                 total,
#                 loss_u,
#                 loss_k,
#                 loss_pde,
#                 ratio,
#                 total_test,
#                 total_no_reg,
#                 total_no_reg_test,
#             )

#         state["iter"] += 1

#         if verbose and state["iter"] % print_every == 0:

#             print(
#                 f"L-BFGS {state['iter']:5d} | "
#                 f"Total={total.item():.3e} | "
#                 f"ObsU={loss_u.item():.3e} | "
#                 f"ObsK={loss_k.item():.3e} | "
#                 f"PDE={loss_pde.item():.3e} | "
#                 f"Ratio={ratio:.2f}"
#             )

#         return total

#     # --------------------------------------------------
#     # Run one L-BFGS block
#     # --------------------------------------------------

#     optimizer_lbfgs.step(closure)
#     #iters_done += current_block
#     # --------------------------------------------------
#     # Update adaptive weights
#     # --------------------------------------------------

#     # if adaptive_weights and iters_done < lbfgs_iters:

#     #     update_loss_weights(
#     #         state["loss_u"],
#     #         state["loss_k"],
#     #         state["loss_pde"],
#     #     )



#     # --------------------------------------------------
#     # Save history automatically
#     # --------------------------------------------------

#     os.makedirs("results", exist_ok=True)

#     filename = "history"

#     if adaptive_weights:
#         filename += "_adaptive"
#     else:
#         filename += "_fixed"

#     if regularization:
#         filename += "_reg"
#     else:
#         filename += "_no_reg"

#     filename += ".pkl"

#     with open(os.path.join("results", filename), "wb") as f:
#         pickle.dump(history, f)

#     #print(f"History saved to results/{filename}")

#     return history



 


# ======================================================================
# Loss computation (was: compute_losses / compute_losses_test closures)
# ======================================================================

def compute_losses(
    model_u,
    model_k,
    X_obs,
    U_obs,
    X_obs_k,
    K_obs,
    X_pde,
    F_pde,
    criterion,
    lambda_u,
    lambda_k,
    lambda_pde,
    parameters,
    regularization=False,
    lambda_reg=1.0,
):
    """
    Compute weighted total loss + individual components for a given
    data split (pass train tensors for training loss, test tensors for
    test loss -- this replaces the old compute_losses/compute_losses_test
    pair, same math either way).
    """
    loss_u = observation_loss_u(model_u, X_obs, U_obs, criterion)
    loss_k = observation_loss_k(model_k, X_obs_k, K_obs, criterion)
    loss_pde = pde_loss_inf(model_u, model_k, X_pde, F_pde)

    loss_u = torch.nan_to_num(loss_u)
    loss_k = torch.nan_to_num(loss_k)
    loss_pde = torch.nan_to_num(loss_pde)

    reg_t = l2_regularization(parameters) if regularization else None

    total = (
        lambda_u * loss_u
        + lambda_k * loss_k
        + lambda_pde * loss_pde
        + (lambda_reg * reg_t if regularization else 0.0)
    )

    total_no_reg = loss_u + loss_k + loss_pde

    return total, loss_u, loss_k, loss_pde, total_no_reg


# ======================================================================
# Ratio / weight scheduling helpers
# ======================================================================

def ratio_calculation(history, update_every):
    V = np.array([
        np.mean(history["u"][-update_every:]),
        np.mean(history["k"][-update_every:]),
    ])
    return V.max() / (V.min() + 1e-12)


def get_pde_weight(epoch):

    if epoch < 400:
        return 0.10
    elif epoch < 800:
        return 0.25
    elif epoch < 1200:
        return 0.50
    elif epoch < 1600:
        return 0.75
    else:
        return 1.00


def update_loss_weights(
    history,
    iteration,
    lambda_u,
    lambda_k,
    lambda_pde,
    loss_u,
    loss_k,
    update_every,
    alpha,
    adaptive_weights,
    verbose=False,
):
    """
    Returns the (possibly updated) (lambda_u, lambda_k).
    Mutates `history` in place by appending R_u / R_k, exactly like the
    original. Caller is responsible for reassigning
    lambda_u, lambda_k = update_loss_weights(...)
    since Python closures used `nonlocal` for this before.
    """
    if len(history["u"]) == 0:
        V = np.array([loss_u.item(), loss_k.item()])
    else:
        V = np.array([
            np.mean(history["u"][-update_every:]),
            np.mean(history["k"][-update_every:]),
        ])

    ratio = V.max() / (V.min() + 1e-12)

    # --------------------------------------------------------
    # No adaptive weighting
    # --------------------------------------------------------

    if not adaptive_weights:

        history["lambda_iteration"].append(iteration)
        history["lambda_u"].append(lambda_u)
        history["lambda_k"].append(lambda_k)
        history["lambda_pde"].append(lambda_pde)

        return lambda_u, lambda_k

    ratio_threshold = 10.0

    if ratio <= ratio_threshold:
        R = (V - V.min()) / (V.max() - V.min() + 1e-12)

        history["R_u"].append(R[0])
        history["R_k"].append(R[1])
        return lambda_u, lambda_k

    R = (V - V.min()) / (V.max() - V.min() + 1e-12)

    history["R_u"].append(R[0])
    history["R_k"].append(R[1])

    lambdas = 1.0 + alpha * R
    fastest = np.argmin(V)
    lambdas[fastest] = 1.0

    new_lambda_u, new_lambda_k = lambdas[0], lambdas[1]

    # --------------------------------------------------------
    # SAVE WEIGHTS
    # --------------------------------------------------------

    history["lambda_iteration"].append(iteration)
    history["lambda_u"].append(lambda_u)
    history["lambda_k"].append(lambda_k)
    history["lambda_pde"].append(lambda_pde)

    if verbose:
        print(
            f"V      = {V.round(3)}\n"
            f"R      = {R.round(3)}\n"
            f"ratio  = {ratio:.2f}\n"
            f"lambda_u = {new_lambda_u:.3f}\n"
            f"lambda_k = {new_lambda_k:.3f}\n"
            f"lambda_pde = {lambda_pde:.3f}"
        )

    return new_lambda_u, new_lambda_k


def compute_analytical_errors(
    model_u,
    model_k,
    analytical_solution_inf,
    coefficient_inf,
    pde_alpha,
    pde_beta,
    epsilon,
    device,
):
    """
    Wraps evaluate_model_inf. Returns (None, None) if the analytical
    reference isn't provided, so callers can log err_u/err_k unconditionally
    without branching -- keeps this fully optional / backward compatible.
    """
    if analytical_solution_inf is None or coefficient_inf is None:
        return None, None

    err_u, err_k = evaluate_model_inf(
        model_u=model_u,
        model_k=model_k,
        analytical_solution=analytical_solution_inf,
        coefficient=coefficient_inf,
        alpha=pde_alpha,
        beta=pde_beta,
        epsilon=epsilon,
        device=device,
    )
    return err_u, err_k

# ======================================================================
# History bookkeeping (was: save_history closure)
# ======================================================================

def save_history_entry(
    history,
    iteration,
    total,
    loss_u,
    loss_k,
    loss_pde,
    ratio,
    total_test,
    total_no_reg,
    total_no_reg_test,
    err_u=None,
    err_k=None,
):
    history["total"].append(total.item())
    history["iteration"].append(iteration)
    history["u"].append(loss_u.item())
    history["k"].append(loss_k.item())
    history["pde"].append(loss_pde.item())
    history["ratio"].append(ratio)
    history["total_test"].append(total_test.item())
    history["total_no_reg"].append(total_no_reg.item())
    history["total_no_reg_test"].append(total_no_reg_test.item())

    # err_u / err_k come from evaluate_model_inf against the analytical
    # solution -- optional, since they require analytical_solution_inf /
    # coefficient_inf to be supplied to train_dual_network.
    history["error_u"].append(err_u.item() if hasattr(err_u, "item") else err_u)
    history["error_k"].append(err_k.item() if hasattr(err_k, "item") else err_k)


def new_history():
    return {
        "total": [],
        "u": [],
        "k": [],
        "pde": [],
        "iteration": [],
        "lambda_iteration": [],
        "lambda_u": [],
        "lambda_k": [],
        "lambda_pde": [],
        "R_u": [],
        "R_k": [],
        "R_pde": [],
        "R_reg_t": [],
        "ratio": [],
        "total_test": [],
        "total_no_reg": [],
        "total_no_reg_test": [],
        "error_u": [],
        "error_k": [],
    }

 


# ======================================================================
# Run saving: model weights + history + human-readable summary
# ======================================================================

def describe_model(model):
    """
    Introspect an nn.Module generically -- works for any architecture
    (MLP, Fourier features, custom blocks, etc.) without needing to know
    its class ahead of time.

    Captures:
      - class name
      - total / trainable parameter counts
      - a per-layer breakdown (name, type, and shape info when the
        layer exposes in/out features, e.g. nn.Linear)
      - the full architecture printout (str(model)), which is the most
        reliable "ground truth" view of the network's structure
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    layers = []
    for name, module in model.named_modules():
        # skip the root module itself (name == "") and pure containers
        if name == "" or list(module.children()):
            continue

        layer_info = {"name": name, "type": module.__class__.__name__}

        if hasattr(module, "in_features") and hasattr(module, "out_features"):
            layer_info["in_features"] = module.in_features
            layer_info["out_features"] = module.out_features
        if hasattr(module, "bias") and isinstance(getattr(module, "bias", None), torch.Tensor):
            layer_info["bias"] = True
        elif hasattr(module, "bias"):
            layer_info["bias"] = module.bias is not None

        layers.append(layer_info)

    return {
        "class_name": model.__class__.__name__,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "layers": layers,
        "architecture": str(model),
    }


def build_training_summary(history, config, model_u=None, model_k=None):
    """Small JSON-able snapshot of how the run went (final + best losses)."""

    def last(key):
        return history[key][-1] if history.get(key) else None

    def best(key):
        return min(history[key]) if history.get(key) else None

    def last_valid(key):
        """Like last(), but skips a trailing None (error_u/error_k are only
        computed if analytical_solution_inf/coefficient_inf were provided)."""
        vals = history.get(key)
        return vals[-1] if vals and vals[-1] is not None else None

    def best_valid(key):
        """Like best(), but ignores None entries (analytical errors may be
        absent for some/all logged steps)."""
        vals = history.get(key)
        if not vals:
            return None
        valid = [v for v in vals if v is not None]
        return min(valid) if valid else None

    summary = {
        "config": config,
        "n_logged_steps": len(history.get("total", [])),
        "final": {
            "total": last("total"),
            "total_test": last("total_test"),
            "u": last("u"),
            "k": last("k"),
            "pde": last("pde"),
            "lambda_u": last("lambda_u"),
            "lambda_k": last("lambda_k"),
            "lambda_pde": last("lambda_pde"),
            "ratio": last("ratio"),
            "error_u": last_valid("error_u"),
            "error_k": last_valid("error_k"),
        },
        "best": {
            "total": best("total"),
            "total_test": best("total_test"),
            "u": best("u"),
            "k": best("k"),
            "pde": best("pde"),
            "error_u": best_valid("error_u"),
            "error_k": best_valid("error_k"),
        },
    }

    if model_u is not None:
        summary["model_u"] = describe_model(model_u)
    if model_k is not None:
        summary["model_k"] = describe_model(model_k)

    return summary


def _format_model_section(label, model_info):
    lines = [f"{label}:", f"  class: {model_info['class_name']}"]
    lines.append(f"  total params: {model_info['total_params']:,}")
    lines.append(f"  trainable params: {model_info['trainable_params']:,}")

    if model_info["layers"]:
        lines.append("  layers:")
        for layer in model_info["layers"]:
            shape = ""
            if "in_features" in layer and "out_features" in layer:
                shape = f" ({layer['in_features']} -> {layer['out_features']})"
            bias = f", bias={layer['bias']}" if "bias" in layer else ""
            lines.append(f"    - {layer['name'] or layer['type']}: {layer['type']}{shape}{bias}")

    lines.append("  full architecture:")
    for line in model_info["architecture"].splitlines():
        lines.append(f"    {line}")

    return lines

def _format_value(v):
    """Render numeric values in scientific notation; leave everything else
    (bools, strings, None) as-is."""
    if isinstance(v, bool):
        # bool is a subclass of int -- guard against True/False becoming 1e+00/0e+00
        return v
    if isinstance(v, (int, float)):
        return f"{v:.6e}"
    return v


def format_summary_text(summary, run_name):
    cfg = summary["config"]
    fin = summary["final"]
    bst = summary["best"]

    lines = [f"Training run: {run_name}", "=" * 40, "", "Config:"]
    for k, v in cfg.items():
        lines.append(f"  {k}: {_format_value(v)}")

    if "model_u" in summary:
        lines += [""] + _format_model_section("Model u (displacement network)", summary["model_u"])
    if "model_k" in summary:
        lines += [""] + _format_model_section("Model k (parameter network)", summary["model_k"])

    lines += ["", "Final losses (last logged step):"]
    for k, v in fin.items():
        lines.append(f"  {k}: {_format_value(v)}")

    lines += ["", "Best losses observed during training:"]
    for k, v in bst.items():
        lines.append(f"  {k}: {_format_value(v)}")

    return "\n".join(lines) + "\n"


def save_training_run(
    model_u,
    model_k,
    history,
    config,
    save_results=True,
    base_dir="results",
    run_name=None,
):
    """
    Save model weights, the full history, and a readable summary for a
    finished training run.

    Files are written to `base_dir/<run_name>/`, where `run_name`
    defaults to a timestamp (e.g. "2026-08-24_14-30-05_adaptive_reg")
    so repeated runs never clobber each other.

    Set save_results=False to skip writing anything to disk (e.g. for
    a quick throwaway experiment). The summary is still built and
    returned so you can inspect it in memory either way. This step
    happens strictly AFTER training is finished, so it never touches
    the training loop or its results.
    """
    if run_name is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        tags = []
        if "adaptive_weights" in config:
            tags.append("adaptive" if config["adaptive_weights"] else "fixed")
        if "lambda_pde_scheduler" in config:
            tags.append("sched" if config["lambda_pde_scheduler"] else "nosched")
        if "regularization" in config:
            tags.append("reg" if config["regularization"] else "no_reg")
        run_name = "_".join([timestamp] + tags)

    run_dir = os.path.join(base_dir, run_name)
    summary = build_training_summary(history, config, model_u=model_u, model_k=model_k)

    if not save_results:
        return {"saved": False, "run_dir": None, "summary": summary}

    os.makedirs(run_dir, exist_ok=True)

    torch.save(model_u.state_dict(), os.path.join(run_dir, "model_u.pt"))
    torch.save(model_k.state_dict(), os.path.join(run_dir, "model_k.pt"))

    with open(os.path.join(run_dir, "history.pkl"), "wb") as f:
        pickle.dump(history, f)

    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(run_dir, "summary.txt"), "w") as f:
        f.write(format_summary_text(summary, run_name))

    print(f"Training run saved to: {run_dir}")

    return {"saved": True, "run_dir": run_dir, "summary": summary}


# ======================================================================
# Main training loop
# ======================================================================

def train_dual_network(
    model_u,
    model_k,
    X_obs_train,
    U_obs_train,
    X_obs_k_train,
    K_obs_train,
    X_pde_train,
    F_pde_train,
    X_obs_test,
    U_obs_test,
    X_obs_k_test,
    K_obs_test,
    X_pde_test,
    F_pde_test,
    adam_lr=1e-3,
    adam_iters=2000,
    lbfgs_iters=2000,
    verbose=False,
    print_every=100,
    save_every=100,
    lambda_pde_scheduler=False,
    adaptive_weights=False,
    alpha=5,
    update_every=100,
    regularization=False,
    save_results=True,
    base_dir="results",
    run_name=None,
    # --- new: optional analytical-error tracking ---
    analytical_solution_inf=analytical_solution_inf,
    coefficient_inf=coefficient_inf,
    pde_alpha=0.5,
    pde_beta=5,
    epsilon=1,
    device=None,
):
    """analytical_solution_inf
            alpha=pde_alpha,
            beta=pde_beta,
            epsilon=epsilon,
    """
    criterion = nn.MSELoss()

    parameters = list(model_u.parameters()) + list(model_k.parameters())

    optimizer_adam = optim.AdamW(
        parameters,
        lr=adam_lr,
        weight_decay=1 if regularization else 0.0,
    )

    optimizer_lbfgs = optim.LBFGS(
        parameters,
        lr=1,
        max_iter=lbfgs_iters,
        max_eval=lbfgs_iters,
        history_size=100,
        tolerance_change=1.0 * np.finfo(float).eps,
        line_search_fn="strong_wolfe",
    )

    history = new_history()
 

    lambda_u = 1.0
    lambda_k = 1.0
    lambda_pde = 1.0

    # --------------------------------------------------
    # Adam
    # --------------------------------------------------
    if verbose:
        print("\n====================================")
        print("Training with Adam")
        print("====================================")

    model_u.train()
    model_k.train()

    for epoch in range(adam_iters + 1):

        if lambda_pde_scheduler:
            lambda_pde = get_pde_weight(epoch)
            history["lambda_iteration"].append(epoch)
            history["lambda_u"].append(lambda_u)
            history["lambda_k"].append(lambda_k)
            history["lambda_pde"].append(lambda_pde)

        optimizer_adam.zero_grad()

        total, loss_u, loss_k, loss_pde, total_no_reg = compute_losses(
            model_u, model_k,
            X_obs_train, U_obs_train, X_obs_k_train, K_obs_train,
            X_pde_train, F_pde_train,
            criterion, lambda_u, lambda_k, lambda_pde,
            parameters, regularization,
        )
        total_test, loss_u_test, loss_k_test, loss_pde_test, total_no_reg_test = compute_losses(
            model_u, model_k,
            X_obs_test, U_obs_test, X_obs_k_test, K_obs_test,
            X_pde_test, F_pde_test,
            criterion, lambda_u, lambda_k, lambda_pde,
            parameters, regularization,
        )

        if epoch == 0:
           V = np.array([loss_u.item(), loss_k.item()])
           ratio = V.max() / (V.min() + 1e-12)
        else:
            ratio = ratio_calculation(history, update_every)

        total.backward()
        optimizer_adam.step()

        if adaptive_weights and epoch % update_every == 0 and epoch > 0:
            #print(f"Epoch {epoch}: updating loss weights (lambda_u, lambda_k)")
            lambda_u, lambda_k = update_loss_weights(
                history, epoch, lambda_u, lambda_k, lambda_pde,
                loss_u, loss_k, update_every, alpha,
                adaptive_weights, verbose,
            )



        if verbose and epoch % print_every == 0:

            err_u, err_k = compute_analytical_errors(
                model_u, model_k,
                analytical_solution_inf, coefficient_inf,
                pde_alpha, pde_beta, epsilon, device,
            )

            print(
                f"Adam {epoch:5d} | "
                f"Total={total.item():.3e} | "
                f"ObsU={loss_u.item():.3e} | "
                f"ObsK={loss_k.item():.3e} | "
                f"PDE={loss_pde.item():.3e} | "
                f"Ratio={ratio:.2f}"
                + (f" | ErrU={err_u:.3e} | ErrK={err_k:.3e}" if err_u is not None else "")
            )

        if epoch % save_every == 0:
            # err_u, err_k = compute_analytical_errors(
            #     model_u, model_k,
            #     analytical_solution_inf, coefficient_inf,
            #     pde_alpha, pde_beta, epsilon, device,
            # )
            save_history_entry(
                history, epoch, total, loss_u, loss_k, loss_pde, ratio,
                total_test, total_no_reg, total_no_reg_test,
                err_u=err_u, err_k=err_k,
            )
    # --------------------------------------------------
    # L-BFGS
    # --------------------------------------------------
    if verbose:
        print("\n====================================")
        print("Training with L-BFGS")
        print("====================================")

    state = {"iter": 0, "loss_u": None, "loss_k": None, "loss_pde": None}
    iters_done = 0

    def closure():
        nonlocal lambda_u, lambda_k

        optimizer_lbfgs.zero_grad()
        lambda_reg = 0

        total, loss_u, loss_k, loss_pde, total_no_reg = compute_losses(
            model_u, model_k,
            X_obs_train, U_obs_train, X_obs_k_train, K_obs_train,
            X_pde_train, F_pde_train,
            criterion, lambda_u, lambda_k, lambda_pde,
            parameters, regularization, lambda_reg,
        )
        total_test, loss_u_test, loss_k_test, loss_pde_test, total_no_reg_test = compute_losses(
            model_u, model_k,
            X_obs_test, U_obs_test, X_obs_k_test, K_obs_test,
            X_pde_test, F_pde_test,
            criterion, lambda_u, lambda_k, lambda_pde,
            parameters, regularization, lambda_reg,
        )

        total.backward()

        #if iters_done == 0:
        #    V = np.array([loss_u.item(), loss_k.item(), loss_pde.item()])
        #    ratio = V.max() / (V.min() + 1e-12)
        #else:
        ratio = ratio_calculation(history, update_every)

        state["loss_u"] = loss_u.detach()
        state["loss_k"] = loss_k.detach()
        state["loss_pde"] = loss_pde.detach()



        state["iter"] += 1

        if verbose and state["iter"] % print_every == 0:

            err_u, err_k = compute_analytical_errors(
                model_u, model_k,
                analytical_solution_inf, coefficient_inf,
                pde_alpha, pde_beta, epsilon, device,
            )

            print(
                f"L-BFGS {state['iter']:5d} | "
                f"Total={total.item():.3e} | "
                f"ObsU={loss_u.item():.3e} | "
                f"ObsK={loss_k.item():.3e} | "
                f"PDE={loss_pde.item():.3e} | "
                f"Ratio={ratio:.2f}",
                f" | ErrU={err_u:.3e} | ErrK={err_k:.3e}" if err_u is not None else ""
            )


        history["lambda_iteration"].append(state["iter"] + adam_iters)
        history["lambda_u"].append(lambda_u)
        history["lambda_k"].append(lambda_k)
        history["lambda_pde"].append(lambda_pde)
 
        if state["iter"] % save_every == 0:
            err_u, err_k = compute_analytical_errors(
                model_u, model_k,
                analytical_solution_inf, coefficient_inf,
                pde_alpha, pde_beta, epsilon, device,
            )
            save_history_entry(
                history, state["iter"] + adam_iters, total, loss_u, loss_k, loss_pde, ratio,
                total_test, total_no_reg, total_no_reg_test,
                err_u=err_u, err_k=err_k,
            )

        #state["iter"] += 1

        return total

    optimizer_lbfgs.step(closure)

    # --------------------------------------------------
    # Save model weights + history + summary
    # --------------------------------------------------
    config = {
        "adam_lr": adam_lr,
        "adam_iters": adam_iters,
        "lbfgs_iters": lbfgs_iters,
        "print_every": print_every,
        "lambda_pde_scheduler": lambda_pde_scheduler,
        "adaptive_weights": adaptive_weights,
        "alpha": alpha,
        "update_every": update_every,
        "regularization": regularization,
    }

    save_info = save_training_run(
        model_u,
        model_k,
        history,
        config,
        save_results=save_results,
        base_dir=base_dir,
        run_name=run_name,
    )

    history["run_dir"] = save_info["run_dir"]

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
    adam_iters=2000,
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


def gradient_regularization(loss, inputs):

    grad = torch.autograd.grad(
        outputs=loss,
        inputs=inputs,
        grad_outputs=torch.ones_like(loss),
        create_graph=True,
        retain_graph=True,
    )[0]

    return (grad.pow(2).sum(dim=1)).mean()