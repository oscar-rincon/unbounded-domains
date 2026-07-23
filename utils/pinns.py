
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
    grid_size=3,
    spline_order=3,
):
    model_u = KAN(
        layers_hidden=[2] + [hidden_units] * hidden_layers + [1],
        grid_size=grid_size,
        spline_order=spline_order,
        grid_range=[-5,5],
        base_activation=torch.nn.Tanh
    ).to(device)

    model_k = KAN(
        layers_hidden=[2] + [hidden_units] * hidden_layers + [1],
        grid_size=grid_size,
        spline_order=spline_order,
        grid_range=[-5,5],
        base_activation=torch.nn.Tanh
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
    adam_iters=1000,
    lbfgs_iters=2000,
    verbose=False,
    print_every=100, 
    adaptive_weights=True,
    alpha=100,
    update_every=100, 
    regularization=False,    
):
    ratio = 1
    criterion = nn.MSELoss()

    parameters = (
        list(model_u.parameters())
        + list(model_k.parameters())
    )

    optimizer_adam = optim.AdamW(
        parameters,
        lr=adam_lr ,
        weight_decay=1 if regularization else 0.0
    )

    optimizer_lbfgs = optim.LBFGS(
        parameters,
        lr=1,
        max_iter=lbfgs_iters,
        max_eval=lbfgs_iters,
        history_size=100,
        tolerance_change=1.0 * np.finfo(float).eps,
        line_search_fn="strong_wolfe")

 

    history = {
        "total": [],
        "u": [],
        "k": [],
        "pde": [],

        "lambda_u": [],
        "lambda_k": [],
        "lambda_pde": [],

        # New entries
        "R_u": [],
        "R_k": [],
        "R_pde": [],
        "R_reg_t": [],
        "ratio": [],
        "total_test": [],
        "total_no_reg": [],
        "total_no_reg_test": [],
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

    def compute_losses(lambda_reg=1.0):

        loss_u = observation_loss_u(
            model_u,
            X_obs_train,
            U_obs_train,
            criterion,
        )

        loss_k = observation_loss_k(
            model_k,
            X_obs_k_train,
            K_obs_train,
            criterion,
        )

        loss_pde = pde_loss_inf(
            model_u,
            model_k,
            X_pde_train,
            F_pde_train,
        )

        loss_u = torch.nan_to_num(loss_u)
        loss_k = torch.nan_to_num(loss_k)
        loss_pde = torch.nan_to_num(loss_pde)


        if regularization:
            reg_t = l2_regularization(parameters)
            
  
        total = (
            lambda_u * loss_u
            + lambda_k * loss_k
            + lambda_pde * loss_pde
            + (lambda_reg * reg_t if regularization else 0.0)
        )

        total_no_reg = (
            lambda_u * loss_u
            + lambda_k * loss_k
            + lambda_pde * loss_pde
        )

        return total, loss_u, loss_k, loss_pde, total_no_reg

    def compute_losses_test(lambda_reg=1.0):

        loss_u = observation_loss_u(
            model_u,
            X_obs_test,
            U_obs_test,
            criterion,
        )

        loss_k = observation_loss_k(
            model_k,
            X_obs_k_test,
            K_obs_test,
            criterion,
        )

        loss_pde = pde_loss_inf(
            model_u,
            model_k,
            X_pde_test,
            F_pde_test,
        )

        loss_u = torch.nan_to_num(loss_u)
        loss_k = torch.nan_to_num(loss_k)
        loss_pde = torch.nan_to_num(loss_pde)

        if regularization:
            reg_t = l2_regularization(parameters)
 
        total = (
            lambda_u * loss_u
            + lambda_k * loss_k
            + lambda_pde * loss_pde
            + (lambda_reg * reg_t if regularization else 0.0)
        )

        total_no_reg = (
            lambda_u * loss_u
            + lambda_k * loss_k
            + lambda_pde * loss_pde
        )

        return total, loss_u, loss_k, loss_pde, total_no_reg

 
    def ratio_calculation():
        V = np.array([
            np.mean(history["u"][-update_every:]),
            np.mean(history["k"][-update_every:]),
            np.mean(history["pde"][-update_every:]),
            #np.mean(history["reg_t"][-update_every:])
        ])

        # ----------------------------------------------------
        # Step 7: Ratio
        # ----------------------------------------------------

        ratio = V.max() / (V.min() + 1e-12)
        return ratio

    def update_loss_weights(loss_u, loss_k, loss_pde):

        nonlocal lambda_u, lambda_k, lambda_pde

        # ----------------------------------------------------
        # Step 6: Current losses or moving average
        # ----------------------------------------------------

        if len(history["u"]) == 0:

            V = np.array([
                loss_u.item(),
                loss_k.item(),
                loss_pde.item(),
            ])

        else:

            V = np.array([
                np.mean(history["u"][-update_every:]),
                np.mean(history["k"][-update_every:]),
                np.mean(history["pde"][-update_every:]),
            ])

        # ----------------------------------------------------
        # Step 7: Ratio
        # ----------------------------------------------------

        ratio = V.max() / (V.min() + 1e-12)

        if not adaptive_weights:
            return

        ratio_threshold = 10

        if ratio <= ratio_threshold:

            history["R_u"].append(0.0)
            history["R_k"].append(0.0)
            history["R_pde"].append(0.0)
            return

        # ----------------------------------------------------
        # Step 8: Compute R
        # ----------------------------------------------------

        R = (V - V.min()) / (V.max() - V.min() + 1e-12)

        history["R_u"].append(R[0])
        history["R_k"].append(R[1])
        history["R_pde"].append(R[2])

        # ----------------------------------------------------
        # Step 9: Update lambdas
        # ----------------------------------------------------

        lambdas = 1.0 + alpha * R

        fastest = np.argmin(V)
        lambdas[fastest] = 1.0

        lambda_u = lambdas[0]
        lambda_k = lambdas[1]
        lambda_pde = lambdas[2]

        if verbose:
            print(
                f"V      = {V.round(3)}\n"
                f"R      = {R.round(3)}\n"
                f"ratio  = {ratio:.2f}\n"
                f"lambda = {lambdas.round(3)}"
            )
        #lambda_reg_t = lambdas[3]
        #if verbose:

            #print(
            #    f"V      = {V.round(4)}\n"
            #    f"R      = {R.round(3)}\n"
            #    f"ratio  = {ratio:.2f}\n"
            #    f"lambda = {lambdas.round(3)}"
            #)
        
    # --------------------------------------------------
    # Helper: save history
    # --------------------------------------------------

    def save_history(total, loss_u, loss_k, loss_pde, ratio, total_test, total_no_reg,total_no_reg_test):

        history["total"].append(total.item())
        history["u"].append(loss_u.item())
        history["k"].append(loss_k.item())
        history["pde"].append(loss_pde.item())
        #history["reg_t"].append(reg_t.item())
        history["lambda_u"].append(lambda_u)
        history["lambda_k"].append(lambda_k)
        history["lambda_pde"].append(lambda_pde)
        #history["lambda_reg_t"].append(lambda_reg_t)
        history["ratio"].append(ratio)
        history["total_test"].append(total_test.item())
        history["total_no_reg"].append(total_no_reg.item())
        history["total_no_reg_test"].append(total_no_reg_test.item())

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

        total_no_reg_test = torch.tensor(0.0)  # Initialize total_no_reg_test to avoid undefined variable error
        total_test = torch.tensor(0.0)  # Initialize total_test to avoid undefined
        lambda_reg=0

        total, loss_u, loss_k, loss_pde, total_no_reg = compute_losses(lambda_reg)
        #total_test, loss_u_test, loss_k_test, loss_pde_test, total_no_reg_test = compute_losses_test(lambda_reg)

        if epoch == 0:

            V = np.array([
                loss_u.item(),
                loss_k.item(),
                loss_pde.item(),
                #reg_t.item()
            ])

            ratio = V.max() / (V.min() + 1e-12)
            update_loss_weights(loss_u, loss_k, loss_pde) 
            total, loss_u, loss_k, loss_pde, total_no_reg = compute_losses(lambda_reg)
            #total_test, loss_u_test, loss_k_test, loss_pde_test, total_no_reg_test = compute_losses_test(lambda_reg)

        else:
            ratio = ratio_calculation()
        

        total.backward()
        optimizer_adam.step()

        #expect in zero 
        if (epoch) % update_every == 0 and epoch > 0:
            update_loss_weights(loss_u, loss_k, loss_pde)



        save_history(total, loss_u, loss_k, loss_pde, ratio, total_test, total_no_reg, total_no_reg_test)

        if verbose and epoch % print_every == 0:

            print(
                f"Adam {epoch:5d} | "
                f"Total={total.item():.3e} | "
                f"ObsU={loss_u.item():.3e} | "
                f"ObsK={loss_k.item():.3e} | "
                f"PDE={loss_pde.item():.3e} | "
                f"Ratio={ratio:.2f}"
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


        lambda_reg=1e-4

        total, loss_u, loss_k, loss_pde, total_no_reg = compute_losses(lambda_reg)
        #total_test, loss_u_test, loss_k_test, loss_pde_test, total_no_reg_test = compute_losses_test(lambda_reg)

        total.backward()

 

        if epoch == 0:

            V = np.array([
                loss_u.item(),
                loss_k.item(),
                loss_pde.item(),
                #reg_t.item()
            ])

            ratio = V.max() / (V.min() + 1e-12)
             
        else:
            ratio = ratio_calculation()

        #if (state["iter"]) % update_every == 0 and state["iter"] > 0:
        #    update_loss_weights(loss_u, loss_k, loss_pde)
  
        save_history(total, loss_u, loss_k, loss_pde, ratio, total_test, total_no_reg, total_no_reg_test)
    
        state["iter"] += 1


        if verbose and state["iter"] % print_every == 0:

            print(
                f"L-BFGS {state['iter']:5d} | "
                f"Total={total.item():.3e} | "
                f"ObsU={loss_u.item():.3e} | "
                f"ObsK={loss_k.item():.3e} | "
                f"PDE={loss_pde.item():.3e} | "
                f"Ratio={ratio:.2f}"
            )


        return total

    optimizer_lbfgs.step(closure)

    # --------------------------------------------------
    # Save history automatically
    # --------------------------------------------------

    os.makedirs("results", exist_ok=True)

    filename = "history"

    if adaptive_weights:
        filename += "_adaptive"
    else:
        filename += "_fixed"

    if regularization:
        filename += "_reg"
    else:
        filename += "_no_reg"

    filename += ".pkl"

    with open(os.path.join("results", filename), "wb") as f:
        pickle.dump(history, f)

    #print(f"History saved to results/{filename}")

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


def gradient_regularization(loss, inputs):

    grad = torch.autograd.grad(
        outputs=loss,
        inputs=inputs,
        grad_outputs=torch.ones_like(loss),
        create_graph=True,
        retain_graph=True,
    )[0]

    return (grad.pow(2).sum(dim=1)).mean()