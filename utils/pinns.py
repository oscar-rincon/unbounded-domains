
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch 
import torch.nn as nn
import time
from matplotlib.patches import Rectangle
from functools import partial   
from matplotlib.gridspec import GridSpec
#gaussian_kde
from scipy.stats import gaussian_kde

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
      