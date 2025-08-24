from jaxtyping import Float
import torch
import torch.nn as nn


class cElegansFwdSAE(nn.Module):
    def __init__(self, hidden_dim: int = 189):
        super().__init__()
        self.encode = nn.Linear(189, hidden_dim)

    def forward(self, x: Float[torch.Tensor, "b neurons"]):
        x = self.encode(x)
        return x


class cElegansBwdSAE(nn.Module):
    def __init__(self, hidden_dim: int = 189):
        super().__init__()
        self.decode = nn.Linear(hidden_dim, 189)

    def forward(self, x: Float[torch.Tensor, "b neurons"]):
        x = self.decode(x)
        return x


class cElegansSAE(nn.Module):
    def __init__(self, hidden_dim: int = 378):
        super().__init__()
        self.encode = nn.Linear(189, hidden_dim)
        self.decode = nn.Linear(hidden_dim, 189)
        self.nonlinearity = nn.ReLU()

    def forward(self, x: Float[torch.Tensor, "b neurons"]):
        x = self.encode(x)
        x = self.nonlinearity(x)
        x = self.decode(x)
        return x
