# models/rbm.py

import torch
import torch.nn as nn

class RBM(nn.Module):
    """
    Máquina de Boltzmann Restringida (RBM) para sistemas de recomendación.

    Args:
        nv (int): Número de unidades visibles (valoraciones + características).
        nh (int): Número de unidades ocultas.
        device (torch.device): Dispositivo (CPU o CUDA).
        lr (float): Tasa de aprendizaje.

    Attributes:
        W (nn.Parameter): Matriz de pesos entre unidades visibles y ocultas.
        a (nn.Parameter): Sesgos de las unidades ocultas.
        b (nn.Parameter): Sesgos de las unidades visibles.
    """

    def __init__(self, nv, nh, device, lr=0.1):
        super(RBM, self).__init__()
        self.nv = nv  # Unidades visibles
        self.nh = nh  # Unidades ocultas
        self.device = device
        self.lr = lr

        # Inicializar parámetros
        self.W = nn.Parameter(torch.randn(nh, nv, device=device) * 0.01)
        self.a = nn.Parameter(torch.zeros(1, nh, device=device))
        self.b = nn.Parameter(torch.zeros(1, nv, device=device))

    def sample_h(self, x):
        """
        Muestrea las unidades ocultas dadas las visibles.

        Args:
            x (torch.Tensor): Unidades visibles.

        Returns:
            p_h_given_v (torch.Tensor): Probabilidad de activación de las unidades ocultas.
            h_sample (torch.Tensor): Muestras binarias de las unidades ocultas.
        """
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a
        p_h_given_v = torch.sigmoid(activation)
        h_sample = torch.bernoulli(p_h_given_v)
        return p_h_given_v, h_sample

    def sample_v(self, y):
        """
        Muestrea las unidades visibles dadas las ocultas.

        Args:
            y (torch.Tensor): Unidades ocultas.

        Returns:
            p_v_given_h (torch.Tensor): Probabilidad de activación de las unidades visibles.
            v_sample (torch.Tensor): Muestras binarias de las unidades visibles.
        """
        wy = torch.mm(y, self.W)
        activation = wy + self.b
        p_v_given_h = torch.sigmoid(activation)
        v_sample = torch.bernoulli(p_v_given_h)
        return p_v_given_h, v_sample

    def train_rbm(self, v0, vk, ph0, phk):
        """
        Realiza una actualización de los pesos y sesgos utilizando Contrastive Divergence.

        Args:
            v0 (torch.Tensor): Entrada original.
            vk (torch.Tensor): Entrada reconstruida después de k pasos de Gibbs.
            ph0 (torch.Tensor): Probabilidades de activación iniciales de las unidades ocultas.
            phk (torch.Tensor): Probabilidades de activación finales de las unidades ocultas.
        """
        positive_grad = torch.mm(ph0.t(), v0)
        negative_grad = torch.mm(phk.t(), vk)

        # Gradientes
        self.W.grad = -(positive_grad - negative_grad) / v0.size(0)
        self.b.grad = -torch.mean(v0 - vk, dim=0, keepdim=True)
        self.a.grad = -torch.mean(ph0 - phk, dim=0, keepdim=True)
