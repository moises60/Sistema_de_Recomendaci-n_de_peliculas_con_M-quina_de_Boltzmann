# utils/evaluation.py

import torch

def compute_loss(v0, vk, num_visible_ratings):
    """
    Calcula el Error Absoluto Medio (MAE) entre las valoraciones originales y reconstruidas.

    Args:
        v0 (torch.Tensor): Valoraciones originales.
        vk (torch.Tensor): Valoraciones reconstruidas.
        num_visible_ratings (int): Número de columnas correspondientes a las valoraciones.

    Returns:
        loss_mae (float): Error Absoluto Medio.
    """
    mask = v0[:, :num_visible_ratings] >= 0  # Solo considerar las valoraciones
    if mask.sum() > 0:
        loss_mae = torch.mean(torch.abs(v0[:, :num_visible_ratings][mask] - vk[:, :num_visible_ratings][mask])).item()
    else:
        loss_mae = 0
    return loss_mae
