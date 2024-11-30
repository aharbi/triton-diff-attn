import math
import torch

from flash_attn import flash_attn_func
from apex.normalization import FusedRMSNorm as RMSNorm

from kernel import _diff_attention


def MultiheadDiffAttnKernel(
    q1: torch.tensor,
    q2: torch.tensor,
    k1: torch.tensor,
    k2: torch.tensor,
    v: torch.tensor,
    lambda_scale: float = -0.5,
    lambda_init: float = 0.8,
    rms_norm: bool = False,
):
    _, _, N, head_dim = q1.size()

    sm_scale = 1 / math.sqrt(head_dim)

    return _diff_attention.apply(
        q1, q2, k1, k2, v, sm_scale, lambda_scale, lambda_init, rms_norm
    )


def MultiheadAttn(
    q: torch.tensor, k: torch.tensor, v: torch.tensor, rms_norm: bool = False
):

    _, _, N, head_dim = q.size()

    sm_scale = 1 / math.sqrt(head_dim)

    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    p = torch.softmax(p.float(), dim=-1).half()

    attn = torch.matmul(p, v)

    if rms_norm:
        norm = RMSNorm(normalized_shape=head_dim, elementwise_affine=False)
        attn = norm(attn)

    return attn


def MultiheadDiffAttn(
    q1: torch.tensor,
    q2: torch.tensor,
    k1: torch.tensor,
    k2: torch.tensor,
    v: torch.tensor,
    lambda_scale: float = -0.5,
    lambda_init: float = 0.8,
    rms_norm: bool = False,
):

    _, _, N, head_dim = q1.size()

    sm_scale = 1 / math.sqrt(head_dim)

    p1 = torch.matmul(q1, k1.transpose(2, 3)) * sm_scale
    p1 = torch.softmax(p1.float(), dim=-1).half()

    p2 = torch.matmul(q2, k2.transpose(2, 3)) * sm_scale
    p2 = torch.softmax(p2.float(), dim=-1).half()

    p = p1 - lambda_scale * p2

    attn = torch.matmul(p, v)

    if rms_norm:
        norm = RMSNorm(normalized_shape=2 * head_dim, elementwise_affine=False)
        attn = norm(attn)

        attn = (1 - lambda_init) * attn

    return attn


def MultiheadFlashAttn(
    q: torch.tensor, k: torch.tensor, v: torch.tensor, rms_norm: bool = False
):

    _, _, N, head_dim = q.size()

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    attn = flash_attn_func(q, k, v, causal=False)

    attn = attn.transpose(1, 2)

    if rms_norm:
        norm = RMSNorm(normalized_shape=head_dim, elementwise_affine=False)
        attn = norm(attn)

    return attn


def MultiheadFlashDiffAttn(
    q1: torch.tensor,
    q2: torch.tensor,
    k1: torch.tensor,
    k2: torch.tensor,
    v: torch.tensor,
    lambda_scale: float = -0.5,
    lambda_init: float = 0.8,
    rms_norm: bool = False,
):
    # Code adapted from: https://github.com/microsoft/unilm/tree/master/Diff-Transformer

    B, num_heads, N, head_dim = q1.size()

    v = v.view(B, num_heads, N, 2, head_dim)

    v1, v2 = v[:, :, :, 0], v[:, :, :, 1]

    q1 = q1.transpose(1, 2)
    q2 = q2.transpose(1, 2)

    k1 = k1.transpose(1, 2)
    k2 = k2.transpose(1, 2)

    v1 = v1.transpose(1, 2)
    v2 = v2.transpose(1, 2)

    attn11 = flash_attn_func(q1, k1, v1, causal=False)
    attn12 = flash_attn_func(q1, k1, v2, causal=False)
    attn1 = torch.cat([attn11, attn12], dim=-1)

    attn21 = flash_attn_func(q2, k2, v1, causal=False)
    attn22 = flash_attn_func(q2, k2, v2, causal=False)
    attn2 = torch.cat([attn21, attn22], dim=-1)

    attn = attn1 - lambda_scale * attn2

    attn = attn.transpose(1, 2)

    if rms_norm:
        norm = RMSNorm(normalized_shape=2 * head_dim, elementwise_affine=False)
        attn = norm(attn)
        attn = (1 - lambda_init) * attn

    return attn
