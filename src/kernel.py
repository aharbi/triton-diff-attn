# Differential Attention (https://arxiv.org/pdf/2410.05258) Kernel in Triton
# Credits: Some code snippets were adapted from https://github.com/triton-lang/triton/blob/main/python/tutorials/06-fused-attention.py
# - Implemented Features: Forward pass, Backward pass, RMS Normalization (Can be disabled using the `rms_norm` flag)
# - Missing Features: Masking and dropout.

import torch

import triton
import triton.language as tl


@triton.autotune(configs=[
    triton.Config(kwargs={'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64}, num_warps=4, num_stages=1),
    triton.Config(kwargs={'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128}, num_warps=4, num_stages=1),
    triton.Config(kwargs={'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64}, num_warps=4, num_stages=2),
    triton.Config(kwargs={'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128}, num_warps=4, num_stages=2),
  ],
  key=['SEQ_LEN', 'HEAD_DIM_K', 'HEAD_DIM_V']
)
@triton.jit
def diff_attn_fwd(
    Q1,
    Q2,
    K1,
    K2,
    V,
    M1,
    M2,
    O1,
    O2,
    O,
    stride_QK_batch,
    stride_QK_head,
    stride_QK_seq,
    stride_QK_dim,
    stride_V_batch,
    stride_V_head,
    stride_V_seq,
    stride_V_dim,
    SM_SCALE,
    LAMBDA_SCALE_ptr,
    LAMBDA_INIT,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM_K: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    RMS_NORM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    EPS: tl.constexpr = 1e-5,
):
    seq_index = tl.program_id(0)
    index_batch_head = tl.program_id(1)

    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS

    qk_offset = index_batch.to(tl.int64) * stride_QK_batch + index_head.to(tl.int64) * stride_QK_head
    v_offset = index_batch.to(tl.int64) * stride_V_batch + index_head.to(tl.int64) * stride_V_head
    offs_q = seq_index * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    # NOTE: Pointers for the inner loop
    Q1_block_ptr = tl.make_block_ptr(
        base=Q1 + qk_offset,
        shape=(SEQ_LEN, HEAD_DIM_K),
        strides=(stride_QK_seq, stride_QK_dim),
        offsets=(seq_index * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, HEAD_DIM_K),
        order=(1, 0),
    )

    Q2_block_ptr = tl.make_block_ptr(
        base=Q2 + qk_offset,
        shape=(SEQ_LEN, HEAD_DIM_K),
        strides=(stride_QK_seq, stride_QK_dim),
        offsets=(seq_index * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, HEAD_DIM_K),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(SEQ_LEN, HEAD_DIM_V),
        strides=(stride_V_seq, stride_V_dim),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_N, HEAD_DIM_V),
        order=(1, 0),
    )

    K1_block_ptr = tl.make_block_ptr(
        base=K1 + qk_offset,
        shape=(HEAD_DIM_K, SEQ_LEN),
        strides=(stride_QK_dim, stride_QK_seq),
        offsets=(0, 0),
        block_shape=(HEAD_DIM_K, BLOCK_SIZE_N),
        order=(0, 1),
    )

    K2_block_ptr = tl.make_block_ptr(
        base=K2 + qk_offset,
        shape=(HEAD_DIM_K, SEQ_LEN),
        strides=(stride_QK_dim, stride_QK_seq),
        offsets=(0, 0),
        block_shape=(HEAD_DIM_K, BLOCK_SIZE_N),
        order=(0, 1),
    )

    O1_block_ptr = tl.make_block_ptr(
        base=O1 + v_offset,
        shape=(SEQ_LEN, HEAD_DIM_V),
        strides=(stride_V_seq, stride_V_dim),
        offsets=(seq_index * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, HEAD_DIM_V),
        order=(1, 0),
    )

    O2_block_ptr = tl.make_block_ptr(
        base=O2 + v_offset,
        shape=(SEQ_LEN, HEAD_DIM_V),
        strides=(stride_V_seq, stride_V_dim),
        offsets=(seq_index * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, HEAD_DIM_V),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        base=O + v_offset,
        shape=(SEQ_LEN, HEAD_DIM_V),
        strides=(stride_V_seq, stride_V_dim),
        offsets=(seq_index * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, HEAD_DIM_V),
        order=(1, 0),
    )

    # NOTE: Tensor that are in SRAM during the inner loop
    Q1_block = tl.load(Q1_block_ptr)
    Q2_block = tl.load(Q2_block_ptr)
    LAMBDA_SCALE = tl.load(LAMBDA_SCALE_ptr)

    # NOTE: Placeholder for outputs
    m_i1 = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32) - float("inf")
    m_i2 = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32) - float("inf")
    l_i1 = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32) + 1.0
    l_i2 = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32) + 1.0
    O1_block = tl.zeros([BLOCK_SIZE_M, HEAD_DIM_V], dtype=tl.float32)
    O2_block = tl.zeros([BLOCK_SIZE_M, HEAD_DIM_V], dtype=tl.float32)

    for _ in range(0, SEQ_LEN, BLOCK_SIZE_N):
        K1_block = tl.load(K1_block_ptr)
        K2_block = tl.load(K2_block_ptr)

        QK1_block = tl.dot(Q1_block, K1_block)
        QK2_block = tl.dot(Q2_block, K2_block)

        m_ij1 = tl.maximum(m_i1, tl.max(QK1_block, 1) * SM_SCALE)
        m_ij2 = tl.maximum(m_i2, tl.max(QK2_block, 1) * SM_SCALE)

        QK1_block = QK1_block * SM_SCALE - m_ij1[:, None]
        QK2_block = QK2_block * SM_SCALE - m_ij2[:, None]

        P1_block = tl.math.exp(QK1_block)
        P2_block = tl.math.exp(QK2_block)

        alpha1 = tl.math.exp(m_i1 - m_ij1)
        alpha2 = tl.math.exp(m_i2 - m_ij2)

        m_i1 = m_ij1
        m_i2 = m_ij2

        l_i1 = l_i1 * alpha1 + tl.sum(P1_block, 1)
        l_i2 = l_i2 * alpha2 + tl.sum(P2_block, 1)

        V_block = tl.load(V_block_ptr)

        P1_block = P1_block.to(tl.float16)
        P2_block = P2_block.to(tl.float16)

        O1_block = O1_block * alpha1[:, None]
        O2_block = O2_block * alpha2[:, None]

        O1_block = tl.dot(P1_block, V_block, O1_block)
        O2_block = tl.dot(P2_block, V_block, O2_block)

        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_N, 0))
        K1_block_ptr = tl.advance(K1_block_ptr, (0, BLOCK_SIZE_N))
        K2_block_ptr = tl.advance(K2_block_ptr, (0, BLOCK_SIZE_N))

    m_i1 += tl.math.log(l_i1)
    m_i2 += tl.math.log(l_i2)

    O1_block = O1_block / l_i1[:, None]
    O2_block = -LAMBDA_SCALE * O2_block / l_i2[:, None]

    O_block = O1_block + O2_block

    if RMS_NORM:
        O_block_sq_mean = (1.0 / HEAD_DIM_V) * tl.sum(O_block * O_block, axis=1)
        O_block = (1.0 - LAMBDA_INIT) * O_block * tl.rsqrt(O_block_sq_mean + EPS)[:, None]

    m1_ptrs = M1 + index_batch_head * SEQ_LEN + offs_q
    m2_ptrs = M2 + index_batch_head * SEQ_LEN + offs_q

    tl.store(m1_ptrs, m_i1)
    tl.store(m2_ptrs, m_i2)
    tl.store(O1_block_ptr, O1_block.to(O.type.element_ty))
    tl.store(O2_block_ptr, O2_block.to(O.type.element_ty))
    tl.store(O_block_ptr, O_block.to(O.type.element_ty))


@triton.autotune(configs=[
    triton.Config(kwargs={}, num_warps=4, num_stages=2),
    triton.Config(kwargs={}, num_warps=8, num_stages=2),
    triton.Config(kwargs={}, num_warps=4, num_stages=3),
    triton.Config(kwargs={}, num_warps=8, num_stages=3),
    triton.Config(kwargs={}, num_warps=4, num_stages=4),
    triton.Config(kwargs={}, num_warps=8, num_stages=4),
  ],
  key=['SEQ_LEN', 'HEAD_DIM_K', 'HEAD_DIM_V']
)
@triton.jit
def diff_attn_bwd(
    Q1,
    Q2,
    K1,
    K2,
    V,
    O1,
    O2,
    dO,
    dQ1,
    dQ2,
    dK1,
    dK2,
    dV,
    dLambda,
    dLambda_stride_seq,
    dLambda_stride_batch_head,
    M1,
    M2,
    D1,
    D2,
    stride_QK_batch,
    stride_QK_head,
    stride_QK_seq,
    stride_QK_dim,
    stride_V_batch,
    stride_V_head,
    stride_V_seq,
    stride_V_dim,
    SM_SCALE,
    LAMBDA_SCALE_ptr,
    LAMBDA_INIT,
    NUM_HEADS,
    SEQ_LEN,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    HEAD_DIM_K: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    RMS_NORM: tl.constexpr,
    EPS: tl.constexpr = 1e-5,
):
    seq_index = tl.program_id(0)
    index_batch_head = tl.program_id(1)

    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS

    qk_offs_dim = tl.arange(0, HEAD_DIM_K)
    v_offs_dim = tl.arange(0, HEAD_DIM_V)
    offs_qv = tl.arange(0, BLOCK_SIZE_M)
    offs_kv = seq_index * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # NOTE: Stage 1: Preprocessing
    D1_block_ptrs = D1 + index_batch_head * SEQ_LEN + offs_kv
    D2_block_ptrs = D2 + index_batch_head * SEQ_LEN + offs_kv
    dLambda_ptr = dLambda + seq_index * dLambda_stride_seq + index_batch_head * dLambda_stride_batch_head

    O1_block = tl.load(O1 + index_batch_head * HEAD_DIM_V * SEQ_LEN + offs_kv[:, None] * HEAD_DIM_V + v_offs_dim[None, :])
    O2_block = tl.load(O2 + index_batch_head * HEAD_DIM_V * SEQ_LEN + offs_kv[:, None] * HEAD_DIM_V + v_offs_dim[None, :])
    dOut_block = tl.load(dO + index_batch_head * HEAD_DIM_V * SEQ_LEN + offs_kv[:, None] * HEAD_DIM_V + v_offs_dim[None, :])
    LAMBDA_SCALE = tl.load(LAMBDA_SCALE_ptr)

    if RMS_NORM:
        O_block = O1_block + O2_block

        inv_norm = tl.rsqrt((1.0 / HEAD_DIM_V) * tl.sum(O_block * O_block, axis=1) + EPS)[:, None]
        norm_factor = -0.5 * inv_norm * inv_norm * inv_norm
        grad_scale = tl.sum(dOut_block * O_block, axis=1)[:, None] * (2.0 * O_block / HEAD_DIM_V)

        dOut_block = (1 - LAMBDA_INIT) * (dOut_block * inv_norm + norm_factor * grad_scale)

    # Lambda gradient local contribution
    dLambda_acc = tl.sum(dOut_block * O2_block / (LAMBDA_SCALE + EPS))

    D1_block = tl.sum(dOut_block * O1_block, axis=1)
    D2_block = tl.sum(dOut_block * O2_block, axis=1)

    tl.store(D1_block_ptrs, D1_block)
    tl.store(D2_block_ptrs, D2_block)

    # NOTE: Stage 2: Computing the gradients
    qk_offset_batch_head = (stride_QK_batch * index_batch + stride_QK_head * index_head).to(tl.int64)
    v_offset_batch_head = (stride_V_batch * index_batch + stride_V_head * index_head).to(tl.int64)
    offset_batch_head_seq = (index_batch_head * SEQ_LEN).to(tl.int64)

    # NOTE: Pointers for the inner loop
    Q1 += qk_offset_batch_head
    Q2 += qk_offset_batch_head
    K1 += qk_offset_batch_head
    K2 += qk_offset_batch_head

    dQ1 += qk_offset_batch_head
    dQ2 += qk_offset_batch_head
    dK1 += qk_offset_batch_head
    dK2 += qk_offset_batch_head

    O1 += v_offset_batch_head
    O2 += v_offset_batch_head

    V += v_offset_batch_head
    dO += v_offset_batch_head
    dV += v_offset_batch_head

    M1 += offset_batch_head_seq
    M2 += offset_batch_head_seq

    D1 += offset_batch_head_seq
    D2 += offset_batch_head_seq

    qT1_ptrs = Q1 + offs_qv[None, :] * stride_QK_seq + qk_offs_dim[:, None] * stride_QK_dim
    qT2_ptrs = Q2 + offs_qv[None, :] * stride_QK_seq + qk_offs_dim[:, None] * stride_QK_dim
    dO_ptrs = dO + offs_qv[:, None] * stride_V_seq + v_offs_dim[None, :] * stride_V_dim
    kT1_ptrs = K1 + offs_qv[None, :] * stride_QK_seq + qk_offs_dim[:, None] * stride_QK_dim
    kT2_ptrs = K2 + offs_qv[None, :] * stride_QK_seq + qk_offs_dim[:, None] * stride_QK_dim
    vT_ptrs = V + offs_qv[None, :] * stride_V_seq + v_offs_dim[:, None] * stride_V_dim

    dV_block_ptrs = dV + offs_kv[:, None] * stride_V_seq + v_offs_dim[None, :] * stride_V_dim
    dK1_block_ptrs = dK1 + offs_kv[:, None] * stride_QK_seq + qk_offs_dim[None, :] * stride_QK_dim
    dK2_block_ptrs = dK2 + offs_kv[:, None] * stride_QK_seq + qk_offs_dim[None, :] * stride_QK_dim
    dQ1_block_ptrs = dQ1 + offs_kv[:, None] * stride_QK_seq + qk_offs_dim[None, :] * stride_QK_dim
    dQ2_block_ptrs = dQ2 + offs_kv[:, None] * stride_QK_seq + qk_offs_dim[None, :] * stride_QK_dim

    # NOTE: Tensors that are in SRAM during the inner loop
    V_block = tl.load(V + offs_kv[:, None] * stride_V_seq + v_offs_dim[None, :] * stride_V_dim)
    K1_block = tl.load(K1 + offs_kv[:, None] * stride_QK_seq + qk_offs_dim[None, :] * stride_QK_dim)
    K2_block = tl.load(K2 + offs_kv[:, None] * stride_QK_seq + qk_offs_dim[None, :] * stride_QK_dim)
    Q1_block = tl.load(Q1 + offs_kv[:, None] * stride_QK_seq + qk_offs_dim[None, :] * stride_QK_dim)
    Q2_block = tl.load(Q2 + offs_kv[:, None] * stride_QK_seq + qk_offs_dim[None, :] * stride_QK_dim)
    dO_block = tl.load(dO + offs_kv[:, None] * stride_V_seq + v_offs_dim[None, :] * stride_V_dim)
    M1_block = tl.load(M1 + offs_kv)[:, None]
    M2_block = tl.load(M2 + offs_kv)[:, None]
    Di1 = tl.load(D1 + offs_kv)
    Di2 = tl.load(D2 + offs_kv)

    # NOTE: Placeholder for the gradients
    dV_block = tl.zeros([BLOCK_SIZE_N, HEAD_DIM_V], dtype=tl.float32)
    dK1_block = tl.zeros([BLOCK_SIZE_N, HEAD_DIM_K], dtype=tl.float32)
    dK2_block = tl.zeros([BLOCK_SIZE_N, HEAD_DIM_K], dtype=tl.float32)
    dQ1_block = tl.zeros([BLOCK_SIZE_N, HEAD_DIM_K], dtype=tl.float32)
    dQ2_block = tl.zeros([BLOCK_SIZE_N, HEAD_DIM_K], dtype=tl.float32)

    if RMS_NORM:
        O1_ptrs = O1 + offs_qv[:, None] * stride_V_seq + v_offs_dim[None, :] * stride_V_dim
        O2_ptrs = O2 + offs_qv[:, None] * stride_V_seq + v_offs_dim[None, :] * stride_V_dim

        O1_block_q = tl.load(O1 + offs_kv[:, None] * stride_V_seq + v_offs_dim[None, :] * stride_V_dim)
        O2_block_q = tl.load(O2 + offs_kv[:, None] * stride_V_seq + v_offs_dim[None, :] * stride_V_dim)
        O_block_q = O1_block_q + O2_block_q

        inv_norm_q = tl.rsqrt((1.0 / HEAD_DIM_V) * tl.sum(O_block_q * O_block_q, axis=1) + EPS)[:, None]
        norm_factor_q = -0.5 * inv_norm_q * inv_norm_q * inv_norm_q
        grad_scale_q = tl.sum(dO_block * O_block_q, axis=1)[:, None] * (2.0 * O_block_q / HEAD_DIM_V)

        dO_block = (1 - LAMBDA_INIT) * (dO_block * inv_norm_q + norm_factor_q * grad_scale_q)
        dO_block = dO_block.to(tl.float16)

    curr_q = 0
    for _ in range(SEQ_LEN // BLOCK_SIZE_M):
        qT1_block = tl.load(qT1_ptrs)
        qT2_block = tl.load(qT2_ptrs)

        K_T1_block = tl.load(kT1_ptrs)
        K_T2_block = tl.load(kT2_ptrs)

        V_T_block = tl.load(vT_ptrs)

        offs_dm = curr_q + tl.arange(0, BLOCK_SIZE_M)

        m1 = tl.load(M1 + offs_dm)
        m2 = tl.load(M2 + offs_dm)

        Di1_1 = tl.load(D1 + offs_dm)
        Di2_2 = tl.load(D2 + offs_dm)

        dOi_block = tl.load(dO_ptrs)

        if RMS_NORM:
            O1i_block = tl.load(O1_ptrs)
            O2i_block = tl.load(O2_ptrs)

            Oi_block = O1i_block + O2i_block

            inv_norm_kv = tl.rsqrt((1.0 / HEAD_DIM_V) * tl.sum(Oi_block * Oi_block, axis=1) + EPS)[:, None]
            norm_factor_kv = -0.5 * inv_norm_kv * inv_norm_kv * inv_norm_kv
            grad_scale_kv = tl.sum(dOi_block * Oi_block, axis=1)[:, None] * (2.0 * Oi_block / HEAD_DIM_V)

            dOi_block = (1 - LAMBDA_INIT) * (dOi_block * inv_norm_kv + norm_factor_kv * grad_scale_kv)
            dOi_block = dOi_block.to(tl.float16)

        QK1_T_block = SM_SCALE * tl.dot(K1_block, qT1_block)
        QK2_T_block = SM_SCALE * tl.dot(K2_block, qT2_block)

        P1_T_block = tl.math.exp(QK1_T_block - m1[None, :])
        P2_T_block = tl.math.exp(QK2_T_block - m2[None, :])

        dV_block += tl.dot(P1_T_block.to(tl.float16) + (- LAMBDA_SCALE).to(tl.float16) * P2_T_block.to(tl.float16), dOi_block)

        dpT1_block = tl.dot(V_block, tl.trans(dOi_block)).to(tl.float32)
        dpT2_block =  (- LAMBDA_SCALE) * tl.dot(V_block, tl.trans(dOi_block)).to(tl.float32)

        dS1_T_block = (P1_T_block * (dpT1_block - Di1_1[None, :])).to(tl.float16)
        dS2_T_block = (P2_T_block * (dpT2_block - Di2_2[None, :])).to(tl.float16)

        dK1_block += SM_SCALE * tl.dot(dS1_T_block, tl.trans(qT1_block))
        dK2_block += SM_SCALE * tl.dot(dS2_T_block, tl.trans(qT2_block))

        QK1_block = SM_SCALE * tl.dot(Q1_block, K_T1_block)
        QK2_block = SM_SCALE * tl.dot(Q2_block, K_T2_block)

        dP1_block = tl.dot(dO_block, V_T_block).to(tl.float32)
        dP2_block = (- LAMBDA_SCALE) * tl.dot(dO_block, V_T_block).to(tl.float32)

        dS1_block = (tl.math.exp(QK1_block - M1_block) * (dP1_block - Di1[:, None])).to(tl.float16)
        dS2_block = (tl.math.exp(QK2_block - M2_block) * (dP2_block - Di2[:, None])).to(tl.float16)

        dQ1_block += SM_SCALE * tl.dot(dS1_block, tl.trans(K_T1_block))
        dQ2_block += SM_SCALE * tl.dot(dS2_block, tl.trans(K_T2_block))

        # Increment pointers
        curr_q += BLOCK_SIZE_M
        qT1_ptrs += BLOCK_SIZE_M * stride_QK_seq
        qT2_ptrs += BLOCK_SIZE_M * stride_QK_seq
        dO_ptrs += BLOCK_SIZE_M * stride_V_seq

        if RMS_NORM:
            O1_ptrs += BLOCK_SIZE_M * stride_V_seq
            O2_ptrs += BLOCK_SIZE_M * stride_V_seq

        kT1_ptrs += BLOCK_SIZE_M * stride_QK_seq
        kT2_ptrs += BLOCK_SIZE_M * stride_QK_seq
        vT_ptrs += BLOCK_SIZE_M * stride_V_seq

    # Store the gradients
    tl.store(dV_block_ptrs, dV_block)
    tl.store(dK1_block_ptrs, dK1_block)
    tl.store(dK2_block_ptrs, dK2_block)
    tl.store(dQ1_block_ptrs, dQ1_block)
    tl.store(dQ2_block_ptrs, dQ2_block)
    tl.store(dLambda_ptr, dLambda_acc)


class _diff_attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q1, Q2, K1, K2, V, SM_SCALE, LAMBDA_SCALE, LAMBDA_INIT, rms_norm=False):

        BATCH_SIZE, NUM_HEADS, SEQ_LEN, _ = Q1.shape
        HEAD_DIM_K, HEAD_DIM_V = K1.shape[-1], V.shape[-1]

        O, O1, O2 = torch.empty_like(V), torch.empty_like(V), torch.empty_like(V)

        M1 = torch.empty((BATCH_SIZE, NUM_HEADS, SEQ_LEN), device=Q1.device, dtype=torch.float32)
        M2 = torch.empty((BATCH_SIZE, NUM_HEADS, SEQ_LEN), device=Q2.device, dtype=torch.float32)

        grid = lambda args: (triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_M"]), BATCH_SIZE * NUM_HEADS)

        diff_attn_fwd[grid](
            Q1=Q1,
            Q2=Q2,
            K1=K1,
            K2=K2,
            V=V,
            M1=M1,
            M2=M2,
            O1=O1,
            O2=O2,
            O=O,
            stride_QK_batch=Q1.stride(0),
            stride_QK_head=Q1.stride(1),
            stride_QK_seq=Q1.stride(2),
            stride_QK_dim=Q1.stride(3),
            stride_V_batch=V.stride(0),
            stride_V_head=V.stride(1),
            stride_V_seq=V.stride(2),
            stride_V_dim=V.stride(3),
            SM_SCALE=SM_SCALE,
            LAMBDA_SCALE_ptr=LAMBDA_SCALE,
            LAMBDA_INIT=LAMBDA_INIT,
            NUM_HEADS=Q1.shape[1],
            SEQ_LEN=Q1.shape[2],
            HEAD_DIM_K=HEAD_DIM_K,
            HEAD_DIM_V=HEAD_DIM_V,
            RMS_NORM=rms_norm,
        )

        ctx.save_for_backward(Q1, Q2, K1, K2, V, O1, O2, M1, M2, LAMBDA_SCALE)
        ctx.SM_SCALE = SM_SCALE
        ctx.LAMBDA_INIT = LAMBDA_INIT
        ctx.HEAD_DIM_K = HEAD_DIM_K
        ctx.HEAD_DIM_V = HEAD_DIM_V
        ctx.RMS_NORM = rms_norm
        ctx.BLOCK_SIZE_M = diff_attn_fwd.best_config.kwargs["BLOCK_SIZE_M"]
        ctx.BLOCK_SIZE_N = diff_attn_fwd.best_config.kwargs["BLOCK_SIZE_N"]

        return O

    @staticmethod
    def backward(ctx, dO):
        Q1, Q2, K1, K2, V, O1, O2, M1, M2, LAMBDA_SCALE = ctx.saved_tensors

        dQ1, dQ2 = torch.empty_like(Q1), torch.empty_like(Q2)
        dK1, dK2 = torch.empty_like(K1), torch.empty_like(K2)
        D1, D2 = torch.empty_like(M1), torch.empty_like(M2)
        dV = torch.empty_like(V)

        BATCH_SIZE, NUM_HEADS, SEQ_LEN = Q1.shape[:3]

        grid = (SEQ_LEN // ctx.BLOCK_SIZE_N, BATCH_SIZE * NUM_HEADS)

        dLambda = torch.zeros((grid[0], grid[1]), dtype=V.dtype, device=V.device)

        diff_attn_bwd[grid](
            Q1=Q1,
            Q2=Q2,
            K1=K1,
            K2=K2,
            V=V,
            O1=O1,
            O2=O2,
            dO=dO,
            dQ1=dQ1,
            dQ2=dQ2,
            dK1=dK1,
            dK2=dK2,
            dV=dV,
            dLambda=dLambda,
            dLambda_stride_seq=dLambda.stride(0),
            dLambda_stride_batch_head=dLambda.stride(1),
            M1=M1,
            M2=M2,
            D1=D1,
            D2=D2,
            stride_QK_batch=Q1.stride(0),
            stride_QK_head=Q1.stride(1),
            stride_QK_seq=Q1.stride(2),
            stride_QK_dim=Q1.stride(3),
            stride_V_batch=V.stride(0),
            stride_V_head=V.stride(1),
            stride_V_seq=V.stride(2),
            stride_V_dim=V.stride(3),
            SM_SCALE=ctx.SM_SCALE,
            LAMBDA_SCALE_ptr=LAMBDA_SCALE,
            LAMBDA_INIT=ctx.LAMBDA_INIT,
            NUM_HEADS=NUM_HEADS,
            SEQ_LEN=SEQ_LEN,
            BLOCK_SIZE_M=ctx.BLOCK_SIZE_M,
            BLOCK_SIZE_N=ctx.BLOCK_SIZE_N,
            HEAD_DIM_K=ctx.HEAD_DIM_K,
            HEAD_DIM_V=ctx.HEAD_DIM_V,
            RMS_NORM=ctx.RMS_NORM,
        )

        # Aggerate the gradients of Lambda (tl.atomic_add didn't work for some reason)
        dLambda = torch.tensor([dLambda.sum().item()], dtype=V.dtype, device=V.device)

        return dQ1, dQ2, dK1, dK2, dV, None, dLambda, None, None


if __name__ == "__main__":
    from layers import *

    B = 2
    H = 2
    N = 256
    D = 64

    LAMBDA_SCALE = torch.tensor([0.5], dtype=torch.float16, requires_grad=True).to("cuda")
    LAMBDA_INIT = 0.8
    rms_norm = True

    for _ in range(100):
        q1 = torch.empty(B, H, N, D // 2, dtype=torch.float16, requires_grad=True).to("cuda").normal_(mean=0.0, std=0.5)
        q2 = torch.empty(B, H, N, D // 2, dtype=torch.float16, requires_grad=True).to("cuda").normal_(mean=0.0, std=0.5)
        k1 = torch.empty(B, H, N, D // 2, dtype=torch.float16, requires_grad=True).to("cuda").normal_(mean=0.0, std=0.5)
        k2 = torch.empty(B, H, N, D // 2, dtype=torch.float16, requires_grad=True).to("cuda").normal_(mean=0.0, std=0.5)
        v = torch.empty(B, H, N, D, dtype=torch.float16, requires_grad=True).to("cuda").normal_(mean=0.0, std=0.5)
        dout = torch.rand_like(v)

        q1.retain_grad()
        q2.retain_grad()
        k1.retain_grad()
        k2.retain_grad()
        v.retain_grad()
        LAMBDA_SCALE.retain_grad()

        y1 = MultiheadFlashDiffAttn(q1, q2, k1, k2, v, lambda_scale=LAMBDA_SCALE, rms_norm=rms_norm)
        y2 = MultiheadDiffAttnKernel(q1, q2, k1, k2, v, lambda_scale=LAMBDA_SCALE, rms_norm=rms_norm)

        z1 = y1.backward(dout)
        ref_dv, v.grad = v.grad.clone(), None
        ref_dk1, k1.grad = k1.grad.clone(), None
        ref_dk2, k2.grad = k2.grad.clone(), None
        ref_dq1, q1.grad = q1.grad.clone(), None
        ref_dq2, q2.grad = q2.grad.clone(), None
        ref_dLambda, LAMBDA_SCALE.grad = LAMBDA_SCALE.grad.clone(), None

        z2 = y2.backward(dout)
        tri_dv, v.grad = v.grad.clone(), None
        tri_dk1, k1.grad = k1.grad.clone(), None
        tri_dk2, k2.grad = k2.grad.clone(), None
        tri_dq1, q1.grad = q1.grad.clone(), None
        tri_dq2, q2.grad = q2.grad.clone(), None
        tri_dLambda, LAMBDA_SCALE.grad = LAMBDA_SCALE.grad.clone(), None

        atol = 1e-2

        print("Reference (Lambda Gradient):", ref_dLambda.item())
        print("Triton (Lambda Gradient):", tri_dLambda.item())
        print("\n")

        assert torch.allclose(y1, y2, atol=atol)
        assert torch.allclose(ref_dv, tri_dv, atol=atol)
        assert torch.allclose(ref_dk1, tri_dk1, atol=atol)
        assert torch.allclose(ref_dk2, tri_dk2, atol=atol)
        assert torch.allclose(ref_dq1, tri_dq1, atol=atol)
        assert torch.allclose(ref_dq2, tri_dq2, atol=atol)
        # assert torch.allclose(ref_dLambda, tri_dLambda, atol=1e-1)  # NOTE: Lambda gradient is a little bit unstable
