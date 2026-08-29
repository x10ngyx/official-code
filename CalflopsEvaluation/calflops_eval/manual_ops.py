from __future__ import annotations

from typing import Any


def dense_attention_counts(
    *,
    batch_size: int,
    query_tokens: int,
    key_value_tokens: int,
    num_heads: int,
    head_dim: int,
    softmax_flops_per_element: int = 5,
) -> dict[str, Any]:
    """Return theoretical dense attention-core counts, excluding Q/K/V projections.

    The two matrix products are QK^T and softmax(QK^T)V. One MAC is
    represented as two FLOPs. The default softmax convention counts five
    operations per score element and is surfaced in the returned metadata.
    """

    values = {
        "batch_size": batch_size,
        "query_tokens": query_tokens,
        "key_value_tokens": key_value_tokens,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "softmax_flops_per_element": softmax_flops_per_element,
    }
    for name, value in values.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    score_elements = batch_size * num_heads * query_tokens * key_value_tokens
    macs_per_matmul = score_elements * head_dim
    macs = 2 * macs_per_matmul
    matmul_flops = 2 * macs
    softmax_flops = softmax_flops_per_element * score_elements
    return {
        "flops": matmul_flops + softmax_flops,
        "macs": macs,
        "matmul_flops": matmul_flops,
        "softmax_flops": softmax_flops,
        "score_elements": score_elements,
        "formula": "4*B*H*Nq*Nk*Dh + softmax_ops*B*H*Nq*Nk",
        "inputs": values,
    }


def elementwise_flops(*, num_elements: int, operations_per_element: int = 1) -> int:
    if not isinstance(num_elements, int) or isinstance(num_elements, bool) or num_elements < 0:
        raise ValueError("num_elements must be a non-negative integer")
    if (
        not isinstance(operations_per_element, int)
        or isinstance(operations_per_element, bool)
        or operations_per_element < 0
    ):
        raise ValueError("operations_per_element must be a non-negative integer")
    return num_elements * operations_per_element
