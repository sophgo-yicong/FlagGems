import logging

import torch
import triton

from flag_gems.utils import pointwise_dynamic
from flag_gems.utils.codegen_config_utils import get_codegen_config_with_max_tile_size


config_ = get_codegen_config_with_max_tile_size(4096 * 2)


@pointwise_dynamic(
    promotion_methods=[(0, "DEFAULT")],
    config=config_,
)
@triton.jit
def resolve_conj_func(x):
    return x


def resolve_conj(A: torch.Tensor):
    logging.debug("SOPHGO GEMS RESOLVE_CONJ")
    if A.is_complex():
        if A.is_conj():
            return torch.complex(A.real, -A.imag)
        return A
    return resolve_conj_func(A)