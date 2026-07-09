import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic
from flag_gems.utils.codegen_config_utils import CodeGenConfig, get_codegen_config

logger = logging.getLogger(__name__)

# Larger tile + wide grid, matching the tuned gelu/where overrides. This
# bandwidth-bound elementwise op benefits from the bigger tile size (default
# SOPHGO config is only 1024 / grid (512,1,1)).
_base = get_codegen_config()
_config = CodeGenConfig(
    max_tile_size=2048,
    max_grid_size=(65536, 1, 1),
    max_num_warps_per_cta=_base.max_num_warps_per_cta,
    prefer_block_pointer=_base.prefer_block_pointer,
    prefer_1d_tile=_base.prefer_1d_tile,
)


@pointwise_dynamic(promotion_methods=[(0, "ALWAYS_BOOL")], config=_config)
@triton.jit
def logical_not_func(x):
    if x.dtype.is_floating():
        x = tl.abs(x)
    return not x.to(tl.int1)


def logical_not(A):
    logger.debug("GEMS LOGICAL_NOT")
    return logical_not_func(A)
