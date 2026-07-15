import importlib
import logging
import math
import os
from typing import Any, Callable, Mapping, Tuple

import torch
import triton
import triton.language as tl

from flag_gems.ops.scatter import generate_destination_passing_wrapper, generate_imports
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.code_cache import code_cache_dir
from flag_gems.utils.code_utils import IndentedBuffer, write_atomic
from flag_gems.utils.shape_utils import (
    MemOverlap,
    has_internal_overlapping,
    restride_dim,
)

logger = logging.getLogger(__name__)

# sophgo scatter fast path for the common 2D dim==1 contiguous case.
# The codegen kernel flattens to 1D and recovers the per-element coordinates
# with integer divmod (the TPU slow path, ~3 GB/s). Here a 2D grid supplies
# (i, j) straight from the program ids, so there is no divmod: the row index i
# is affine and only the scattered output column comes from `index`. This
# mirrors the gather dim==1 rowwise kernel.
_SCATTER_GRID_CAP = 64
_SCATTER_BLOCK_N = 4096


@libentry()
@triton.jit
def _scatter_dim1_rowwise_kernel(
    out_ptr,
    index_ptr,
    src_ptr,
    M,  # index/src row count
    N,  # index/src column count
    out_row_stride,  # elements between consecutive out rows
    src_row_stride,  # elements between consecutive src rows
    BLOCK_N: tl.constexpr,
    ROWS_PER_PROG,
    MODE: tl.constexpr,  # 0 = store, 1 = add, 2 = multiply
):
    pid = tle.program_id(0)
    nprog = tle.num_programs(0)
    col = tl.arange(0, BLOCK_N)
    for t in range(ROWS_PER_PROG):
        i = pid + t * nprog
        if i < M:
            idx_row_base = i * N
            out_row_base = i * out_row_stride
            src_row_base = i * src_row_stride
            for c0 in range(0, N, BLOCK_N):
                cols = c0 + col
                m = cols < N
                idx = tl.load(index_ptr + idx_row_base + cols, mask=m, other=0).to(
                    tl.int32
                )
                val = tl.load(src_ptr + src_row_base + cols, mask=m, other=0)
                out_off = out_row_base + idx
                if MODE == 0:
                    tl.store(out_ptr + out_off, val, mask=m)
                elif MODE == 1:
                    tl.atomic_add(out_ptr + out_off, val, mask=m)
                else:  # MODE == 2 (multiply)
                    cur = tl.load(out_ptr + out_off, mask=m, other=0)
                    tl.store(out_ptr + out_off, cur * val, mask=m)


def _scatter_dim1_fast(out, index, src, reduce):
    M, N = index.shape
    out_row_stride = out.stride(0)
    src_row_stride = src.stride(0)
    if reduce is None:
        mode = 0
    elif reduce in ("add", "sum"):
        mode = 1
    elif reduce in ("mul", "multiply", "prod"):
        mode = 2
    else:
        return False
    grid = min(M, _SCATTER_GRID_CAP) if M > 0 else 1
    rows_per_prog = math.ceil(M / grid) if grid > 0 else 1
    with torch_device_fn.device(out.device):
        _scatter_dim1_rowwise_kernel[(grid,)](
            out,
            index,
            src,
            M,
            N,
            out_row_stride,
            src_row_stride,
            BLOCK_N=_SCATTER_BLOCK_N,
            ROWS_PER_PROG=rows_per_prog,
            MODE=mode,
        )
    return True


@libentry()
@triton.jit
def _scatter_dim1_fused_kernel(
    inp_ptr,
    out_ptr,
    index_ptr,
    src_ptr,
    M_inp,  # out/inp row count
    N,  # out/inp column count
    M_src,  # index/src row count (<= M_inp)
    K,  # index/src column count
    inp_row_stride,
    src_row_stride,
    BLOCK_N: tl.constexpr,
    ROWS_PER_PROG,
    MODE: tl.constexpr,  # 0 = store, 1 = add, 2 = multiply
):
    # Fuses the out-of-place clone (inp -> out) with the scatter write so the
    # whole op is a single launch — the per-launch overhead on the TPU emulator
    # dominates the small-src cases, and a separate clone kernel loses there.
    pid = tle.program_id(0)
    nprog = tle.num_programs(0)
    col = tl.arange(0, BLOCK_N)
    for t in range(ROWS_PER_PROG):
        i = pid + t * nprog
        if i < M_inp:
            row_base = i * inp_row_stride
            # 1. copy inp row -> out row
            for c0 in range(0, N, BLOCK_N):
                cols = c0 + col
                m = cols < N
                tl.store(
                    out_ptr + row_base + cols,
                    tl.load(inp_ptr + row_base + cols, mask=m, other=0),
                    mask=m,
                )
            # 2. scatter src row into out row at indexed columns (if this row has src)
            if i < M_src:
                src_row_base = i * src_row_stride
                idx_row_base = i * K
                for c1 in range(0, K, BLOCK_N):
                    kk = c1 + col
                    mk = kk < K
                    idx = tl.load(index_ptr + idx_row_base + kk, mask=mk, other=0).to(
                        tl.int32
                    )
                    val = tl.load(src_ptr + src_row_base + kk, mask=mk, other=0)
                    out_off = row_base + idx
                    if MODE == 0:
                        tl.store(out_ptr + out_off, val, mask=mk)
                    elif MODE == 1:
                        tl.atomic_add(out_ptr + out_off, val, mask=mk)
                    else:  # MODE == 2 (multiply)
                        cur = tl.load(out_ptr + out_off, mask=mk, other=0)
                        tl.store(out_ptr + out_off, cur * val, mask=mk)


def _scatter_dim1_fused_fast(inp, out, index, src, reduce):
    M_inp, N = inp.shape
    M_src, K = index.shape
    inp_row_stride = inp.stride(0)
    src_row_stride = src.stride(0)
    if reduce is None:
        mode = 0
    elif reduce in ("add", "sum"):
        mode = 1
    elif reduce in ("mul", "multiply", "prod"):
        mode = 2
    else:
        return False
    grid = min(M_inp, _SCATTER_GRID_CAP) if M_inp > 0 else 1
    rows_per_prog = math.ceil(M_inp / grid) if grid > 0 else 1
    with torch_device_fn.device(out.device):
        _scatter_dim1_fused_kernel[(grid,)](
            inp,
            out,
            index,
            src,
            M_inp,
            N,
            M_src,
            K,
            inp_row_stride,
            src_row_stride,
            BLOCK_N=_SCATTER_BLOCK_N,
            ROWS_PER_PROG=rows_per_prog,
            MODE=mode,
        )
    return True


def _can_scatter_dim1_fast(inp, dim, index, src, reduce):
    if dim != 1 or inp.ndim != 2:
        return False
    if not (inp.is_contiguous() and index.is_contiguous() and src.is_contiguous()):
        return False
    if index.shape != src.shape or index.shape[0] > inp.shape[0]:
        return False
    if index.dtype not in (torch.int32, torch.int64):
        return False
    if reduce is not None and inp.dtype is torch.bfloat16:
        return False
    return True


def generate_scatter_kernel(
    rank: int,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    code.newline()

    code.writeline("def heur_block(args):")
    with code.indent():
        code.writeline("if(flag_gems.vendor_name in ['metax', 'iluvatar']):")
        with code.indent():
            code.writeline("return 256")
        code.writeline("return 128")
    code.newline()
    code.newline()

    code.writeline("def loop_count(args):")
    with code.indent():
        code.writeline("return 4")
    code.newline()
    code.newline()

    code.writeline("@libentry()")
    code.writeline("@triton.heuristics(")
    with code.indent():
        code.writeline("{")
        with code.indent():
            code.writeline('"BLOCK": heur_block,')
            code.writeline('"LOOP": loop_count,')
        code.writeline("}")
    code.writeline(")")
    inp_stride_vars = ",".join(f"'inp_stride_{i}'" for i in range(rank))
    index_stride_vars = ",".join(f"'index_stride_{i}'" for i in range(rank))
    src_stride_vars = ",".join(f"'src_stride_{i}'" for i in range(rank))
    shape_vars = ",".join(f"'shape_{i}'" for i in range(rank))
    code.writeline(
        f"@triton.jit(do_not_specialize=['N','stride_dim','inp_size_dim',"
        f"{inp_stride_vars},{index_stride_vars},{src_stride_vars},{shape_vars}])"
    )

    code.writeline(f"def {kernel_name}(")
    with code.indent():
        if rank > 0:
            code.writeline("src_strided,")
            code.writeline("index,")
            code.writeline("inp,")
            code.writeline("out,")

            stride_args = ", ".join(f"inp_stride_{i}: int" for i in range(rank))
            code.writeline(f"{stride_args}, # stride for inp")

            stride_args = ", ".join(f"index_stride_{i}: int" for i in range(rank))
            code.writeline(f"{stride_args}, # stride for index")

            stride_args = ", ".join(f"src_stride_{i}: int" for i in range(rank))
            code.writeline(f"{stride_args}, # stride for src")

            shape_args = ", ".join(f"shape_{i}: int" for i in range(rank))
            code.writeline(f"{shape_args}, # shape")
            code.writeline("inp_size_dim,")
            code.writeline("stride_dim,")
            code.writeline("N,")
            code.writeline("IS_ADD: tl.constexpr,")
            code.writeline("IS_MUL: tl.constexpr,")
            code.writeline("BLOCK: tl.constexpr,")
            code.writeline("LOOP: tl.constexpr,")
            code.writeline("INT32_OFFSET: tl.constexpr")

    code.writeline("):")

    with code.indent():
        code.writeline("pid = tl.program_id(0)")
        code.writeline("if not INT32_OFFSET:")
        with code.indent():
            code.writeline("pid = pid.to(tl.int64)")
        code.writeline("offsets = pid * LOOP * BLOCK + tl.arange(0, BLOCK)")

        code.writeline("for loop_iter in tl.static_range(LOOP):")
        with code.indent():
            code.writeline("mask = offsets < N")
            code.writeline("cur_idx = offsets")
            code.writeline("if INT32_OFFSET:")
            with code.indent():
                code.writeline("inp_offsets = tl.zeros((BLOCK, ), dtype=tl.int32)")
                code.writeline("idx_offsets = tl.zeros((BLOCK, ), dtype=tl.int32)")
                code.writeline("src_offsets = tl.zeros((BLOCK, ), dtype=tl.int32)")
            code.writeline("else:")
            with code.indent():
                code.writeline("inp_offsets = tl.zeros((BLOCK, ), dtype=tl.int64)")
                code.writeline("idx_offsets = tl.zeros((BLOCK, ), dtype=tl.int64)")
                code.writeline("src_offsets = tl.zeros((BLOCK, ), dtype=tl.int64)")
            for i in range(rank)[::-1]:
                code.writeline("if INT32_OFFSET:")
                with code.indent():
                    code.writeline(f"shape_{i} = shape_{i}.to(tl.int32)")
                    code.writeline(f"inp_stride_{i} = inp_stride_{i}.to(tl.int32)")
                    code.writeline(f"index_stride_{i} = index_stride_{i}.to(tl.int32)")
                    code.writeline(f"src_stride_{i} = src_stride_{i}.to(tl.int32)")
                code.writeline(f"mod = cur_idx % shape_{i}")
                code.writeline(f"inp_offsets += mod * inp_stride_{i}")
                code.writeline(f"idx_offsets += mod * index_stride_{i}")
                code.writeline(f"src_offsets += mod * src_stride_{i}")
                if i != 0:
                    code.writeline(f"cur_idx = cur_idx // shape_{i}")

            code.writeline(
                "cur_src = tl.load(src_strided + src_offsets, mask=mask, other=0)"
            )
            code.writeline(
                "cur_index = tl.load(index + idx_offsets, mask=mask, other=0)"
            )
            code.writeline("if INT32_OFFSET:")
            with code.indent():
                code.writeline("cur_index = cur_index.to(tl.int32)")
                code.writeline("stride_dim = stride_dim.to(tl.int32)")

            code.writeline("dim_offsets = cur_index * stride_dim")
            code.writeline("inp_offsets += dim_offsets")
            code.newline()
            code.writeline("if IS_ADD: ")
            with code.indent():
                code.writeline(
                    "tl.atomic_add(out + inp_offsets, cur_src, mask=mask, sem='relaxed')"
                )
            code.writeline("elif IS_MUL: ")
            with code.indent():
                code.writeline(
                    "cur_inp = tl.load(out + inp_offsets, mask=mask, other=0)"
                )
                code.writeline("res = cur_inp * cur_src")
                code.writeline("tl.store(out + inp_offsets, res, mask=mask)")

            code.writeline("else: ")
            with code.indent():
                code.writeline("tl.store(out + inp_offsets, cur_src, mask=mask)")

            code.writeline("offsets += BLOCK")

    code.newline()
    code.newline()
    return code


def generate_code(
    inputs: Tuple[Any],
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    shape = inputs[1].shape
    rank = len(shape)

    code = generate_imports(code)
    code = generate_scatter_kernel(rank, kernel_name, code)
    code = generate_destination_passing_wrapper(rank, wrapper_name, kernel_name, code)
    return code


class ScatterFunction:
    def __init__(self):
        self.pid = os.getpid()
        self.overloads: Mapping[str, Callable] = {}

    def __call__(self, *args, **kwargs):
        key = f"{self.arg_key(*args)}"
        if key in self.overloads:
            overload = self.overloads[key]
        else:
            code = IndentedBuffer()
            code = generate_code(
                args,
                "_scatter_wrapper",
                "_scatter_jit_function",
                code,
            )

            file_name = f"scatter_rank_{key}.py"
            file_path = code_cache_dir() / file_name
            write_atomic(file_path, code.getvalue())

            spec = importlib.util.spec_from_file_location(
                f"_gen_module_rank_{key}",
                file_path,
            )

            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            overload = getattr(m, "_scatter_wrapper")
            self.overloads[key] = overload

        return overload(*args, **kwargs)

    def arg_key(self, *args):
        tensors = [item for item in args if torch.is_tensor(item)]
        max_rank = max(item.ndim for item in tensors)
        return max_rank


_scatter_func = ScatterFunction()


def scatter(inp, dim, index, src, reduce=None):
    logger.debug("GEMS SCATTER (sophgo_tpu)")
    if reduce is not None:
        assert inp.dtype not in (
            torch.bfloat16,
        ), "Unsupported operation: reduce scatter bfloat tensors."

    # Fast path: 2D dim==1 contiguous -> single fused launch that copies inp to
    # out AND scatters src (avoids a separate clone kernel, whose per-launch
    # overhead dominates the small-src benchmark cases).
    if _can_scatter_dim1_fast(inp, dim, index, src, reduce):
        out = torch.empty_like(inp)
        if _scatter_dim1_fused_fast(inp, out, index, src, reduce):
            return out

    out = inp.clone()

    if has_internal_overlapping(out) == MemOverlap.Yes:
        out = out.contiguous()

    src_strided = src.as_strided(index.shape, src.stride())
    inp_restrided = restride_dim(inp, dim, index.shape)
    dim_size = inp.size(dim)
    dim_stride = inp.stride(dim)
    N = index.numel()

    int32_size_dim = lambda x: x.stride(dim) * x.size(dim) < 2**32
    use_int32_offset = all(map(int32_size_dim, (inp, index, src)))
    _scatter_func(
        src_strided,
        index,
        inp_restrided,
        out,
        dim_size,
        dim_stride,
        N,
        reduce,
        int32_offset=use_int32_offset,
    )

    return out


def scatter_(inp, dim, index, src, reduce=None):
    logger.debug("GEMS SCATTER_")
    out = inp

    if reduce is not None:
        assert inp.dtype not in (
            torch.bfloat16,
        ), "Unsupported operation: reduce scatter bfloat tensors."

    assert (
        has_internal_overlapping(out) != MemOverlap.Yes
    ), "Unsupported operation: trying to inplace write to an internally overlapping tensor."

    if _can_scatter_dim1_fast(inp, dim, index, src, reduce):
        if _scatter_dim1_fast(out, index, src, reduce):
            return inp

    src_restrided = src.as_strided(index.shape, src.stride())
    inp_restrided = restride_dim(inp, dim, index.shape)
    dim_size = inp.size(dim)
    dim_stride = inp.stride(dim)
    N = index.numel()

    int32_size_dim = lambda x: x.stride(dim) * x.size(dim) < 2**32
    use_int32_offset = all(map(int32_size_dim, (inp, index, src)))
    _scatter_func(
        src_restrided,
        index,
        inp_restrided,
        out,
        dim_size,
        dim_stride,
        N,
        reduce,
        int32_offset=use_int32_offset,
    )

    return inp
