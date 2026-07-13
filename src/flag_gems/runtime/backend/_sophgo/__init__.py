from backend_utils import VendorInfoBase  # noqa: E402

# from triton.runtime import driver  # noqa: E402

vendor_info = VendorInfoBase(
    vendor_name="sophgo",
    device_name="tpu",
    device_query_cmd="tpu-smi",
    dispatch_key="PrivateUse1",
    triton_extra_name="sophgo",
)

# driver.active.get_active_torch_device()

CUSTOMIZED_UNUSED_OPS = (
    # "abs",
    # "abs_",
    # "add",
    # "add_",
    # "addmm",
    # "angle",
    # "arange_start",
    # "arange_start",
    # "arange",
    # "batch_norm",
    "batch_norm_backward",
    # "bitwise_and_tensor",
    # "bitwise_and_tensor_",
    # "bitwise_and_scalar",
    # "bitwise_and_scalar_",
    # "bitwise_and_scalar_tensor",
    # "bitwise_not",
    # "bitwise_not_",
    # "bitwise_or_tensor",
    # "bitwise_or_tensor_",
    # "bitwise_or_scalar",
    # "bitwise_or_scalar_",
    # "bitwise_or_scalar_tensor",
    # "bmm",
    # "clamp",
    # "clamp_",
    # "clamp_tensor",
    # "clamp_tensor_",
    # "cos",
    # "cos_",
    # "pad",
    # "constant_pad_nd",
    # "cumsum",
    # "cummin",
    # "true_divide",
    # "true_divide_",
    # "true_divide",
    # "true_divide_",
    # "div_mode",
    # "div_mode_",
    # "div_mode",
    # "div_mode_",
    # "true_divide",
    # "true_divide_",
    # "true_divide",
    # "true_divide_",
    # "div_mode",
    # "div_mode_",
    # "div_mode",
    # "div_mode_",
    # "true_divide",
    # "true_divide_",
    # "true_divide",
    # "true_divide_",
    # "floor_divide",
    # "floor_divide_",
    # "floor_divide",
    # "floor_divide_",
    # "remainder",
    # "remainder_",
    # "remainder",
    # "remainder_",
    # "remainder",
    # "dropout",
    "dropout_backward",
    # "erf",
    # "erf_",
    # "embedding",
    "embedding_backward",
    # "eq",
    # "eq_scalar",
    # "exp",
    # "exp_",
    # "exponential_",
    # "ge",
    # "ge_scalar",
    # "gelu",
    # "gelu_",
    "gelu_backward",
    # "group_norm",
    "group_norm_backward",
    # "weight_norm_interface",
    "weight_norm_interface_backward",
    # "gt",
    # "gt_scalar",
    # "isfinite",
    # "isin",
    # "isin",
    # "isin",
    # "isinf",
    # "isnan",
    # "minimum",
    # "maximum",
    # "layer_norm",
    "layer_norm_backward",
    # "le",
    # "le_scalar",
    # "lt",
    # "lt_scalar",
    # "log",
    # "rms_norm",
    # "rand",
    # "randn",
    # "rand_like",
    # "randn_like",
    # "zeros",
    # "ones",
    # "full",
    # "zeros_like",
    # "ones_like",
    # "full_like",
    # "linspace",
    # "resolve_neg",
    # "resolve_conj",
    # "normal_tensor_float",
    # "normal_float_tensor",
    # "normal_tensor_tensor",
    # "uniform_",
    # "mean",
    # "mean_dim",
    # "mm",
    # "mul",
    # "mul_",
    # "multinomial",
    # "mv",
    # "nan_to_num",
    # "ne",
    # "ne_scalar",
    # "neg",
    # "neg_",
    # "pow_scalar",
    # "pow_tensor_scalar",
    # "pow_tensor_scalar_",
    # "pow_tensor_tensor",
    # "pow_tensor_tensor_",
    # "reciprocal",
    # "reciprocal_",
    # "relu",
    # "relu_",
    # "rsqrt",
    # "rsqrt_",
    # "sigmoid",
    # "sigmoid_",
    "sigmoid_backward",
    # "silu",
    # "silu_",
    "silu_backward",
    # "sin",
    # "sin_",
    # "softmax",
    "softmax_backward",
    "sort",
    # "sub",
    # "sub_",
    # "tanh",
    # "tanh_",
    "tanh_backward",
    # "threshold",
    "threshold_backward",
    # "triu",
    # "var_mean",
    # "vector_norm",
    # "where_self_out",
    # "where_self",
    # "where_scalar_self",
    # "where_scalar_other",
    # "max",
    # "max_dim",
    # "min",
    # "min_dim",
    # "amax",
    # "argmax",
    # "argmin",
    # "prod",
    # "prod_dim",
    # "sum",
    # "sum_dim",
    # "scaled_dot_product_attention",
    # "all",
    # "all_dim",
    # "all_dims",
    # "any",
    # "any_dim",
    # "any_dims",
    # "quantile",
    # "log_softmax",
    "log_softmax_backward",
    # "nll_loss_forward",
    "nll_loss_backward",
    # "nll_loss2d_forward",
    "nll_loss2d_backward",
    # "scatter",
    # "scatter",
    # "gather",
    "gather_backward",
    # "isclose",
    # "allclose",
    # "fill_scalar",
    # "fill_tensor",
    # "fill_scalar_",
    # "fill_tensor_",
    # "flip",
    # "slice_scatter",
    # "select_scatter",
    # "index_select",
    # "tile",
    # "masked_fill",
    # "masked_fill",
    # "masked_fill_",
    # "masked_fill_",
    # "_unique2",
    # "_upsample_bicubic2d_aa",
    # "upsample_nearest2d",
    # "nonzero",
    # "repeat",
    # "masked_select",
    # "stack",
    # "hstack",
    # "cat",
    # "repeat_interleave_self_int",
    # "vstack",
    # "repeat_interleave_tensor",
    # "repeat_interleave_self_tensor",
    # "randperm",
    # "diag",
    # "diag_embed",
    "diagonal_backward",
    # "index_add",
    # "count_nonzero",
    # "logical_or",
    # "logical_and",
    # "polar",
    # "logical_xor",
    # "logical_not",
    # "dot",
    # "kron",
    # "elu",
    # "index_put_",
    # "index_put",
    # "contiguous",
    # "log_sigmoid",
    # "vdot",
    # "mse_loss",
    "to_copy",
    "index",
)


__all__ = ["*"]


def _register_sophgo_extra_ops():
    """Register sophgo-only op overrides that the shared registry omits.

    Problem this targets: when the backend binds its own ``aten::dropout``
    kernel on both ``PrivateUse1`` and ``AutogradPrivateUse1`` without routing
    it through ``native_dropout``, FlagGems' ``native_dropout`` override can
    never reach ``dropout`` and the backend kernel runs instead. Rather than
    assume this holds, ``_needs_dropout_override`` probes the live dispatch
    table at registration time and the override applies only when it does.

    The shared registration table in ``flag_gems/__init__.py`` intentionally
    omits ``dropout`` (it relies on CPU decomposition), and there is no
    per-vendor hook to append extra op keys. So we wrap ``Register.for_each``
    here -- this module is only imported for the sophgo vendor -- to append the
    ``dropout`` override onto whatever ``lib`` the current ``enable()`` /
    ``use_gems()`` created, keeping it scoped to the same lifecycle.
    """
    import torch

    from flag_gems.runtime.register import Register

    if getattr(Register, "_sophgo_extra_patched", False):
        return
    original_for_each = Register.for_each

    def _needs_dropout_override(reg_key, reg_bac_key):
        """Probe whether this backend actually preempts ``aten::dropout``.

        The override is only needed when the backend has bound a kernel for
        ``dropout`` on both dispatch keys while leaving ``native_dropout``
        unbound (so overriding only ``native_dropout`` cannot reach it). This
        is exactly the situation the override targets; if a future backend
        stops preempting ``dropout`` (or routes it through ``native_dropout``),
        the probe returns False and the override is skipped automatically.

        Must be probed before FlagGems registers anything in this pass: our own
        ``dropout`` registration would make ``dropout_bound`` trivially True, and
        the shared table's ``native_dropout`` registration would flip
        ``native_bound`` and mask a genuine preemption.
        """
        has = torch._C._dispatch_has_kernel_for_dispatch_key
        dropout_bound = has("aten::dropout", reg_key) and has(
            "aten::dropout", reg_bac_key
        )
        native_bound = has("aten::native_dropout", reg_key) or has(
            "aten::native_dropout", reg_bac_key
        )
        return dropout_bound and not native_bound

    def for_each(self):
        # This patch relies on Register internals (reg_key / reg_bac_key / lib).
        # If an upstream FlagGems refactor removes them, fail loudly here instead
        # of silently skipping the override -- otherwise dropout falls back to the
        # pre-existing device kernel with no visible error.
        assert all(hasattr(self, attr) for attr in ("reg_key", "reg_bac_key", "lib")), (
            "sophgo dropout patch: Register interface changed, update "
            "_register_sophgo_extra_ops in _sophgo/__init__.py"
        )
        # Probe the backend's own dispatch state BEFORE any registration in this
        # pass, then cache it for the process -- once we (or the shared table)
        # register, the probe no longer reflects the backend's native state.
        if not hasattr(Register, "_sophgo_dropout_preempted"):
            Register._sophgo_dropout_preempted = _needs_dropout_override(
                self.reg_key, self.reg_bac_key
            )

        original_for_each(self)
        from _sophgo.ops import dropout

        if dropout.__name__ in self.unused_ops:
            return
        # Only override when the backend actually preempts dropout as described
        # above; otherwise the shared native_dropout override already suffices.
        if not Register._sophgo_dropout_preempted:
            return
        # Override both keys: AutogradPrivateUse1 handles the grad-enabled path
        # (default), PrivateUse1 the no-grad path; the backend occupies both.
        for key in (self.reg_key, self.reg_bac_key):
            self.lib.impl("dropout", dropout, key)
        self.all_ops.append("dropout")

    Register.for_each = for_each
    Register._sophgo_extra_patched = True


_register_sophgo_extra_ops()
