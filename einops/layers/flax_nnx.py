"""Einops layers for Flax NNX (the Linen layers remain in ``einops.layers.flax``)."""

from typing import Any, cast

import jax
import jax.numpy as jnp
from flax import nnx

from . import RearrangeMixin, ReduceMixin
from ._einmix import _EinmixMixin


class Rearrange(nnx.Module):
    """NNX module equivalent to ``einops.rearrange``; dimensions are keyword arguments."""

    def __init__(self, pattern: str, **axes_lengths: Any) -> None:
        self.rearranger = RearrangeMixin(pattern, **axes_lengths)

    def __call__(self, input: jax.Array) -> jax.Array:
        return self.rearranger._apply_recipe(input)


class Reduce(nnx.Module):
    """NNX module equivalent to ``einops.reduce``; dimensions are keyword arguments."""

    def __init__(self, pattern: str, reduction: str, **axes_lengths: Any) -> None:
        self.reducer = ReduceMixin(pattern, reduction, **axes_lengths)

    def __call__(self, input: jax.Array) -> jax.Array:
        return self.reducer._apply_recipe(input)


class EinMix(nnx.Module, _EinmixMixin):
    """EinMix with NNX parameters, initialized eagerly using ``rngs.params()``.

    Pattern, weight/bias shapes and dimension keywords follow the other EinMix
    layers. Pass ``rngs=nnx.Rngs(seed)`` to initialize trainable parameters.
    """

    def __init__(
        self,
        pattern: str,
        weight_shape: str,
        bias_shape: str | None = None,
        *,
        rngs: nnx.Rngs,
        **axes_lengths: Any,
    ) -> None:
        # RNG streams are needed only at initialization, not in the module state.
        self._rngs = rngs
        try:
            _EinmixMixin.__init__(self, pattern, weight_shape, bias_shape, **axes_lengths)
        finally:
            del self._rngs

    def _create_parameters(
        self, weight_shape: list[int], weight_bound: float, bias_shape: list[int] | None, bias_bound: float
    ) -> None:
        self.weight = nnx.Param(
            jax.random.uniform(self._rngs.params(), weight_shape, minval=-weight_bound, maxval=weight_bound)
        )
        self.bias = (
            None
            if bias_shape is None
            else nnx.Param(jax.random.uniform(self._rngs.params(), bias_shape, minval=-bias_bound, maxval=bias_bound))
        )

    def _create_rearrange_layers(
        self,
        pre_reshape_pattern: str | None,
        pre_reshape_lengths: dict[str, Any] | None,
        post_reshape_pattern: str | None,
        post_reshape_lengths: dict[str, Any] | None,
    ) -> None:
        self.pre_rearrange = (
            None if pre_reshape_pattern is None else Rearrange(pre_reshape_pattern, **cast(dict, pre_reshape_lengths))
        )
        self.post_rearrange = (
            None
            if post_reshape_pattern is None
            else Rearrange(post_reshape_pattern, **cast(dict, post_reshape_lengths))
        )

    def __call__(self, input: jax.Array) -> jax.Array:
        if self.pre_rearrange is not None:
            input = self.pre_rearrange(input)
        result = jnp.einsum(self.einsum_pattern, input, self.weight[...])
        if self.bias is not None:
            result += self.bias[...]
        if self.post_rearrange is not None:
            result = self.post_rearrange(result)
        return result
