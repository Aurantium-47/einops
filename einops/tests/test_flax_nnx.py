"""NNX layer integration tests, run with the existing JAX test selection."""

import numpy as np
import pytest

from einops import EinopsError, rearrange, reduce
from einops.tests import is_backend_tested

pytestmark = pytest.mark.skipif(not is_backend_tested("jax"), reason="JAX backend not selected")


@pytest.mark.parametrize(
    "pattern,shape,axes",
    [
        ("b (h w) c -> b h (w c)", (2, 12, 5), dict(h=3)),
        ("... c -> c ...", (2, 3, 4, 5), {}),
        ("... c -> (...) c", (5,), {}),
        ("b c -> c b", (0, 3), {}),
    ],
)
def test_rearrange(pattern, shape, axes):
    import jax
    import jax.numpy as jnp
    from flax import nnx

    from einops.layers.flax_nnx import Rearrange

    x = jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape)
    layer = Rearrange(pattern, **axes)
    expected = rearrange(x, pattern, **axes)
    np.testing.assert_array_equal(layer(x), expected)
    np.testing.assert_array_equal(nnx.jit(lambda module, x: module(x))(layer, x), expected)
    np.testing.assert_array_equal(jax.jit(layer)(x), expected)
    assert not nnx.state(layer)
    np.testing.assert_array_equal(nnx.merge(*nnx.split(layer))(x), expected)
    np.testing.assert_array_equal(jax.grad(lambda x: jnp.sum(layer(x)))(x), jnp.ones_like(x))


@pytest.mark.parametrize("reduction", ["min", "max", "sum", "mean", "prod", "any", "all"])
def test_reduce(reduction):
    import jax.numpy as jnp
    from flax import nnx

    from einops.layers.flax_nnx import Reduce

    x = jnp.arange(48, dtype=jnp.float32).reshape(2, 6, 4) / 48
    if reduction in ("any", "all"):
        x = x > 0.5
    pattern = "b (h h2) c -> b h"
    layer = Reduce(pattern, reduction, h2=2)
    expected = reduce(x, pattern, reduction, h2=2)
    np.testing.assert_allclose(layer(x), expected)
    np.testing.assert_allclose(nnx.jit(lambda m, x: m(x))(layer, x), expected)
    assert not nnx.state(layer)


@pytest.mark.parametrize("bias", [None, "out"])
@pytest.mark.parametrize("grouped", [False, True])
def test_einmix_forward_and_gradients(bias, grouped):
    import jax
    import jax.numpy as jnp
    from flax import nnx

    from einops.layers.flax_nnx import EinMix

    pattern = "b (head inp) -> b (head out)" if grouped else "... head inp -> ... head out"
    x = jnp.arange(24, dtype=jnp.float32).reshape((2, 12) if grouped else (2, 3, 4)) / 24
    layer = EinMix(pattern, "head inp out", bias, head=3, inp=4, out=5, rngs=nnx.Rngs(12))
    assert isinstance(layer.weight, nnx.Param)
    assert (layer.bias is None) == (bias is None)
    assert set(nnx.state(layer)) == ({"weight"} if bias is None else {"weight", "bias"})

    def reference(weight, bias, x):
        result = jnp.einsum("bhi,hio->bho", x.reshape(2, 3, 4), weight)
        if bias is not None:
            result = result + bias
        return result.reshape(2, 15) if grouped else result

    bias_array = None if layer.bias is None else layer.bias[...]
    expected = reference(layer.weight[...], bias_array, x)
    np.testing.assert_allclose(layer(x), expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(nnx.jit(lambda m, x: m(x))(layer, x), expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        jax.grad(lambda x: jnp.square(layer(x)).sum())(x),
        jax.grad(lambda x: jnp.square(reference(layer.weight[...], bias_array, x)).sum())(x),
        rtol=1e-6,
        atol=1e-6,
    )
    grads = nnx.to_pure_dict(nnx.grad(lambda m: jnp.square(m(x)).sum())(layer))
    weight_grad, bias_grad = jax.grad(lambda w, b: jnp.square(reference(w, b, x)).sum(), argnums=(0, 1))(
        layer.weight[...], bias_array
    )
    np.testing.assert_allclose(grads["weight"], weight_grad, rtol=1e-6, atol=1e-6)
    if bias is not None:
        np.testing.assert_allclose(grads["bias"], bias_grad, rtol=1e-6, atol=1e-6)


def test_einmix_initialization():
    import jax.numpy as jnp
    from flax import nnx

    from einops.layers.flax_nnx import EinMix

    def make(rngs):
        return EinMix("b inp -> b out", "inp out", "out", inp=32, out=16, rngs=rngs)

    rngs = nnx.Rngs(params=0)
    first, second = make(rngs), make(rngs)
    repeated = make(nnx.Rngs(params=0))
    np.testing.assert_array_equal(first.weight[...], repeated.weight[...])
    np.testing.assert_array_equal(first.bias[...], repeated.bias[...])
    assert not jnp.array_equal(first.weight[...], second.weight[...])
    for parameter, bound in [(first.weight, (3 / 32) ** 0.5), (first.bias, (1 / 32) ** 0.5)]:
        assert jnp.all(parameter[...] >= -bound)
        assert jnp.all(parameter[...] < bound)
        assert jnp.any(parameter[...] < 0)
        assert jnp.any(parameter[...] > 0)
    assert not nnx.state(first, nnx.RngState)


def test_model_training_and_serialization():
    import flax.serialization
    import jax
    import jax.numpy as jnp
    from flax import nnx

    from einops.layers.flax_nnx import EinMix, Rearrange, Reduce

    def make(seed):
        return nnx.Sequential(
            EinMix(
                "b (h h2) (w w2) c -> b h (w out)", "h2 w2 c out", "out", h2=2, w2=3, c=4, out=5, rngs=nnx.Rngs(seed)
            ),
            Rearrange("b h wc -> b (h wc)"),
            Reduce("b features -> b", "mean"),
        )

    model = make(0)
    x = jnp.arange(864, dtype=jnp.float32).reshape(2, 4, 9, 12)[:, :, :, :4] / 864

    @nnx.jit
    def step(model, x):
        loss, grads = nnx.value_and_grad(lambda m: jnp.square(m(x)).mean())(model)
        params = nnx.state(model, nnx.Param)
        nnx.update(model, jax.tree.map(lambda p, g: p - 0.001 * g, params, grads))
        return loss

    before = step(model, x)
    after = jnp.square(model(x)).mean()
    assert after < before
    graphdef, state = nnx.split(model)
    np.testing.assert_allclose(nnx.merge(graphdef, state)(x), model(x))
    # Checkpoint only array state, then restore into an independently initialized graph.
    raw_state = nnx.to_pure_dict(state)
    restored_raw = flax.serialization.from_bytes(raw_state, flax.serialization.to_bytes(raw_state))
    restored = make(23)
    assert not jnp.allclose(restored(x), model(x))
    restored_state = nnx.state(restored)
    nnx.replace_by_pure_dict(restored_state, restored_raw)
    nnx.update(restored, restored_state)
    np.testing.assert_allclose(restored(x), model(x), rtol=1e-6, atol=1e-6)


def test_invalid_patterns_and_shapes():
    import jax.numpy as jnp
    from flax import nnx

    from einops.layers.flax_nnx import EinMix, Rearrange, Reduce

    with pytest.raises(EinopsError):
        Rearrange("a -> b")
    with pytest.raises(EinopsError):
        Reduce("a -> a", "unknown")
    with pytest.raises(EinopsError):
        Rearrange("b (h w) -> b h w", h=3)(jnp.zeros((2, 5)))
    with pytest.raises(EinopsError):
        EinMix("b inp -> b out", "inp out", inp=3, rngs=nnx.Rngs(0))


def test_transformed_initialization():
    import jax.numpy as jnp
    from flax import nnx

    from einops.layers.flax_nnx import EinMix

    @nnx.jit
    def build(rngs):
        return EinMix("b inp -> b out", "inp out", "out", inp=3, out=4, rngs=rngs)

    x = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
    model = build(nnx.Rngs(0))
    np.testing.assert_allclose(model(x), x @ model.weight[...] + model.bias[...], atol=1e-6)

    @nnx.vmap(in_axes=0, out_axes=0)
    def build_batch(rngs):
        return EinMix("b inp -> b out", "inp out", inp=3, out=4, rngs=rngs)

    rngs = nnx.Rngs(0)
    with nnx.split_rngs(rngs, splits=2):
        models = build_batch(rngs)
    actual = nnx.vmap(lambda model, x: model(x), in_axes=(0, None))(models, x)
    expected = jnp.einsum("bi,nio->nbo", x, models.weight[...])
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    assert not jnp.array_equal(models.weight[0], models.weight[1])
