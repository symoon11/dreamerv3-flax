from chex import Array
import jax.numpy as jnp


def identity(x: Array) -> Array:
    """Defines the identity function."""
    return x


def symlog(x: Array) -> Array:
    """Defines the symlog function."""
    return jnp.sign(x) * jnp.log(1 + jnp.abs(x))


def symexp(x: Array) -> Array:
    """Defines the symexp function."""
    return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1)


def where(condition: Array, x: Array, y: Array = None) -> Array:
    """Defines the mask function."""
    x = jnp.einsum("b...,b->b...", x, condition)
    if y is not None:
        x += jnp.einsum("b...,b->b...", y, 1 - condition)
    return x
