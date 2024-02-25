from chex import Array, ArrayTree
import flax.linen as nn
import jax.numpy as jnp


class Normalizer(nn.Module):
    """Normalizer module."""

    decay: float = 0.99
    max_scale: float = 1.0
    q_low: float = 5.0
    q_high: float = 95.0

    def setup(self):
        """Initializes a normalizer."""
        # Statistics
        self.low = self.variable("stats", "low", jnp.zeros, (), jnp.float32)
        self.high = self.variable("stats", "high", jnp.zeros, (), jnp.float32)

    def __call__(self, x: Array) -> ArrayTree:
        """Runs the forward pass of the normalizer."""
        # Update the statistics.
        self.update_stat(x)

        # Get the statistics.
        offset, inv_scale = self.get_stat()

        return offset, inv_scale

    def update_stat(self, x: Array):
        """Updates the statistics given the input."""
        # Get the percentiles.
        low = jnp.percentile(x, self.q_low)
        high = jnp.percentile(x, self.q_high)

        # Update the statistics.
        self.low.value = self.decay * self.low.value + (1 - self.decay) * low
        self.high.value = self.decay * self.high.value + (1 - self.decay) * high

    def get_stat(self) -> ArrayTree:
        """Returns the statistics."""
        # Get the statistics.
        offset = self.low.value
        inv_scale = jnp.maximum(1.0 / self.max_scale, self.high.value - self.low.value)

        return offset, inv_scale
