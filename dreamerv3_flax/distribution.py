from typing import Tuple

from chex import Array, PRNGKey
import distrax
import flax.linen as nn
from jax.lax import stop_gradient
import jax.numpy as jnp

from dreamerv3_flax.jax_util import identity, symexp, symlog


Dist = distrax.Distribution


class Discrete(distrax.Categorical):
    """Discrete distribution."""

    def __init__(
        self,
        logits: Array,
        low: float = -20.0,
        high: float = 20.0,
        trans_type: str = "none",
    ):
        """Initializes a distribution."""
        super().__init__(logits)

        # Bins
        self.bins = jnp.linspace(low, high, num=logits.shape[-1], dtype=jnp.float32)

        # Transform
        if trans_type == "none":
            self.trans = identity
            self.trans_inv = identity
        elif trans_type == "symlog":
            self.trans = symlog
            self.trans_inv = symexp
        else:
            raise NotImplementedError(trans_type)

    def mean(self) -> Array:
        """Calculates the mean."""
        # Calculate the mean.
        mean = jnp.sum(self.probs * self.bins, axis=-1)

        # Apply the inverse transform.
        mean = self.trans_inv(mean)

        return mean

    def log_prob(self, value: Array) -> Array:
        """Calculates the log probability."""
        # Apply the transform.
        value = self.trans(value)

        # Calculate the largest bin index below the value.
        below = self.bins <= value[..., None]
        below = jnp.sum(jnp.astype(below, jnp.int32), axis=-1) - 1
        below = jnp.clip(below, 0, len(self.bins) - 1)

        # Calculate the smallest bin index above the value.
        above = self.bins > value[..., None]
        above = len(self.bins) - jnp.sum(jnp.astype(above, jnp.int32), axis=-1)
        above = jnp.clip(above, 0, len(self.bins) - 1)

        # Calculate the distance between the value and each of the bins.
        equal = below == above
        dist_to_below = jnp.where(equal, 1, jnp.abs(self.bins[below] - value))
        dist_to_above = jnp.where(equal, 1, jnp.abs(self.bins[above] - value))

        # Calculate the weight for each of the bins.
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total

        # Calculate the target.
        target_below = nn.one_hot(below, len(self.bins)) * weight_below[..., None]
        target_above = nn.one_hot(above, len(self.bins)) * weight_above[..., None]
        target = target_below + target_above

        # Calculate the log probability.
        log_prob = jnp.sum(target * self.logits, axis=-1)

        return log_prob


class MSE(distrax.Distribution):
    """MSE distribution."""

    def __init__(self, loc: Array, trans_type: str = "none"):
        """Initializes a distribution."""
        super().__init__()

        # Location
        self._loc = loc

        # Transform
        if trans_type == "none":
            self.trans = identity
            self.trans_inv = identity
        elif trans_type == "symlog":
            self.trans = symlog
            self.trans_inv = symexp
        else:
            raise NotImplementedError(trans_type)

    @property
    def event_shape(self) -> Tuple[int, ...]:
        """Returns the event shape."""
        return ()

    @property
    def loc(self) -> Array:
        """Returns the location."""
        return self._loc

    def _sample_n(self, key: PRNGKey, n: int) -> Array:
        """Returns samples."""
        # Get samples.
        sample = jnp.repeat(self.loc[None], n, axis=0)

        # Apply the inverse transform.
        sample = self.trans_inv(sample)

        return sample

    def mode(self) -> Array:
        """Returns the mode"""
        # Apply the inverse transform.
        mode = self.trans_inv(self.loc)

        return mode

    def log_prob(self, value: Array) -> Array:
        """Calculates the log probability."""
        # Calculate the negative MSE.
        log_prob = -jnp.square(self.loc - self.trans(value))

        return log_prob


class OneHotCategorical(distrax.OneHotCategorical):
    """One-hot categorical distribution."""

    def __init__(self, logits: Array, uniform_mix: float = 0.01):
        """Initializes a distribution."""
        if uniform_mix:
            # Calculate the probability.
            probs = nn.softmax(logits, axis=-1)

            # Define the uniform distribution.
            uniform = jnp.ones_like(probs) / probs.shape[-1]

            # Mix the probability with the uniform distribution.
            probs = (1.0 - uniform_mix) * probs + uniform_mix * uniform

            # Calculate the logit.
            logits = jnp.log(probs)

        super().__init__(logits)

    def _sample_n(self, key: PRNGKey, n: int) -> Array:
        """Returns samples."""
        # Get samples.
        sample = super()._sample_n(key, n)

        # Calculate the straight-through estimator.
        sample += self.probs - stop_gradient(self.probs)

        return sample
