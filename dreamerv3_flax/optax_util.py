from chex import ArrayTree
from flax.training import train_state
from flax.training.dynamic_scale import DynamicScale
import optax


def adam_clip(lr: float, max_norm: float, **kwargs) -> optax.GradientTransformation:
    """Returns the Adam optimizer with the gradient clipping."""
    tx = optax.chain(optax.clip_by_global_norm(max_norm), optax.adam(lr, **kwargs))
    return tx


class TrainState(train_state.TrainState):
    """Train state with the dynamic scale."""

    stats: ArrayTree
    dynamic_scale: DynamicScale
