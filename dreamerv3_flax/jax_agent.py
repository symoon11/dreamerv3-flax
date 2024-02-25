from functools import partial
from typing import Dict, Tuple

from chex import Array, ArrayTree, PRNGKey
from flax.core.frozen_dict import FrozenDict
from flax.training.dynamic_scale import DynamicScale
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

from gym import spaces

from dreamerv3_flax.agent import Agent
from dreamerv3_flax.env import VecCrafterEnv
from dreamerv3_flax.optax_util import adam_clip, TrainState


class JAXAgent:
    """JAX Agent."""

    def __init__(
        self,
        env: VecCrafterEnv,
        seed: int = 0,
        model_opt_kwargs: Dict = FrozenDict(lr=1e-4, max_norm=1000.0),
        policy_opt_kwargs: Dict = FrozenDict(lr=3e-5, max_norm=100.0),
    ):
        """Initializes an agent."""
        # Environment
        obs_space = env.single_observation_space
        action_space = env.single_action_space
        assert isinstance(obs_space, spaces.Box)
        assert isinstance(action_space, spaces.Discrete)
        obs_shape = obs_space.shape
        num_actions = action_space.n

        # Agent
        self.agent = Agent(obs_shape, num_actions)

        # Key
        self.key = jax.random.key(seed)

        # Model state
        self.model_state = self.init_model_state(model_opt_kwargs)

        # Policy state
        self.policy_state = self.init_policy_state(policy_opt_kwargs)

    def initial_state(self, batch_size: int) -> ArrayTree:
        """Returns the initial state."""
        variables = {"params": self.model_state.params, "stats": self.model_state.stats}
        state = self.agent.apply(variables, batch_size, method=self.agent.initial_state)
        return state

    def init_model_state(self, model_opt_kwargs: Dict) -> TrainState:
        """Initializes the model state."""
        # Define the model variables.
        param_key, post_key, prior_key, self.key = jax.random.split(self.key, 4)
        rngs = {"params": param_key, "post": post_key, "prior": prior_key}
        data = {
            "obs": jnp.zeros((1, 1, *self.agent.obs_shape), jnp.uint8),
            "action": jnp.zeros((1, 1, self.agent.num_actions), jnp.float32),
            "reward": jnp.zeros((1, 1), jnp.float32),
            "cont": jnp.zeros((1, 1), jnp.float32),
            "first": jnp.zeros((1, 1), jnp.float32),
        }
        variables = self.agent.init(rngs, **data, method=self.agent.model_loss)

        # Define the model state.
        params = variables["params"]
        tx = adam_clip(**model_opt_kwargs)
        dynamic_scale = DynamicScale()
        model_state = TrainState.create(
            apply_fn=partial(self.agent.apply, method=self.agent.model_loss),
            params=params,
            tx=tx,
            stats={},
            dynamic_scale=dynamic_scale,
        )

        return model_state

    def init_policy_state(self, policy_opt_kwargs: Dict) -> TrainState:
        """Initializes the policy state."""
        # Define the policy variables.
        variables = {"params": self.model_state.params, "stats": self.model_state.stats}
        latent_size = self.agent.apply(variables, method=self.agent.latent_size)
        param_key, self.key = jax.random.split(self.key, 2)
        rngs = {"params": param_key}
        traj = {
            "latent": jnp.zeros((2, 1, latent_size), jnp.float16),
            "action": jnp.zeros((2, 1, self.agent.num_actions), jnp.float32),
            "reward": jnp.zeros((1, 1), jnp.float32),
            "cont": jnp.zeros((2, 1), jnp.float32),
        }
        variables = self.agent.init(rngs, **traj, method=self.agent.policy_loss)

        # Define the policy state.
        params = variables["params"]
        stats = variables["stats"]
        tx = adam_clip(**policy_opt_kwargs)
        dynamic_scale = DynamicScale()
        policy_state = TrainState.create(
            apply_fn=partial(self.agent.apply, method=self.agent.policy_loss),
            params=params,
            tx=tx,
            stats=stats,
            dynamic_scale=dynamic_scale,
        )

        return policy_state

    @staticmethod
    @partial(jax.jit, static_argnums=(0,))
    def _act(
        agent: Agent,
        variables: ArrayTree,
        rngs: PRNGKey,
        obs: Array,
        first: Array,
        state: ArrayTree | None = None,
    ) -> ArrayTree:
        """Samples an action (jitted)."""
        return agent.apply(
            variables,
            obs,
            first,
            state=state,
            rngs=rngs,
            method=agent.act,
        )

    def act(
        self,
        obs: Array,
        first: Array,
        state: ArrayTree | None = None,
    ) -> ArrayTree:
        """Samples an action."""
        # Get the agent and key.
        agent = self.agent
        key = self.key

        # Sample an action.
        params = {**self.model_state.params, **self.policy_state.params}
        stats = {**self.model_state.stats, **self.policy_state.stats}
        variables = {"params": params, "stats": stats}
        post_key, prior_key, action_key, key = jax.random.split(key, 4)
        rngs = {"post": post_key, "prior": prior_key, "action": action_key}
        action, state = self._act(agent, variables, rngs, obs, first, state=state)

        # Update the key.
        self.key = key

        return action, state

    @staticmethod
    @jax.jit
    def _train_model(
        model_state: TrainState,
        rngs: PRNGKey,
        data: ArrayTree,
        state: ArrayTree | None = None,
    ) -> Tuple[TrainState, ArrayTree]:
        """Trains the model (jitted)."""

        # Update the model parameters.
        def loss_fn(params: ArrayTree):
            variables = {"params": params, "stats": model_state.stats}
            return model_state.apply_fn(variables, **data, state=state, rngs=rngs)

        grad_fn = model_state.dynamic_scale.value_and_grad(loss_fn, has_aux=True)
        dynamic_scale, finite, aux, grads = grad_fn(model_state.params)
        _, (post, state, model_metric) = aux
        new_model_state = model_state.apply_gradients(grads=grads)

        # Update the model state.
        opt_state = tree_map(
            partial(jnp.where, finite),
            new_model_state.opt_state,
            model_state.opt_state,
        )
        params = tree_map(
            partial(jnp.where, finite),
            new_model_state.params,
            model_state.params,
        )
        model_state = new_model_state.replace(
            opt_state=opt_state,
            params=params,
            dynamic_scale=dynamic_scale,
        )

        return model_state, (post, state, model_metric)

    def train_model(self, data: ArrayTree, state: ArrayTree | None = None) -> ArrayTree:
        """Trains the model."""
        # Get the model state and key.
        model_state = self.model_state
        key = self.key

        # Train the model.
        post_key, prior_key, key = jax.random.split(key, 3)
        rngs = {"post": post_key, "prior": prior_key}
        model_state, (post, state, model_metric) = self._train_model(
            model_state,
            rngs,
            data,
            state=state,
        )

        # Update the model state and key.
        self.model_state = model_state
        self.key = key

        return post, state, model_metric

    @staticmethod
    @partial(jax.jit, static_argnums=(0,))
    def _imagine(
        agent: Agent,
        variables: ArrayTree,
        rngs: PRNGKey,
        post: ArrayTree,
        data: Array,
    ) -> ArrayTree:
        """Runs an imagination (jitted)."""
        return agent.apply(variables, post, **data, rngs=rngs, method=agent.imagine)

    def imagine(self, post: ArrayTree, data: Array) -> ArrayTree:
        """Runs an imagination."""
        # Get the agent and key.
        agent = self.agent
        key = self.key

        # Run an imagination.
        params = {**self.model_state.params, **self.policy_state.params}
        stats = {**self.model_state.stats, **self.policy_state.stats}
        variables = {"params": params, "stats": stats}
        post_key, prior_key, action_key, key = jax.random.split(key, 4)
        rngs = {"post": post_key, "prior": prior_key, "action": action_key}
        traj = self._imagine(agent, variables, rngs, post, data)

        # Update the key.
        self.key = key

        return traj

    @staticmethod
    @jax.jit
    def _train_policy(
        policy_state: TrainState,
        rngs: PRNGKey,
        traj: ArrayTree,
    ) -> Tuple[TrainState, ArrayTree]:
        """Trains the policy (jitted)."""

        # Update the policy parameters.
        def loss_fn(params: ArrayTree):
            variables = {"params": params, "stats": policy_state.stats}
            (policy_loss, policy_metric), variables = policy_state.apply_fn(
                variables,
                **traj,
                mutable="stats",
                rngs=rngs,
            )
            return policy_loss, (policy_metric, variables)

        grad_fn = policy_state.dynamic_scale.value_and_grad(loss_fn, has_aux=True)
        dynamic_scale, finite, aux, grads = grad_fn(policy_state.params)
        _, (policy_metric, variables) = aux
        new_policy_state = policy_state.apply_gradients(grads=grads)

        # Update the policy state.
        opt_state = tree_map(
            partial(jnp.where, finite),
            new_policy_state.opt_state,
            policy_state.opt_state,
        )
        params = tree_map(
            partial(jnp.where, finite),
            new_policy_state.params,
            policy_state.params,
        )
        stats = variables["stats"]
        policy_state = new_policy_state.replace(
            opt_state=opt_state,
            params=params,
            stats=stats,
            dynamic_scale=dynamic_scale,
        )

        return policy_state, policy_metric

    def train_policy(self, traj: ArrayTree) -> ArrayTree:
        """Trains the policy."""
        # Get the policy state and key.
        policy_state = self.policy_state
        key = self.key

        # Train the policy.
        action_key, key = jax.random.split(key, 2)
        rngs = {"action": action_key}
        policy_state, policy_metric = self._train_policy(policy_state, rngs, traj)

        # Update the policy state and key.
        self.policy_state = policy_state
        self.key = key

        return policy_metric

    @staticmethod
    @partial(jax.jit, static_argnums=(0,))
    def _update_policy(agent: Agent, policy_state: TrainState) -> ArrayTree:
        """Updates the policy (jitted)."""
        # Update the policy variables.
        variables = {"params": policy_state.params, "stats": policy_state.stats}
        _, variables = agent.apply(
            variables,
            mutable=["params", "stats"],
            method=agent.update_policy,
        )

        # Update the policy state.
        params = variables["params"]
        stats = variables["stats"]
        policy_state = policy_state.replace(params=params, stats=stats)

        return policy_state

    def update_policy(self):
        """Updates the policy."""
        # Get the agent and policy state.
        agent = self.agent
        policy_state = self.policy_state

        # Update the policy.
        policy_state = self._update_policy(agent, policy_state)

        # Update the policy state.
        self.policy_state = policy_state

    def train(self, data: ArrayTree, state: ArrayTree | None = None) -> ArrayTree:
        """Trains the agent."""
        # Train the agent.
        post, state, model_metric = self.train_model(data, state=state)
        traj = self.imagine(post, data)
        policy_metric = self.train_policy(traj)
        self.update_policy()

        # Define the train metric.
        train_metric = {**model_metric, **policy_metric}

        return state, train_metric
