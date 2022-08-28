# TODO: writ e a wrapper around the Search GYM 

from typing import Any, Tuple, TypeVar
import jax.numpy as jnp
from copy import deepcopy
import chex
import pax
E = TypeVar('E')

class mcts_env_wrapper(pax.Module):
    """A template for environments."""

    obs: chex.Array
    terminated: chex.Array


    def __init__(self,env_module):
        super().__init__()
        self.env = env_module
        self.terminated = False

    def step(self, action: chex.Array) -> Tuple[E, chex.Array]:
        """A single env step."""
        # construct action dictionary
        rewards,terminated = self.env.step_all(dict({0:action}))
        self.terminated = terminated
        env_new = mcts_env_wrapper(deepcopy(self.env))
        return env_new,jnp.array(rewards).sum()

    def reset(self):
        """Reset the enviroment."""
        self.env.reset()

    def is_terminated(self) -> chex.Array:
        """The env is terminated."""
        return self.terminated

    def observation(self) -> Any:
        obs_all = self.env.get_obs_all()['obs'][0]
        return jnp.array(obs_all)

    def canonical_observation(self) -> Any:
        return self.observation()

    @property
    def num_actions(self) -> int:
        """Return the size of the action space."""
        return self.env.action_size

    def invalid_actions(self) -> chex.Array:
        """An boolean array indicating invalid actions.
        Returns:
            invalid_action: the i-th element is true if action `i` is invalid. [num_actions].
        """
        valids = self.env.get_obs_all()['valids'][0]
        return jnp.array(valids)

    def max_num_steps(self) -> int:
        """Return the maximum number of steps until the game is terminated."""
        return self.env.max_steps

    @property
    def input_size(self):
        return self.env.input_size

