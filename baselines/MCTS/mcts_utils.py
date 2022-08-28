"""Useful functions."""

import importlib
from functools import partial
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import pax

from mcts_env import mcts_env_wrapper as E
from env.env_setter import env_setter

@pax.pure
def batched_policy(agent, states):
    """Apply a policy to a batch of states.
    Also return the updated agent.
    """
    return agent, agent(states, batched=True)


def replicate(value: chex.ArrayTree, repeat: int) -> chex.ArrayTree:
    """Replicate along the first axis."""
    return jax.tree_util.tree_map(lambda x: jnp.stack([x] * repeat), value)


@pax.pure
def reset_env(env: E) -> E:
    """Return a reset enviroment."""
    env.reset()
    return env


@jax.jit
def env_step(env: E, action: chex.Array) -> Tuple[E, chex.Array]:
    """Execute one step in the enviroment."""
    env, reward = env.step(action)
    return env, reward

# TODO: Need to import MCTS GP env instances
def import_class(args_dict: dict) -> E:
    """Import a class from a python file.
    For example:
    >> Game = import_class("connect_two_game.Connect2Game")
    Game is the Connect2Game class from `connection_two_game.py`.
    """
    env_module = env_setter.set_env(args_dict)

    return E(env_module)


def select_tree(pred: jnp.ndarray, a, b):
    """Selects a pytree based on the given predicate."""
    assert pred.ndim == 0 and pred.dtype == jnp.bool_, "expected boolean scalar"
    return jax.tree_util.tree_map(partial(jax.lax.select, pred), a, b)