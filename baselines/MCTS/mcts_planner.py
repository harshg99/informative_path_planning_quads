
from baselines.il_wrapper import il_wrapper
import pickle
import random
import warnings
from functools import partial

import chex
import jax
import jax.numpy as jnp


from mcts_env import mcts_env_wrapper as Enviroment
from mcts_treesearch import improve_policy_with_mcts, recurrent_fn
from mcts_utils import env_step, import_class, replicate, reset_env

# TODO: change this function

def play_one_move(
    agent,
    env: Enviroment,
    rng_key: chex.Array,
    enable_mcts: bool = False,
    num_simulations: int = 1024,
    random_action: bool = True,
):
    """Play a move using agent's policy"""
    if enable_mcts:
        batched_env = replicate(env, 1)
        rng_key, rng_key_1 = jax.random.split(rng_key)
        policy_output = improve_policy_with_mcts(
            agent,
            batched_env,
            rng_key_1,
            rec_fn=recurrent_fn,
            num_simulations=num_simulations,
        )
        action_weights = policy_output.action_weights[0]
        root_idx = policy_output.search_tree.ROOT_INDEX
        value = policy_output.search_tree.node_values[0, root_idx]
    else:
        action_logits, value = agent(env.canonical_observation())
        action_weights = jax.nn.softmax(action_logits, axis=-1)

    if random_action:
        action = jax.random.categorical(rng_key, jnp.log(action_weights), axis=-1)
    else:
        action = jnp.argmax(action_weights)
    return action, action_weights, value

class mcts_planner(il_wrapper):
    def __init__(self):
        pass