import functools

import numpy as np

from env.SemanticMap import  GPSemanticMap
import env_params.Semantic as env_params
import Utilities

import multiprocessing as mp
env_params = Utilities.set_dict(env_params)

def unit_create_semantic_tests(test_index =None,num_maps_index = None,save_dir = None):
    map_index = test_index
    proximities = list()

    if save_dir is None:
        save_dir = env_params['assets_folder']

    for j in range(num_maps_index):
        # Initialize the maps
        map_config_dict = {
            'resolution': env_params['resolution'],
            'world_map_size': env_params['rewwardMapSize'] + 2 *env_params['pad_size'],
            'padding': env_params['pad+ size'],
            'num_semantics': env_params['num_semantics'],
            'target_belief_thresh': env_params['TargetBeliefThresh']
        }
        ground_truth_semantic_map = GPSemanticMap(map_config_dict, isGroundTruth=True)
        belief_semantic_map = GPSemanticMap(map_config_dict)

        # map_index= 10
        load_dict = {
            'semantic_file_path': env_params['assets_folder'] + 'sem{}.npy'.format(map_index),
            'map_image_file_path': env_params['assets_folder'] + 'gmap{}.png'.format(map_index)
        }

        ground_truth_semantic_map.init_map(load_dict)

        params_dict = {
            "randomness": env_params['TARGET_NOISE_SCALE'],
            "num_centres": env_params['RANDOM_CENTRES'],
            "sigma": env_params['CENTRE_SIZE'],
            "clip": env_params['MAX_CLIP']
        }
        proximity = belief_semantic_map.init_prior_semantics(params_dict=params_dict,
                                                                       ground_truth_map=ground_truth_semantic_map)
        # Store the prior map in the save_dir
        import os
        if os.path.exists(save_dir) is False:
            os.makedirs(save_dir)

        np.save(save_dir + 'sem{}.npy'.format(test_index*env_params['TOTAL_MAPS']+j),
                belief_semantic_map.semantic_map,
                allow_pickle=False)

        proximities.append(proximity)

    return proximities

def create_semantic_tests(test_indices ,num_maps_index = None,num_cpus = None,save_dir =None):

    if num_cpus is None:
        num_cpus = mp.cpu_count()
    if num_maps_index is None:
        num_maps_index = 3
    if save_dir is None:
        save_dir = env_params['assets_folder']

    unit_create_semantic_tests_red = functools.partial(unit_create_semantic_tests,
                                                       num_maps_index=num_maps_index,
                                                       save_dir = save_dir)
    pool = mp.Pool(num_cpus)
    proximities = pool.map(unit_create_semantic_tests_red, test_indices)

    return proximities


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_cpus', type=int, default=5)
    parser.add_argument('--save_dir',type = str,default='tests/semantic_tests/')
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    num_maps = args.num_maps
    num_cpus = args.num_cpus


    test_indices = list(range(env_params['TOTAL_MAPS']))
    proximities = create_semantic_tests(test_indices, env_params['TEST_PER_MAP'], num_cpus)

    proximities = np.concatenate(proximities,axis=1)
    divergences_mean = proximities.mean()
    divergences_std = proximities.std()
    print('Divergences Mean{} Std{} Max{} Min {}'.format(divergences_mean,
                                                         divergences_std,
                                                         proximities.max(),
                                                         proximities.min()))
    np.save(args.save_dir + 'proximities.npy', proximities)


