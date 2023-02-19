import matplotlib.pyplot as plt
import numpy as np
import cupy
from typing import *
from skimage.transform import resize
from skimage.transform import rotate
from copy import deepcopy
import PIL
DEBUG = False
from skimage.measure import block_reduce
import os

def Gaussian(mean, cov,map_size,scale = 0.5):
    x = np.array([np.linspace(0, map_size - 1, map_size),
                  np.linspace(0, map_size - 1, map_size)]).T

    x1Mesh, x2Mesh = np.meshgrid(x[:, 0:1], x[:, 1:2])
    # Making the gaussians circular


    xPred = np.array([np.reshape(x1Mesh, (map_size * map_size,)) \
                         , np.reshape(x2Mesh, (map_size * map_size,))])
    diff = xPred - mean.reshape((mean.shape[0], 1))
    diff_ = np.linalg.norm(diff,axis=0)
    gaussian = 1 / (2 * np.pi *np.square(cov)) * np.exp(-0.5*np.square(diff_/cov))
    gaussian = (gaussian/gaussian.sum()).reshape((map_size,map_size))
    indicator = (np.max(np.abs(diff),axis=0)<cov).reshape((map_size,map_size)).astype(np.int)
    #plt.imshow((gaussian/gaussian.max()).get().reshape((336, 336)))
    #plt.show()
    return gaussian,indicator

class GPSemanticMap:

    def __init__(self, config_dict , isGroundTruth = False):
        '''
        GP Semantic Map models the necessary transitions and the updates to the sematic map based on the
        quadrotors position
        Maintains

        @param
            config_dict: Consists of the parameters for setting up tbe environment
            - num_semantics : number of semantics
            - world_map_size : tuple(int,int)
            - resolution :  tuple(int,int)
            -target_belief_thresh : int
        '''
        self.config = config_dict
        self.resolution = self.config['resolution']

        self.world_map_size = self.config['world_map_size']

        self.map_size = (self.config['world_map_size']*self.resolution,
                         self.config['world_map_size']*self.resolution,
                         self.config['num_semantics'])


        self.num_semantics = self.config['num_semantics']
        self.semantic_map = None
        self.coverage_map = None
        self.obstacle_map = None
        self.detected_semantic_map = None # current semantic category
        self.map_image = None

        self.padding = self.config['padding']

        self.isGroundTruth = isGroundTruth
        if self.isGroundTruth:
            self.semantic_proportion = None
            self.semantic_list = None

        self.centre_locations = None
        self.target_belief_thresh = self.config['target_belief_thresh']


    def get_row_col(self,pos):
        '''
        @params:
        pos: tuple, list np.array (position of agent)
        '''

        return [int(pos[0]*self.resolution),int(pos[1]*self.resolution)]

    def get_pos(self,row_col):
        '''
        @params
        row_col : the row and column to convert the position into
        '''

        return int(row_col[0]/self.resolution),int(row_col[1]/self.resolution)

    def get_observations(self, pos, fov, scale = None, type='semantic',return_distance = False,resolution =None):
        '''
        Returns the necessary observations (semantic,coverage or obstacle)
        To return the distance map based on the field of view
        @params:
        pos :  Position of the agent
        fov :  field of view in Global coordinate system
        type: type of the map
        scale: Integer
        return_distance:  returns distance mebeddings

        @ return:
        feature, distance array:
        '''

        if type=='semantic':
            map = deepcopy(self.semantic_map)
        elif type=='detected_semantic':
            map = deepcopy(self.detected_semantic_map)
        elif type=='obstacle':
            map = deepcopy(self.obstacle_map)
        else:
            map = deepcopy(self.coverage_map)

        try:
            r,c = self.get_row_col(pos)
        except:
            print("Position {}".format(pos))

        if scale is None:
            scale = 1

        if resolution is None:
            resolution  = self.resolution

        range = list(self.get_row_col(fov))

        range[0] = scale*range[0]
        range[1] = scale*range[1]

        min_x = np.max([r - range[0], 0])
        min_y = np.max([c - range[1], 0])
        max_x = np.min([r + range[0], self.map_size[0]])
        max_y = np.min([c + range[1], self.map_size[1]])

        feature = np.zeros((2 * range[0], 2 * range[1]))
        if type=='semantic':
            feature = np.zeros((2 * range[0], 2 * range[1],self.num_semantics))

        feature[min_x - (r - range[0]):2 * range[0] - (r + range[0] - max_x), \
        min_y - (c - range[1]):2 * range[1] - (c + range[1] - max_y)] = map[min_x:max_x, min_y:max_y]

        if type is not self.detected_semantic_map:
            if type =='semantic':
                feature = block_reduce(feature, (scale, scale,1), np.mean)
            else:
                feature = block_reduce(feature, (scale, scale), np.mean)
        distances = self.distances(range,scale,resolution)

            # print("{:d} {:d} {:d} {:d}".format(min_x, min_y, max_x, max_y))
        return np.array(feature),distances

    def distances(self,range,scale,resolution):

        distances_x = np.repeat(np.expand_dims(
            np.arange(2*range[1])/resolution,axis=0),
                repeats=2*range[0],axis=0)
        distances_x = np.square(distances_x - range[1]/resolution)
        distances_y = np.repeat(np.expand_dims(
            np.arange(2*range[0])/resolution,
            axis=1),repeats=2*range[1],axis=1)
        distances_y = np.square(distances_y-range[0]/resolution)

        distances = distances_x + distances_y
        distances = block_reduce(distances, (scale, scale), np.max)
        return np.sqrt(distances),np.sqrt(distances_x),np.sqrt(distances_y)

    def init_map(self, load_dict = None):

        '''
        @ params: Initializes the semantic, coverage and obstacle maps
        Assuming that the last semantic label is the background label
        '''
        # TODO detected semantic map to change accordingly,stores only the semantic label
        if load_dict is None:
            self.semantic_map = np.array(np.zeros(shape=self.map_size))
            self.coverage_map = np.array(np.zeros(shape=(self.map_size[0], self.map_size[1])))
            self.obstacle_map = np.array(np.ones(shape=(self.map_size[0], self.map_size[1])))
            self.detected_semantic_map =  np.array(np.zeros(shape=(self.map_size[0], self.map_size[1]))) - 1
        else:
            # Load the semantic map from the file path
            #self.semantic_map = np.array(np.load(load_dict['semantic_file_path']))
            self.coverage_map = np.array(np.zeros(shape=(self.map_size[0], self.map_size[1])))
            self.obstacle_map = np.array(np.ones(shape=(self.map_size[0], self.map_size[1])))

            detected_semantic_map = np.load(load_dict['semantic_file_path'])
            self.detected_semantic_map = np.zeros(shape=(self.map_size[0],self.map_size[1]))
            for j in range(self.num_semantics):
                map = np.zeros(detected_semantic_map.shape)
                map[detected_semantic_map == j] = 1.0
                map = resize(map, output_shape=(self.map_size[0], self.map_size[1]))
                self.detected_semantic_map[map>0]=j

            self.semantic_list = set(self.detected_semantic_map.reshape(-1).tolist())
            self.semantic_proportion = {j:0  for j in range(self.num_semantics)}
            for sem in self.semantic_list:
                self.semantic_proportion[sem] = np.sum(self.detected_semantic_map==sem)

            self.detected_semantic_map = np.array(rotate(self.detected_semantic_map,90))

        self.map_image = np.array(PIL.Image.open((os.getcwd() + "/"+ load_dict['map_image_file_path'])))/255
        self.map_image =np.array(resize(self.map_image, output_shape=(self.map_size[0], self.map_size[1],self.map_image.shape[-1])))
        self.obstacle_map[int(self.padding*self.resolution): int(self.resolution*(self.world_map_size- self.padding)),
                          int(self.padding*self.resolution): int(self.resolution*(self.world_map_size- self.padding))] = 0.0
        if DEBUG:
            plt.imshow(self.detected_semantic_mapnp)
            plt.figure()
            plt.imshow(self.map_image)

    def load_prior_semantics(self, params_dict=None, ground_truth_map=None):
        '''
        Assigns a prior to the semantic map based on stored data
        @params: params_dict: dictionary containing the prior parameters
                    params_dict['senamtic_test_path'] : path to the semantic test data
        '''

        if self.isGroundTruth:
            raise AttributeError

        self.semantic_map = np.array(np.load(params_dict['semantic_test_path']))
        self.detected_semantic_map = np.argmax(self.semantic_map, axis=2)
        map1 = ground_truth_map.detected_semantic_map >0
        map2 = self.detected_semantic_map >0
        match = np.sum(map1==map2)/ np.prod(ground_truth_map.detected_semantic_map.shape)
        return match

    def init_prior_semantics(self, params_dict = None,ground_truth_map = None):
        '''
        Assigns a prior to the semantic map
        @params: params_dict: dictionary containing the prior parameters
                    params_dict['randomness'] : randomness in the prior [between 0 and 0.4] [can be + or -]
                    params_dict['num_centres'] : number of gaussian centres
                    params_dict['sigma'] : sigma of the gaussian
                    params_dict['clip'] : max or min aprioir belief
        '''

        if self.isGroundTruth:
            raise AttributeError

        self.semantic_map = np.array(np.zeros(shape=self.map_size)+0.5)
        self.coverage_map = np.array(np.zeros(shape=(self.map_size[0], self.map_size[1])))
        self.obstacle_map = np.array(np.zeros(shape=(self.map_size[0], self.map_size[1])))

        if self.centre_locations is None:
            locations_x = np.arange(0,int(np.sqrt(params_dict['num_centres'])))*\
                          self.map_size[0]/np.sqrt(params_dict['num_centres']) + \
                          0.5 *self.map_size[0]/np.sqrt(params_dict['num_centres'])

            locations_y = np.arange(0, int(np.sqrt(params_dict['num_centres']))) * \
                          self.map_size[0] / np.sqrt(params_dict['num_centres']) + \
                          0.5 * self.map_size[0] / np.sqrt(params_dict['num_centres'])

            self.centre_locations = np.meshgrid(locations_x,locations_y)
            self.centre_locations = np.stack(self.centre_locations,axis = -1).reshape(-1,2).astype(np.int32)

        gauss_list = []
        indicator_list = []
        for j,loc in enumerate(self.centre_locations):
            gaussian,indicator = Gaussian(np.array(loc), params_dict['sigma'],
                                          map_size=self.map_size[0],
                                          scale=params_dict['clip'])
            gaussian = np.expand_dims(gaussian,axis=-1)
            indicator = np.expand_dims(indicator,axis=-1)

            gauss_list.append(gaussian)
            indicator_list.append(indicator)

        gauss = np.array(np.concatenate(gauss_list,axis = -1))
        indicator = np.array(np.concatenate(indicator_list,axis = -1))

        # for j in range(self.num_semantics):
        #     map_ = np.expand_dims((ground_truth_map.detected_semantic_map == j).astype(np.float),axis = -1)
        #     weights = 2*(indicator*np.repeat(map_,repeats=self.centre_locations.shape[0],axis=-1)).sum(axis=0).sum(axis=0)\
        #               /indicator.sum(axis=0).sum(axis=0)
        #     proportion = ground_truth_map.semantic_proportion[j]/np.prod(map_.shape)
        #
        #     #weights_gauss = 2*(gauss*np.repeat(map_,repeats=self.centre_locations.shape[0],axis=-1)).sum(axis=0).sum(axis=0)
        #     randomiser = np.random.random(params_dict['num_centres'])*params_dict['randomness'] - 0.5*params_dict['randomness']
        #     if proportion>0.5:
        #         weights -= 1
        #     else:
        #         weights += 1
        #
        #     randomised_weights = 0.5*weights + randomiser
        #     neg = randomised_weights<0
        #     randomised_weights += 0.5*(1+params_dict['randomness'])
        #
        #     final_weights = randomised_weights/randomised_weights.sum()
        #     final_weights[neg] = -final_weights[neg]
        #
        #     b_map = np.sum((gauss/gauss.max()*params_dict['clip']) *
        #                    np.expand_dims(np.expand_dims(final_weights,axis=0),axis=0),axis = -1)
        #
        #     self.semantic_map[:, :, j] += b_map
        #     if DEBUG:
        #         import matplotlib.pyplot as plt
        #         plt.figure()
        #         plt.imshow(self.semantic_map[:, :, j])


        # map_ = np.expand_dims((ground_truth_map.detected_semantic_map == 0).astype(np.float),axis = -1)
        # weights = 2*(indicator*np.repeat(map_,repeats=self.centre_locations.shape[0],axis=-1)).sum(axis=0).sum(axis=0)\
        #           /indicator.sum(axis=0).sum(axis=0)
        # weights -=1

        #weights_gauss = 2*(gauss*np.repeat(map_,repeats=self.centre_locations.shape[0],axis=-1)).sum(axis=0).sum(axis=0)
        # randomiser = np.random.random(params_dict['num_centres'])*params_dict['randomness'] - 0.5*params_dict['randomness']
        #
        # randomised_weights = 0.5*weights + randomiser
        # neg = randomised_weights<0
        # randomised_weights += 0.5*(1+params_dict['randomness'])
        # final_weights = randomised_weights/randomised_weights.sum()

        map_ = np.expand_dims((ground_truth_map.detected_semantic_map > 0).astype(np.float),axis = -1)
        weights = (indicator*np.repeat(map_,repeats=self.centre_locations.shape[0],axis=-1)).sum(axis=0).sum(axis=0)\
                  /indicator.sum(axis=0).sum(axis=0)
        prop = map_.sum()/np.prod(map_.shape)

        randomiser = np.random.random(params_dict['num_centres']) * params_dict['randomness'] * 2 *prop
        weights = weights + (randomiser - prop)
        weights[weights<0] = 0
        weights[weights>prop] = prop

        weights = weights/prop - 0.5

        b_map = np.sum((gauss/gauss.max()*(params_dict['clip']-0.5)) *
                       np.expand_dims(np.expand_dims(weights,axis=0),axis=0),axis = -1)
        b_map = b_map/np.abs(b_map).max()*(params_dict['clip'] - 0.5)


        for j in range(1,self.num_semantics):
            self.semantic_map[:, :, j]  += b_map
            self.semantic_map[:, :, j] = np.clip(self.semantic_map[:, :, j], 1 - params_dict['clip'],params_dict['clip'])
            self.semantic_map[:, :, j] = self.semantic_map[:, :, j] + np.random.random(self.semantic_map[:, :, j].shape)*0.001
        self.semantic_map[:, :, 0] = 1- self.semantic_map[:, :, 1:].mean(axis=-1)
        self.semantic_map[:,:,0] = np.clip(self.semantic_map[:,:,0],1-params_dict['clip'],params_dict['clip'])

        #self.detected_semantic_map = np.argmax(self.semantic_map,axis = -1)
        self.detected_semantic_map = np.zeros((self.semantic_map.shape[0],self.semantic_map.shape[1])) - 1


        self.obstacle_map = deepcopy(ground_truth_map.obstacle_map)
        if DEBUG:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.imshow(b_map)
            plt.figure()
            plt.imshow(ground_truth_map.detected_semantic_map)
            plt.figure()
            plt.imshow(self.detected_semantic_map)
            plt.figure()
            plt.imshow(self.semantic_map[:, :, 0])
            plt.figure()
            plt.imshow(self.semantic_map[:,:,1])
            plt.figure()
            plt.imshow(self.semantic_map[:, :, 2])
            plt.figure()
            plt.imshow(self.semantic_map[:, :, 3])
        # degree of randomness
        map1 = ground_truth_map.detected_semantic_map >0
        map2 = self.detected_semantic_map >0
        match = np.sum(map1==map2)/ np.prod(ground_truth_map.detected_semantic_map.shape)
        return match

    def load_prior_semantics(self, ground_truth_map,semantic_prior_map_path):
        '''
        Loads a prior to the semantic map
        @params: semantic_map_prior: semantic_prior_map: the prior map
        '''


        self.semantic_map =  np.load(semantic_prior_map_path)
        self.coverage_map  = np.zeros((self.semantic_map.shape[0],self.semantic_map.shape[1]))
        self.detected_semantic_map = np.argmax(self.semantic_map,axis = -1)
        self.obstacle_map = deepcopy(ground_truth_map.obstacle_map)

    def get_entropy(self):
        '''
        Returns the total entropy at the desired locations
        '''

        entropy = -np.sum(self.semantic_map * np.log(np.clip(self.semantic_map, 1e-7, 1)),axis = -1) + \
                  -np.sum((1-self.semantic_map) * np.log(np.clip((1-self.semantic_map), 1e-7, 1)), axis=-1)
        return entropy


    def update_semantics(self, state, measurement,sensor_params):
        '''
            Measurement sites
        '''
        # reward_coverage = 0

        sensor_max_unc = sensor_params['sensor_max_acc']
        sensor_range = sensor_params['sensor_range']
        coeff = sensor_params['sensor_decay_coeff']
        sensor_range_map = np.array(self.get_row_col(sensor_range))*1
        range = [sensor_range_map[0],sensor_range_map[1]]
        distances,_,_ = self.distances(range,scale = 1,resolution=self.resolution)

        # asseeting if the measurement shape is equivalent to the sensor shape
        assert 2*sensor_range_map[0] == measurement.shape[0]


        r,c = self.get_row_col(state)
        min_x = np.max([r - sensor_range_map[0], 0])
        min_y = np.max([c - sensor_range_map[1], 0])
        max_x = np.min([r + sensor_range_map[0], self.semantic_map.shape[0]])
        max_y = np.min([c + sensor_range_map[1], self.semantic_map.shape[1]])

        self.coverage_map[min_x:max_x,min_y:max_y] = 1.

        sensor_odds =  np.log(sensor_max_unc *(1-coeff*distances)/(1-sensor_max_unc *(1-coeff*distances)))
        sensor_neg_odds =  np.log((3 - ((sensor_max_unc) *(1-coeff*distances)))/(2 + sensor_max_unc *(1-coeff*distances)))
        semantic_map_log_odds = np.log(self.semantic_map[min_x:max_x, min_y:max_y,:]\
                                       / (1 - self.semantic_map[min_x:max_x,min_y:max_y,:])).reshape((-1,self.num_semantics))

        try:
            semantic_map_log_odds += np.expand_dims(np.array(sensor_neg_odds[min_x - (r - sensor_range_map[0]): max_x - (r - sensor_range_map[0])] \
            [min_y - (c - sensor_range_map[1]): max_y - (c - sensor_range_map[1])]).reshape(-1),axis = -1)
        except:
            print("r,c {}".format([r,c]))
            print("m_x m_y {} {} {} {}".format(min_x,max_x,min_y,max_y))
            print("m_x m_y {} {} {} {}".format(min_x - (r - sensor_range_map[0]),  max_x - (r - sensor_range_map[0]),
                                               min_y - (c - sensor_range_map[1]),  max_y - (c - sensor_range_map[1])))
            print("r1 r2 {} {}".format(sensor_range_map[0],sensor_range_map[1]))

        try:
            shape = semantic_map_log_odds.shape[0]
            measurement_slice = measurement[min_x- (r - sensor_range_map[0]): max_x - (r-sensor_range_map[0])]\
                                                        [min_y - (c - sensor_range_map[1]): max_y - (c - sensor_range_map[1])]

            semantic_map_log_odds[np.arange(shape),measurement_slice.astype(np.int32).reshape(-1)]\
                                                                += np.array(sensor_odds[min_x- (r - sensor_range_map[0]):\
                                                                                           max_x - (r-sensor_range_map[0])]\
                                                                                          [min_y - (c - sensor_range_map[1]): \
                                                                             max_y - (c - sensor_range_map[1])]).reshape(-1)

            semantic_map_log_odds[np.arange(shape),measurement_slice.astype(np.int32).reshape(-1)] \
                                                            += np.array(sensor_neg_odds[min_x- (r - sensor_range_map[0]):\
                                                                                           max_x - (r-sensor_range_map[0])]\
                                                                                          [min_y - (c - sensor_range_map[1]): \
                                                                             max_y - (c - sensor_range_map[1])]).reshape(-1)

            semantic_map_log_odds = semantic_map_log_odds.reshape(([max_x-min_x,max_y-min_y,self.num_semantics]))
        except:
            print("state {}".format(state))
            print("r,c {}".format([r,c]))
            print("m_x m_y {} {} {} {}".format(min_x,max_x,min_y,max_y))
            print("m_x m_y {} {} {} {}".format(min_x - (r - sensor_range_map[0]),  max_x - (r - sensor_range_map[0]),
                                               min_y - (c - sensor_range_map[1]),  max_y - (c - sensor_range_map[1])))
            print("r1 r2 {} {}".format(sensor_range_map[0],sensor_range_map[1]))

        self.semantic_map[min_x:max_x,min_y:max_y] =  1 / (np.exp(-semantic_map_log_odds) + 1)

        self.detected_semantic_map[min_x:max_x,min_y:max_y][np.max(self.semantic_map
                                [min_x:max_x,min_y:max_y,:],axis=-1)>self.target_belief_thresh] = \
            np.argmax(self.semantic_map[min_x:max_x,min_y:max_y][np.max(
                      self.semantic_map[min_x:max_x,min_y:max_y,:],axis=-1)>self.target_belief_thresh],axis=-1)


# TODO: New gym environment with observation structure
