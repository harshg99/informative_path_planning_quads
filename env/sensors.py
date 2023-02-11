import numpy as np
import functools

class FieldSensor:
    def __init__(self,sensor_params):
        self.sensor_unc = sensor_params['sensor_unc']
        self.sensor_range = self.sensor_unc.shape[-1]

    def getMeasurement(self,pos,worldTargetMap):
        sensorMeasurement = np.zeros([self.sensor_range,self.sensor_range])
        r = pos[0]
        c = pos[1]
        range_ = int(self.sensor_range / 2)
        min_x = np.max([r - range_, 0])
        min_y = np.max([c - range_, 0])
        max_x = np.min([r + range_ + 1, worldTargetMap.shape[0]])
        max_y = np.min([c + range_ + 1, worldTargetMap.shape[1]])
        groundTruth = np.zeros([self.sensor_range, self.sensor_range])
        groundTruth[min_x - (r - range_):self.sensor_range - (r + range_ + 1 - max_x), \
        min_y - (c - range_):self.sensor_range - (c + range_ + 1 - max_y)] = \
            worldTargetMap[min_x:max_x, min_y:max_y].copy()
        for j in range(self.sensor_range):
            for k in range(self.sensor_range):
                sensorMeasurement[j, k] = np.random.choice(a=np.array([groundTruth[j, k], 1 - groundTruth[j, k]]),\
                                                           p=np.array([1 - self.sensor_unc[j, k], self.sensor_unc[j, k]]))
        return sensorMeasurement



class SemanticSensor:
    def __init__(self,sensor_params):

        self.sensor_max_unc = sensor_params['sensor_max_acc']
        self.sensor_range = sensor_params['sensor_range']
        self.coeff = sensor_params['sensor_decay_coeff']


    def get_measurements(self,pos, groundTruth):

        ground_truth_semantic,distances = groundTruth.get_observations(pos,self.sensor_range,type='detected_semantic')


        # Measurements
        sensor_measurement = np.zeros( ground_truth_semantic.shape)

        # Out of map bounds, no measurements
        sensor_measurement[ground_truth_semantic==-1] = -1


        prob_matrix = np.repeat(np.expand_dims((1 - self.sensor_max_unc *(1-self.coeff*distances[0]))/
                                               ( groundTruth.num_semantics - 1),axis =-1),axis=-1,repeats=groundTruth.num_semantics)
        prob_matrix = prob_matrix.reshape((-1,groundTruth.num_semantics))

        prob_matrix[np.arange(prob_matrix.shape[0]),
                            ground_truth_semantic.astype(np.int32)[ground_truth_semantic>=0]]\
                            = self.sensor_max_unc * (1 - self.coeff * distances[0]).reshape(-1)
        # prob_matrix[ ground_truth_semantic==-1][0] == self.sensor_max_unc * (1-self.coeff*distances)

        list_matrix = [np.repeat(np.expand_dims(np.arange(groundTruth.num_semantics),axis=0),axis=0,repeats=np.prod(ground_truth_semantic.shape)),
                       prob_matrix]
        matrix = np.concatenate(list_matrix,axis=-1)
        measurement_flatten = np.apply_along_axis(functools.partial(self.sample,num_semantics = groundTruth.num_semantics)
                                                   ,axis = 1, arr=matrix)

        # measurement_flatten = np.random.choice(a=np.repeat(np.arange(groundTruth.num_semantics),
        #                                                    repeats=np.array(ground_truth_semantic.shape).prod()),\
        #                                                    p=prob_matrix)

        sensor_measurement = measurement_flatten.reshape(list( ground_truth_semantic.shape))

        return sensor_measurement

    def sample(self,array,num_semantics = 4):
        return np.random.choice(a=array[:num_semantics:],p=array[num_semantics:])

class sensor_setter:

    @staticmethod
    def set_env(sensor_params):
        if sensor_params['type'] == 'FieldSensor':
            return FieldSensor(sensor_params)
        elif  sensor_params['type'] == 'SemanticSensor':
            # TODO: Define semantic sensor
            return SemanticSensor(sensor_params)