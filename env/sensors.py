import numpy as np


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



class sensor_setter:

    @staticmethod
    def set_env(sensor_params):
        if sensor_params['type'] == 'FieldSensor':
            return FieldSensor(sensor_params)