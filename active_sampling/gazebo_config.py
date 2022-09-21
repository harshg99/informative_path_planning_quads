# Config file for obstacles
# Each obstacle should have a correspinding sdf files in the ~/.gazebo/models folder

import json
import numpy as np

# Use pcg_gazebo for procedurally generating the sdf file
#from ruamel.yaml import YAML as yaml
from pdb import set_trace as T
import yaml
import matplotlib.pyplot as plt
min_var = 3.0
max_var = 6.0

class Models:

	models = 'Obstacle Type' # folder correspinding to sdf files in the ~/.gazebo/models folder
	poses = [0,0,0] # Tuple defined in json dile

	def __init__(self,data:dict):
		for key in data.keys():
			setattr(self,key,data[key])

class file(yaml.YAMLObject):
    yaml_loader = yaml.SafeLoader
    yaml_tag = '!file'
    def __init__(self, val):
        self.val = val

    @classmethod
    def from_yaml(cls, loader, node):
        return cls(node.value)

def Gaussian(mean, cov,map_size):
    x = np.array([np.linspace(0, map_size - 1, map_size),
                  np.linspace(0, map_size - 1, map_size)]).T

    x1Mesh, x2Mesh = np.meshgrid(x[:, 0:1], x[:, 1:2])
    # Making the gaussians circular

    cov1 = np.copy(cov)
    cov[0][0] = np.clip(cov[0][0], min_var, max_var)
    cov[1][1] = np.clip(cov[1][1], min_var, max_var)
    cov[0][1] = np.clip(cov[0][1], 0, min_var / 2)
    cov[1][0] = cov[0][1]

    if np.linalg.det(cov) < 0:
        cov[0][0] = np.clip(cov1[0][1], min_var, max_var)
        cov[1][1] = np.clip(cov1[1][0], min_var, max_var)
        cov[0][1] = np.clip(cov1[1][1], min_var / 2, max_var / 2)
        cov[1][0] = cov[0][1]
    elif np.linalg.det(cov) == 0:
        cov[0][0] = np.clip(cov1[0][1] + np.random.rand(1), min_var, max_var)
        cov[1][1] = np.clip(cov1[1][0] + np.random.rand(1), min_var, max_var)

    xPred = np.array([np.reshape(x1Mesh, (map_size * map_size,)) \
                         , np.reshape(x2Mesh, (map_size * map_size,))])
    gaussian = np.diag(1 / np.sqrt(2 * np.pi * np.abs(np.linalg.det(cov))) * \
                       np.exp(-(xPred - mean.reshape((mean.shape[0], 1))).T @ np.linalg.inv(cov) @ \
                              (xPred - mean.reshape((mean.shape[0], 1)))))
    gaussian = gaussian.reshape((map_size, map_size))
    return gaussian

class GazeboModel:

	def __init__(self,filename:str):
		'''
		filename: config json file to read objects
		worldfilename: world file to output
		'''
		self.filename = filename

	def read_yaml(self):
		with open(self.filename) as file:
			#yaml_obj=yaml(typ="safe")
			return yaml.safe_load(file)

	def import_objects(self):
		model_list =  self.read_yaml()['engines']
		objects = []
		#T()
		for obj in model_list:
			if obj['engine_name'] =='fixed_pose':
				if obj['models'][0] != 'ground_plane' and obj['models'][0] != 'sun':
					infodict = {}
					infodict['models'] = obj['models'][0]
					infodict['poses'] = obj['poses']
					objects.append(Models(infodict))
		return objects

	def generate_map(self,objects:list,resolution:int,x_bounds:tuple,y_bounds:tuple):
		
		num_bins = (int((x_bounds[1]-x_bounds[0])/ resolution) +1, int((y_bounds[1]-y_bounds[0]) / resolution) +1)
		interest_array = np.zeros(num_bins)

		
		origin = [0,0]

		# X axis is columns , Y axis is rows

		map_origin = [int((origin[1]-y_bounds[0])/resolution)+1,int((origin[1]- x_bounds[0])/resolution) + 1]
		for obj in objects:
			for pose in obj.poses:
				r_pos = pose[0]
				c_pos = pose[1]
				r_pos_map = int((c_pos - y_bounds[0])/resolution) + 1
				c_pos_map = int((r_pos - x_bounds[0])/resolution) + 1
				interest_array += Gaussian(np.array([c_pos_map,r_pos_map]),\
                                       np.array([[6.0,0.0],[0.0,6.0]]),num_bins[0])

		interest_array/=interest_array.sum()
		return interest_array,num_bins


if __name__=="__main__":
	loader = GazeboModel('randy.yml')


	#TODO: Cleanup based on extrinsic parameters
	x_bounds = [-15,15]
	y_bounds = [-15,15]

	resolution = 0.5
	#T()
	
	objects = loader.import_objects()
	maps,num_bins = loader.generate_map(objects,resolution,x_bounds,y_bounds)
	plt.figure()
	plt.imshow(maps, origin='lower', extent=[0, resolution*num_bins[0], 0, resolution*num_bins[1]])

	np.save("interest_array.npy",maps)
	plt.show()