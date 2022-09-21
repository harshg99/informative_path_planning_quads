import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure



num_targets = [10,30]
defaultBelief = 0.05
targetBeliefThresh = 0.95
sensor_range = 5
coeff = 0.1
sensor_unc = np.zeros([sensor_range,sensor_range])

for j in range(sensor_range):
    for k in range(sensor_range):
        dist = max(np.abs(j-int((sensor_range/2))),np.abs(k-int((sensor_range/2))))
        sensor_unc[j][k] = 0.01 + coeff*dist

range = np.arange(0,1000)*0.005
data_correct = 0.99 - coeff*range
data_correct[range>2.5] = 0.5
data_incorrect = 1 - data_correct

figure(figsize=(4, 3), dpi=80)
plt.plot(range,data_correct,label='Correct Classification')
plt.plot(range,data_incorrect,label='Incorrect Classification')
plt.xlabel('Distance (m)',fontsize=12)
plt.ylabel('Likelihood of prediction',fontsize=12)
plt.title('Sensor Model')
plt.legend()
plt.show()

