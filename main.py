from APF import APF
import numpy as np

start = np.array([[2, 2]])
target = np.array([[8, 8]])
obs = np.array([
    [2, 3, 0.6],
    [3, 7, 0.8],
    [5, 5, 0.7],
    [6, 7, 0.5],
    [7, 3, 0.4],
    [8, 6, 0.5]
])
step = 0.01
max_count = 3000
k_attr = 1
k_rep = 10
k_rad = 3

apf = APF(start, target, obs, step, max_count, k_attr, k_rep, k_rad)
apf.run()
apf.plot()
