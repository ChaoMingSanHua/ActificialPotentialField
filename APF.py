import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class APF(object):
    def __init__(self, start, target, obs, step, max_count, k_attr, k_rep, k_rad):
        self.__start = start
        self.__target = target
        self.__obs = obs
        self.__step = step
        self.__max_count = max_count
        self.__k_attr = k_attr
        self.__k_rep = k_rep
        self.__k_rad = k_rad
        self.__obs_v = np.zeros((self.__obs.shape[0], 2))
        self.__rob = self.__start
        self.__result = None
        self.__obs_p = None

    def run(self):
        result = np.zeros((self.__max_count, 2))
        obs_p = np.zeros((self.__max_count, self.__obs.shape[0], 2))

        count = 0
        while np.linalg.norm(self.__rob - self.__target) > self.__step and count < self.__max_count:
            force_attr = self.__attractive_force()
            force_req = self.__repulsive_force()
            force = force_attr + force_req
            result[count, :] = self.__rob
            move = self.__step * force / np.linalg.norm(force)
            self.__rob = self.__rob + move
            obs_p[count, :, :] = self.__obs[:, :2]
            if count % 50 == 0:
                self.__obs_v = (np.random.random((self.__obs.shape[0], 2)) - 0.5) / 100
            self.__obs[:, :2] = self.__obs[:, :2] + self.__obs_v
            count += 1
        result[count, :] = self.__target
        obs_p[count, :, :] = self.__obs[:, :2]
        self.__result = result[:count + 1, :]
        self.__obs_p = obs_p[:count + 1, :, :]

    def plot(self):
        def update(i):
            plt.cla()
            plt.scatter(self.__result[i, 0], self.__result[i, 1], c="blue")
            for j in range(self.__obs.shape[0]):
                circle = plt.Circle((self.__obs_p[i, j, 0], self.__obs_p[i, j, 1]), self.__obs[j, 2], color="red",
                                    fill=True)
                plt.gcf().gca().add_artist(circle)
            plt.scatter(self.__start[0, 0], self.__start[0, 1], c="green", marker=".")
            plt.scatter(self.__target[0, 0], self.__target[0, 1], c="green", marker="*")
            plt.axis('equal')
            plt.xlim(0, 10)
            plt.ylim(0, 10)

        fig = plt.figure(1)
        ani = animation.FuncAnimation(fig=fig, func=update, frames=np.arange(0, self.__result.shape[0]), interval=1)
        plt.show()

    def __attractive_force(self):
        force = self.__k_attr * (self.__target - self.__rob)
        return force

    def __repulsive_force(self):
        force = np.zeros((1, 2))
        for i in range(self.__obs.shape[0]):
            rho = np.linalg.norm(self.__obs[i, :2] - self.__rob)
            rho_0 = self.__obs[i, 2] * self.__k_rad
            if rho > rho_0:
                continue
            k = self.__k_rep * (1 / rho - 1 / rho_0) * (1 / rho ** 2)
            force += k * (self.__rob - self.__obs[i, :2]) / rho
        return force
