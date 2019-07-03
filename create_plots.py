import numpy as np
from matplotlib import pyplot as plt

def plot_from_2d_matrix(data):
    means = np.mean(data,axis=0)
    std = np.std(data,axis=0)
    plt.plot(means)
    plt.fill_between(range(len(means)),means-std,means+std,alpha=0.5)

def plot_from_3d_matrix(data):
    for i in range(data.shape[0]):
        plot_from_2d_matrix(data[i,:,:])

if __name__ == "__main__":
    data = np.random.rand(2,10,100)
    plot_from_3d_matrix(data)
    plt.show()
