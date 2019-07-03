import numpy as np
from matplotlib import pyplot as plt

def plot_from_2d_matrix(data, linestyle='-', label=''):
    means = np.mean(data,axis=0)
    std = np.std(data,axis=0)
    plt.plot(means,linestyle=linestyle)
    plt.fill_between(range(len(means)),means-std,means+std,alpha=0.5)

    plt.title(label)
    plt.xlabel('Number of training epochs')
    plt.ylabel('Reward')

def plot_from_3d_matrix(data, labels=None):
    linestyle = ['-', '--', '-.', ':']
    for i in range(data.shape[0]):
        plot_from_2d_matrix(data[i,:,:],linestyle[i])

    if labels:
        plt.legend(labels)
    
    plt.title('Average training reward over {} runs'.format(data.shape[1]))

if __name__ == "__main__":
    data = np.random.rand(4,10,100)
    plot_from_3d_matrix(data,labels=['DQN', 'Double DQN', 'Dueling DQN', 'Actor-Critic'])
    plt.show()
