import numpy as np
from matplotlib import pyplot as plt

# Plot average of rows from a 2D matrix
def plot_from_2d_matrix(data, linestyle='-', label=''):

    # Take the means and standard deviations for the rows
    means = np.mean(data,axis=0)
    std = np.std(data,axis=0)

    # Plot means and standard deviations
    plt.plot(means,linestyle=linestyle)
    plt.fill_between(range(len(means)),means-std,means+std,alpha=0.5)

    # Add title and axis labels
    plt.title(label)
    plt.xlabel('Number of training epochs')
    plt.ylabel('Reward')

# Plot average of rows from multiple 2D matrices
def plot_from_3d_matrix(data, labels=None):

    # Use different line style for each 2d matrix
    linestyle = ['-', '--', '-.', ':']

    # Plot all 2d matrices
    for i in range(data.shape[0]):
        plot_from_2d_matrix(data[i,:,:],linestyle[i])

    # Add legend
    if labels:
        plt.legend(labels)
    
    # Add title
    plt.title('Average training reward over {} runs'.format(data.shape[1]))

# Plot from random noise for testing purposes
if __name__ == "__main__":
    data = np.random.rand(4,10,100)
    plot_from_3d_matrix(data,labels=['DQN', 'Double DQN', 'Dueling DQN', 'Actor-Critic'])
    plt.show()
