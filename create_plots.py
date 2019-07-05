import numpy as np
from matplotlib import pyplot as plt
from glob import glob
from os import listdir
import re

AVG_N = 20
X_AXIS = 2000

def load_results(method_name):
    expr = r"output[0-9]+{}.txt".format(method_name)
    results_matrix = [np.loadtxt(f) for f in listdir('.') if re.search(expr, f)]
    if not results_matrix:
        return None

    for row in range(len(results_matrix)):
        results_matrix[row] = [np.mean(results_matrix[row][i:i+AVG_N]) for i in range(0,len(results_matrix[row]),AVG_N)]
        results_matrix[row] = results_matrix[row][:X_AXIS//AVG_N]

    results_matrix = np.stack(results_matrix)
    return results_matrix

def load_multiple_results(method_names):
    result_list = []
    for method in method_names:
        results = load_results(method)
        if results is not None:
            result_list.append(results)

    result_list = np.stack(result_list)
    return result_list


# Plot average of rows from a 2D matrix
def plot_from_2d_matrix(data, linestyle='-', label=''):

    # Take the means and standard deviations for the rows
    means = np.mean(data,axis=0)
    std = np.std(data,axis=0)

    # Plot means and standard deviations
    plt.plot(np.arange(0, len(means) * AVG_N, AVG_N), means, linestyle=linestyle)
    plt.fill_between(np.arange(0, len(means) * AVG_N, AVG_N), means-std, means+std,alpha=0.5)

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
    data = load_multiple_results(['','double','dueling','a3c'])
    plot_from_3d_matrix(data, labels=['DQN', 'Double DQN', 'Dueling DQN', 'Actor-Critic'])
    plt.show()
