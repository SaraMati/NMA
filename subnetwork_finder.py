import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class SubnetworkFinder:
    """
    Class to detect functional networks in specific regions

    Methods:

    """

    @staticmethod
    def find_network_by_linear_correlation(activity_matrix):
        """
        Method to find the neurons in this area which are correlated in
        terms of activity - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3586814/

        'Measure of degree to which trial-to-trial fluctuations in response strength
        are shared by a pair of neurons - Pearson correlation of their spike count responses'

        Population Pearson correlation coefficient
        cov(X, Y)/(stdX*sddY), where the X and Y are the inner lists of spike times

        :param activity_matrix:
        [
            [list of spike counts per trial] #Neuron 1 - each trial is a feature
            [list of spike counts per trial] #Neuron 2
            .
            .
            .
            [list of spike counts per trial] #Neuron n
        ]

        :return: adjacency_matrix: result of pair-wise correlations between their activities

        Matrix of dimensions: number_neurons x number_neurons - length of activity_matrix
        """

        adjacency_matrix = activity_matrix.T.corr()

        return adjacency_matrix

    @staticmethod
    def create_heatmap_from_adjacency_matrix(region, adjacency_matrix):
        sns.heatmap(adjacency_matrix,
                    fmt='.1g',
                    vmin=-1, vmax=1, center=0,
                    cmap='coolwarm',
                    yticklabels=True, xticklabels=True).set_title(region)
        plt.show()


