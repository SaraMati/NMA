import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import math


class SubnetworkFinder:
    """
    Class to detect functional networks in specific regions

    Methods:

    """
    @staticmethod
    def above_percentile(cell_value, flattened_matrix, percentile):
        if math.isnan(cell_value):
            return False
        return stats.percentileofscore(flattened_matrix, cell_value) >= percentile

    @staticmethod
    def find_functional_subnetwork(adjacency_matrix, percentile=95):
        mask = np.triu(np.ones_like(adjacency_matrix, dtype=np.bool))
        upper_half = adjacency_matrix.where(mask).dropna(axis='columns', how='all').dropna(axis='rows', how='all')

        flattened_matrix = upper_half.to_numpy().flatten()
        flattened_matrix = flattened_matrix[~np.isnan(flattened_matrix)]

        transformed_upper_half = upper_half.applymap(
            lambda cell: SubnetworkFinder.above_percentile(cell, flattened_matrix, percentile))

        np.fill_diagonal(transformed_upper_half.values, False)

        # transformed_upper_half now has true when there is an 'above-threshold' connection
        # between neurons

        return transformed_upper_half.columns[(transformed_upper_half == True).any(axis=1)]

    @staticmethod
    def find_network_by_linear_correlation_full_window(activity_matrix):
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


