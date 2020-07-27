from curated_data_loader import *
from subnetwork_finder import *

if __name__ == "__main__":

    region = "VISam"
    extracted_data = CuratedDataLoader.spikes_in_decision_time_per_neuron(region)
    finder = SubnetworkFinder()
    adjacency_matrix = finder.find_network_by_linear_correlation(extracted_data.activity_matrix)
    finder.create_heatmap_from_adjacency_matrix(region, adjacency_matrix)
