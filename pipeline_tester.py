from curated_data_loader import *
from subnetwork_finder import *

if __name__ == "__main__":

    region = "VISam"
    extracted_data = CuratedDataLoader.spikes_in_decision_time_per_neuron(region)
    """data = CuratedDataLoader.spike_trains_decision_time_per_neuron(region)

    finder = SubnetworkFinder()

    adjacency_matrices_all_bins = list()
    for activity_for_bin in data.activity_matrices:
        adjacency_matrices_all_bins.append(finder.find_network_by_linear_correlation_full_window(activity_for_bin))

    averaged_across_bins = sum(adjacency_matrices_all_bins)/len(data.activity_matrices)"""

    finder = SubnetworkFinder()
    adjacency_matrix = finder.find_network_by_linear_correlation_full_window(extracted_data.activity_matrix)
    finder.create_heatmap_from_adjacency_matrix(region, adjacency_matrix)

    columns = finder.find_functional_subnetwork(adjacency_matrix)
    # TODO: Read in the CSV with the cell types & visualise:
    # 1) what types of cells are in each network how does that vary per region
    # 2) visualise the actual networks with weighted edges and annotated with cell type

    # TODO: other measures of the graph - clustering coefficient? characteristic path length?
