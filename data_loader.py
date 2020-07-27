import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from oneibl.onelight import ONE


class DataLoader:
    """
    Class to load the data using ONE
    """
    steinmetz_url = "https://figshare.com/articles/steinmetz/9974357"

    def __init__(self, session_number=11):
        self.one = self.__initialise_one_session()
        self.session_number = session_number

    def retrieve_spike_time_matrix_per_region(self, region_name):
        sessions = self.one.search(['spikes'])
        session = sessions[self.session_number]

        channels = self.one.load_object(session, 'channels')
        brain_locations = pd.DataFrame(channels.brainLocation)

        # Find all of the channels for the named region
        matching_channels = brain_locations[brain_locations['allen_ontology'] == region_name]
        if matching_channels.empty:
            raise Exception("No channels with brain location: " + region_name)
        # Row indexes of these matching channels in the original brain_locations frame
        ids = matching_channels.index

        clusters = self.one.load_object(session, 'clusters')
        cluster_channels = pd.DataFrame(clusters.peakChannel[clusters._phy_annotation>=2])

        # Taking the values inside cluster_channels to link back to the row indexes in brain_locations
        indexes_of_matching_clusters = cluster_channels[cluster_channels.isin(ids)].dropna().index

        # Use these indexes to find the spikes
        spike_data = self.one.load_object(session, 'spikes')
        spike_clusters = pd.DataFrame(spike_data.clusters)
        # All of the spikes for this region
        spikes = spike_clusters[spike_clusters.isin(indexes_of_matching_clusters)].dropna()

        spikes.columns = {'cluster'}  # rename column; each row is a spike, with its column denoting its cluster
        cluster_spike_counts = spikes.groupby('cluster').cluster.count()

        return cluster_spike_counts

    @staticmethod
    def __initialise_one_session():
        one = ONE()
        one.set_figshare_url(DataLoader.steinmetz_url)
        return one
