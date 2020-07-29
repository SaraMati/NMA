import numpy as np
import os
import csv
import pandas as pd
import requests
from dataclasses import dataclass


class SessionDataForRegion:
    """
    Data container class for holding the extracted data for a region
    Easier to have this as output so whenever we pass around the output (e.g. adjacency matrix)
    , we'll know which region and session it's from
    """

    @dataclass
    class SessionInfo:
        index_in_curated_data: int = -1
        mouse_name: str = "UNKNOWN"
        session_date: str = "UNKNOWN"

    def __init__(self, region):
        self.region = region
        self.session = SessionDataForRegion.SessionInfo()
        self.activity_matrix = pd.DataFrame()
        self.activity_matrices = list()

    def set_activity_matrix(self, activity_matrix):
        self.activity_matrix = activity_matrix

    def set_activity_matrices_per_bin(self, activity_matrices):
        self.activity_matrices = activity_matrices

    def set_session_info(self, info):
        self.session = info

    def session(self):
        return self.session.mouse_name + "_" + self.session.session_date


class CuratedDataLoader:
    """
    Class to load the data in the same way as is done in
    https://colab.research.google.com/github/NeuromatchAcademy/course-content/blob/master/projects/load_steinmetz_decisions.ipynb
    """
    class DataForRegion:
        """
        Data holder class for use during data extraction

        Not for external use
        """
        def __init__(self,
                     indexes_for_neurons,
                     response_bins_for_neurons,
                     spikes_for_neurons):
            self.indexes_for_neurons = indexes_for_neurons
            self.response_bins_for_neurons = response_bins_for_neurons
            self.spikes_for_neurons = spikes_for_neurons

    def __init__(self, session_number=11):
        self.bin_size_ms = -1
        self.number_of_bins_for_analysis = -1
        self.session_number = session_number

    @staticmethod
    def neuron_id_to_cell_type(neuron_ids, path_to_cell_type_csv):
        cell_type_df = pd.read_csv(path_to_cell_type_csv)
        return cell_type_df[cell_type_df.index.isin(neuron_ids)]

    def spike_trains_decision_time_per_neuron(self, region, downloaded_data, decision_time=100):
        neuron_data, session_info = self.load_session(region, downloaded_data, decision_time)

        df_list = [pd.DataFrame() for x in range(self.number_of_bins_for_analysis)]

        for trial in range(neuron_data.spikes_for_neurons.shape[1]):
            allneurons_one_trial = neuron_data.spikes_for_neurons[:, trial, :]

            first_index = int(neuron_data.response_bins_for_neurons[trial][0] - self.number_of_bins_for_analysis)
            last_index = int(neuron_data.response_bins_for_neurons[trial][0])

            for bin_no in range(self.number_of_bins_for_analysis):
                if allneurons_one_trial.shape[1] <= last_index:
                    df_list[bin_no][str(trial)] = np.zeros(len(allneurons_one_trial), dtype=int)
                else:
                    x = allneurons_one_trial.take(indices=
                                                  range(first_index, last_index),
                                                  axis=1)
                    df_list[bin_no][str(trial)] = x[:, bin_no].tolist()

        for df in df_list:
            df.index = neuron_data.indexes_for_neurons

        data = SessionDataForRegion(region)
        data.set_activity_matrices_per_bin(df_list)
        data.set_session_info(session_info)
        return data

    def spikes_in_decision_time_per_neuron(self, region, downloaded_data, decision_time=100):
        neuron_data, session_info = self.load_session(region, downloaded_data, decision_time)

        activity_matrix = pd.DataFrame()
        for trial in range(neuron_data.spikes_for_neurons.shape[1]):
            allneurons_one_trial = neuron_data.spikes_for_neurons[:, trial, :]

            first_index = int(neuron_data.response_bins_for_neurons[trial][0] - self.number_of_bins_for_analysis)
            last_index = int(neuron_data.response_bins_for_neurons[trial][0])
            if allneurons_one_trial.shape[1] <= last_index:
                activity_matrix[str(trial)] = np.zeros(len(allneurons_one_trial), dtype=int)
            else:
                x = allneurons_one_trial.take(indices=
                                              range(first_index, last_index),
                                              axis=1)
                spike_counts_for_this_trial = np.sum(x, axis=1)
                activity_matrix[str(trial)] = spike_counts_for_this_trial.tolist()

        activity_matrix.index = neuron_data.indexes_for_neurons
        data = SessionDataForRegion(region)
        data.set_activity_matrix(activity_matrix)
        data.set_session_info(session_info)
        return data

    def neurons_with_total_spike_count(self, region, downloaded_data):
        dat = downloaded_data[self.session_number]

        neurons_with_brain_areas = dat['brain_area']
        indexes_for_neurons_in_region = np.where(neurons_with_brain_areas == region)[0]

        spike_data = dat['spks']
        spikes_for_neurons_in_region = spike_data[indexes_for_neurons_in_region]
        spike_counts_per_neuron = np.count_nonzero(spikes_for_neurons_in_region, axis=2)
        spike_counts_per_neuron_across_all_trials = np.sum(spike_counts_per_neuron, axis=1)

        counts = pd.Series(spike_counts_per_neuron_across_all_trials, index=indexes_for_neurons_in_region)
        return counts

    def load_session(self, region, downloaded_data, decision_time):
        dat = downloaded_data[self.session_number]

        neurons_with_brain_areas = dat['brain_area']
        indexes_for_neurons_in_region = np.where(neurons_with_brain_areas == region)[0]

        # Response times per trial
        response_t = dat['response_time']
        response_t_ms = response_t * 1000

        self.bin_size_ms = dat['bin_size'] * 1000  # 10ms

        # Bin in which the response occurs for all of the neurons in the region
        response_t_bins = (response_t_ms / self.bin_size_ms)
        response_t_bins = np.ceil(response_t_bins)
        self.number_of_bins_for_analysis = decision_time / self.bin_size_ms

        spike_data = dat['spks']
        spikes_for_neurons_in_region = spike_data[indexes_for_neurons_in_region]

        info = SessionDataForRegion.SessionInfo()
        info.index_in_curated_data = self.session_number
        info.mouse_name = dat['mouse_name']
        info.session_date = dat['date_exp']

        return CuratedDataLoader.DataForRegion(indexes_for_neurons_in_region,
                                               response_t_bins,
                                               spikes_for_neurons_in_region), info

    @staticmethod
    def download_data():
        fnames = []
        for j in range(3):
            fnames.append('steinmetz_part%d.npz' % j)
        url = ["https://osf.io/agvxh/download"]
        url.append("https://osf.io/uv3mw/download")
        url.append("https://osf.io/ehmw2/download")

        for j in range(len(url)):
            if not os.path.isfile(fnames[j]):
                try:
                    r = requests.get(url[j])
                except requests.ConnectionError:
                    print("!!! Failed to download data !!!")
                else:
                    if r.status_code != requests.codes.ok:
                        print("!!! Failed to download data !!!")
                    else:
                        with open(fnames[j], "wb") as fid:
                            fid.write(r.content)
         # TODO: create the giant array in here because that's the slow part so only want to do once
        alldat = np.array([])
        for j in range(len(fnames)):
            alldat = np.hstack((alldat, np.load('steinmetz_part%d.npz' % j, allow_pickle=True)['dat']))
        return alldat

    @staticmethod
    def indexes_of_sessions_for_mouse(mouse_name, downloaded_data):
        index = 0
        matching_indexes = list()
        for session in downloaded_data:
            if session['mouse_name'] == mouse_name:
                matching_indexes.append(index)
            index += 1

        return matching_indexes

