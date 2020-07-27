import numpy as np
import os
import pandas as pd
import requests


class SessionDataForRegion:
    """Data container class for holding the extracted data for a region"""
    def __init__(self, region):
        self.region = region
        self.activity_matrix = pd.DataFrame()
        self.activity_matrices = list()

    def set_activity_matrix(self, activity_matrix):
        self.activity_matrix = activity_matrix

    def set_activity_matrices_per_bin(self, activity_matrices):
        self.activity_matrices = activity_matrices


class CuratedDataLoader:
    """
    Class to load the data in the same way as is done in
    https://colab.research.google.com/github/NeuromatchAcademy/course-content/blob/master/projects/load_steinmetz_decisions.ipynb
    """
    session_number = 11

    # TODO: have single method where you can pick what proportion of the
    #
    @staticmethod
    def spike_trains_decision_time_per_neuron(region, decision_time=100):
        fname = CuratedDataLoader.download_data()
        alldat = np.array([])
        for j in range(len(fname)):
            alldat = np.hstack((alldat, np.load('steinmetz_part%d.npz' % j, allow_pickle=True)['dat']))

        dat = alldat[CuratedDataLoader.session_number]

        neurons_with_brain_areas = dat['brain_area']
        indexes_for_neurons_in_region = np.where(neurons_with_brain_areas == region)[0]

        # Response times per trial
        response_t = dat['response_time']
        response_t_ms = response_t * 1000

        bin_size = dat['bin_size'] * 1000  # 10ms

        # Bin in which the response occurs for all of the neurons in the region
        response_t_bins = (response_t_ms / bin_size)
        response_t_bins = np.ceil(response_t_bins)
        number_of_bins = int(decision_time / bin_size)

        spike_data = dat['spks']
        spikes_for_neurons_in_region = spike_data[indexes_for_neurons_in_region]

        df_list = [pd.DataFrame() for x in range(number_of_bins)]

        for trial in range(spikes_for_neurons_in_region.shape[1]):
            allneurons_one_trial = spikes_for_neurons_in_region[:, trial, :]

            first_index = int(response_t_bins[trial][0] - number_of_bins)
            last_index = int(response_t_bins[trial][0])

            # all of the bins
            for bin_no in range(number_of_bins):
                if allneurons_one_trial.shape[1] <= last_index:
                    df_list[bin_no][str(trial)] = np.zeros(len(allneurons_one_trial), dtype=int)
                else:
                    x = allneurons_one_trial.take(indices=
                                                  range(first_index, last_index),
                                                  axis=1)
                    df_list[bin_no][str(trial)] = x[:, bin_no].tolist()

        for df in df_list:
            df.index = indexes_for_neurons_in_region

        data = SessionDataForRegion(region)
        data.set_activity_matrices_per_bin(df_list)
        return data

    @staticmethod
    def spikes_in_decision_time_per_neuron(region, decision_time=100):
        fname = CuratedDataLoader.download_data()
        alldat = np.array([])
        for j in range(len(fname)):
            alldat = np.hstack((alldat, np.load('steinmetz_part%d.npz' % j, allow_pickle=True)['dat']))

        dat = alldat[CuratedDataLoader.session_number]

        neurons_with_brain_areas = dat['brain_area']
        indexes_for_neurons_in_region = np.where(neurons_with_brain_areas == region)[0]

        # Response times per trial
        response_t = dat['response_time']
        response_t_ms = response_t*1000

        bin_size = dat['bin_size']*1000  # 10ms

        # Bin in which the response occurs for all of the neurons in the region
        response_t_bins = (response_t_ms/bin_size) 
        response_t_bins = np.ceil(response_t_bins)
        number_of_bins = decision_time/bin_size

        spike_data = dat['spks']
        spikes_for_neurons_in_region = spike_data[indexes_for_neurons_in_region]

        activity_matrix = pd.DataFrame()
        for trial in range(spikes_for_neurons_in_region.shape[1]):
            allneurons_one_trial = spikes_for_neurons_in_region[:, trial, :]

            first_index = int(response_t_bins[trial][0] - number_of_bins)
            last_index = int(response_t_bins[trial][0])
            if allneurons_one_trial.shape[1] <= last_index:
                activity_matrix[str(trial)] = np.zeros(len(allneurons_one_trial), dtype=int)
            else:
                x = allneurons_one_trial.take(indices=
                                              range(first_index, last_index),
                                              axis=1)
                spike_counts_for_this_trial = np.sum(x, axis=1)
                activity_matrix[str(trial)] = spike_counts_for_this_trial.tolist()

        activity_matrix.index = indexes_for_neurons_in_region
        data = SessionDataForRegion(region)
        data.set_activity_matrix(activity_matrix)
        return data

    @staticmethod
    def neurons_with_total_spike_count(region):

        fname = CuratedDataLoader.download_data()
        alldat = np.array([])
        for j in range(len(fname)):
            alldat = np.hstack((alldat, np.load('steinmetz_part%d.npz' % j, allow_pickle=True)['dat']))

        dat = alldat[CuratedDataLoader.session_number]

        neurons_with_brain_areas = dat['brain_area']
        indexes_for_neurons_in_region = np.where(neurons_with_brain_areas == region)[0]

        spike_data = dat['spks']
        spikes_for_neurons_in_region = spike_data[indexes_for_neurons_in_region]
        spike_counts_per_neuron = np.count_nonzero(spikes_for_neurons_in_region, axis=2)
        spike_counts_per_neuron_across_all_trials = np.sum(spike_counts_per_neuron, axis=1)

        counts = pd.Series(spike_counts_per_neuron_across_all_trials, index=indexes_for_neurons_in_region)
        return counts

    @staticmethod
    def download_data():
        fname = []
        for j in range(3):
            fname.append('steinmetz_part%d.npz' % j)
        url = ["https://osf.io/agvxh/download"]
        url.append("https://osf.io/uv3mw/download")
        url.append("https://osf.io/ehmw2/download")

        for j in range(len(url)):
            if not os.path.isfile(fname[j]):
                try:
                    r = requests.get(url[j])
                except requests.ConnectionError:
                    print("!!! Failed to download data !!!")
                else:
                    if r.status_code != requests.codes.ok:
                        print("!!! Failed to download data !!!")
                    else:
                        with open(fname[j], "wb") as fid:
                            fid.write(r.content)
        return fname