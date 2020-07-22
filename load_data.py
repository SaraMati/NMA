import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

from oneibl.onelight import ONE
one = ONE()

one.set_figshare_url("https://figshare.com/articles/steinmetz/9974357")
sessions = one.search(['spikes'])
session = sessions[11]
one.list(session)

# loads all the objects from the session
trials = one.load_object(session, 'trials')
spikes = one.load_object(session, 'spikes')
clusters = one.load_object(session, 'clusters')
channels = one.load_object(session, 'channels')
probes = one.load_object(session, 'probes')

'''
See https://github.com/nsteinme/steinmetz-et-al-2019/wiki/data-files for description of dataset

Session 11 has:
channels: 1122
clusters: 1219
spikes: 11,757,530

trial timeline:
1. mouse holds wheel still for a short interval (0.2-0.5s)
2. trial initiates with stimulus onset
3. after stimulus onset, there is random delay interval of 0.5-1.2s
4. at end of delay interval, auditory tone cue is delivered (Go cue)
5. in the following 1.5s, mouse moves wheel (or doesn't, if no stimulus presented) 
6. feedback is delivered at the end of 1.5s, for 1s duration
7. 1s inter-trial interval
8. mouse can initiate another trial by holding the wheel still

windows of interest for firing rates:
- baseline: -0.2 to 0s relative to stimulus onset
- trial firing rate: stimulus onset to 400 ms post-stimulus
- stimulus-driven: 0.05 to 0.15s after stimulus onset
- pre-movement FR: -0.1 to 0.05s relative to movement onset
- post-movement FR: -0.05 to 0.2s relative to movement onset
- post-reward rate: 0 to 0.15s relative to reward delivery for correct NoGos

logic for chaining indices:

1. channels.brainLocation = lists all the channels and their corresponding brain region. Each row here
    is a unique channel ID (3x383 probes, indexed from 0 = 1152 channels)
2. so we extract vis_cx_channels_id to get the channel ids (row #) with specific brain regions
3. clusters.peakChannel = lists all the channel numbers for each cluster
4. we extract vis_cx_clusters_id to get the cluster ids with channel numbers matching vis_cx_channels_id
5. spikes.clusters = lists the cluster number for each spike
6. so we can extract vis_cx_spikes to get the spike IDs with cluster IDs matching vis_cx_channels_id
'''

# get event times
response_times = trials.response_times
visual_stim_times = trials.visualStim_times
gocue = trials.goCue_times
feedback = trials.feedback_times

# getting times relative to visual stimulus onset
response_times = response_times - visual response_times
feedback = feedback - visual_stim_times
gocue = gocue - visual_times

# defining trials
dt = 1/100 # bin size
dT = 2.5 # total time in trial / length of trial
T0 = .5 # stimulus onset within a trial / time of trial start
trial_onset_times = visual_stim_times - T0
n_trials = len(trial_onset_times) # number of trials in this session

# finding the first spikes within a trial, totally stolen
# basically keeps shrinking the window until the first spike after trial onset is identified
def first_spikes(spike_times, trial_onset_times):
    tlow = 0
    thigh = len(spike_times)

    while thigh>tlow+1:
        thalf = (thigh + tlow)//2
        sthalf = spike_times[thalf]
        if trial_onset_times >= sthalf:
            tlow = thalf
        else:
            thigh = thalf
    return thigh


# PSTH for spikes within the trials
# makes an array of n_clusters arrays, n_trials rows, dT/dt columns
n_clusters = np.max(spikes.clusters)+1
time_bins = int(dT/dt)

spikes_trials = np.zeros((n_clusters, ntrials, time_bins))
for trial in range(ntrials):
    first_spk = first_spikes(spike.times, trial_onset_times[trial])
    last_spk = first_spikes(spike.times, trial_onset_times[trial]+dT)
    trial_spk_times = spike_times[first_spk:last_spk] - trial_onset_times[trial]
    trial_clusters = spikes.cluster[first_spk:last_spk]
    spikes_trials[:, trial, :] = 




# pull out channels from regions of interest (Vis cortex here)
regions = "VIS"
brain_locations = pd.DataFrame(channels.brainLocation)
vis_cx_channels = brain_locations[brain_locations['allen_ontology'].str.contains(regions)]
vis_cx_channels_id = vis_cx_channels.index  #gets the channel ID

# clusters.peakChannel contains the channel number of the location of the peak of the cluster's waveform
# thresholding _phy_annotation >= 2 to only count clusters from a single neuron
# pull out clusters from regions of interest using channel ID
# gets the indices/IDs of clusters from the region of interest
cluster_channels = pd.DataFrame(clusters.peakChannel[clusters._phy_annotation>=2])
vis_cx_clusters_id = cluster_channels[cluster_channels.isin(vis_cx_channels_id)].dropna().index

# for example, to get waveform durations of all peaks from vis cx clusters:
vis_cx_waveformDuration = clusters.waveformDuration[vis_cx_clusters_id]

# pull out spikes from regions of interest using cluster ID
# gets the spike IDs of spikes from the region of interest
spike_clusters = pd.DataFrame(spikes.clusters)
vis_cx_spikes = spike_clusters[spike_clusters.isin(vis_cx_clusters_id)].dropna()
vis_cx_spikes.columns = {'cluster'} # rename column; each row is a spike, with its column denoting its cluster 
vis_cx_spikes_id = spike_clusters[spike_clusters.isin(vis_cx_clusters_id)].dropna().index

# counts the number of spikes per cluster
cluster_spike_counts = vis_cx_spikes.groupby('cluster').cluster.count()



# to get spike times of spikes from vis cx:
vis_cx_spikes_times = pd.DataFrame(spikes.times[vis_cx_spikes_id].squeeze())
vis_cx_spikes_times.columns = {'spike time'}
vis_cx_spikes_times.index = vis_cx_spikes_id    # set index to match ID of spike
vis_cx_spikes_times = pd.concat([vis_cx_spikes_times, vis_cx_spikes], axis=1) # 

# get list of spike times grouped by clusters/neurons
vis_cx_spikes_times_clustered = vis_cx_spikes_times.groupby('cluster').apply(lambda x: x['spike time'].unique())

# raster plots of all neurons in region of interest
plt.figure()
plt.eventplot(vis_cx_spikes_times_clustered.values[0], color=".2")
plt.xlabel("Time (s)")
plt.ylabel("Neuron")
plt.yticks([])
plt.show()

