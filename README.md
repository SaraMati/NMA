# NMA


Project members: 
Aishling, Asim, Jane, Kevin, Sara

Dataset description: 

10 mice
39 sessions

Session structure: 'nicklab/Subjects/Cori/2016-12-14/001'
clusters. = collection of spikes arising from a single neuron (via spike sorting)

Anatomical info
depths.npy - um, position of the center of cluster, relative to the probe (deepest = 0, superficial = 3820)

Waveform analysis
to get complete waveforms across all channels in a probe, use both templateWaveformChans.npy and templateWaveforms.npy

Spikes analysis for FR
spikes.amps.npy - [μV] (nSpikes) The peak-to-trough amplitude, obtained from the template and template-scaling amplitude returned by Kilosort (not from the raw data)
spikes.times.npy
