import pyexml as pyml
import pyexlab as pylab
import os 
import matplotlib.pyplot as plt
import numpy as np
import torch

trial_path = r"C:\Users\bwilson\OneDrive - QuEra Computing\Documents\research_code\ISMetric\experiments\EX01_Dynamic_Metric\trials\edge_metric\experiment_2022_09_16_17_08_13"

trial_dict = pylab.fio.load_test_subject_dicts(trial_path)

loss_dict = {}
for subject_id in trial_dict:
    coach_loss_array = trial_dict[subject_id]['time']['loss']
    loss_dict[subject_id] = {}

    for trainer_id in coach_loss_array[0]:
        loss_dict[subject_id][trainer_id] = []

    for loss_val in coach_loss_array:
        for trainer_id in loss_val:
            loss_dict[subject_id][trainer_id].append(loss_val[trainer_id])

#Invert subject_id and trainer_id
inv_dict = {}
for subject_id in trial_dict:
    for trainer_id in loss_dict[subject_id]:
        inv_dict[trainer_id] = {}

for subject_id in trial_dict:
    for trainer_id in loss_dict[subject_id]:
        inv_dict[trainer_id][subject_id] = loss_dict[subject_id][trainer_id]
        
loss_fig, loss_ax = plt.subplots(len(inv_dict), 1)
for s_i, subject_id in enumerate(inv_dict):
    ax = loss_ax[s_i]
    for trainer_id in inv_dict[subject_id]:
        pyml.plot.plot_loss(inv_dict[subject_id][trainer_id], ax, title=subject_id, label=trainer_id, xlabel="Epoch", ylabel="Loss")



#Get nets
metric_nets = {}
map_nets = {}

for subject_id in trial_dict:
    map_net = pyml.models.Simple_FFWDNet(16, 12)
    metric_net = pyml.models.MetricNet(map_net, pyml.geometry.metrics.torch_metrics.Euclidean())
    for trainer_id in trial_dict[subject_id]['info']['trainer_info']['Trainers Info']:
        if 'Trainer' in trial_dict[subject_id]['info']['trainer_info']['Trainers Info'][trainer_id]['Name']:
            model_params = trial_dict[subject_id]['info']['trainer_info']['Trainers Info'][trainer_id]['Model State'][-1]
            if 'Metric' in trial_dict[subject_id]['info']['trainer_info']['Trainers Info'][trainer_id]['Name']:
                metric_net.load_state_dict(model_params)
            else:
                map_net.load_state_dict(model_params)
    
    metric_nets[subject_id] = metric_net
    map_nets[subject_id] = map_net

#Use nets to compute interesting graphs

#Compare H heatmap
H = np.load(r"C:\Users\bwilson\OneDrive - QuEra Computing\Documents\research_code\ISMetric\experiments\EX01_Dynamic_Metric\datasets\Lattice 4x4 Independent Set\IS_PairwiseMetric_HammingLattice_n314_m16.npy")

heat_fig, heat_ax = plt.subplots(len(trial_dict), 1)
domain_set = np.load(r"C:\Users\bwilson\OneDrive - QuEra Computing\Documents\research_code\ISMetric\experiments\EX01_Dynamic_Metric\datasets\Lattice 4x4 Independent Set\IS_n314_m16.npy")

for s_i, subject_id in enumerate(trial_dict):
    vectorized_loss_name = "tensor_loss_metric_%s.npy" %(subject_id)
    loss_path = os.path.join(trial_path, vectorized_loss_name)

    if os.path.exists(loss_path):
        loss = np.load(loss_path)
    else:
        metric_dataset = pyml.datasets.MetricSpaceDataset(metric = None, space = domain_set, precompute=H)
        tester = pyml.trainers.Tester(metric_nets[subject_id], metric_dataset, torch.nn.MSELoss(), alt_name="Metric Tester")

        vectorized_test = tester(vectorized = True)
        n = len(domain_set)
        loss = np.zeros([n,n])

        for idx in range(n**2):
            i = int(idx / n)
            j = int(idx % n)
            loss[i,j] = vectorized_test[idx]
    
        np.save(loss_path, loss)
    pyml.plot.seismic_heat_map(loss, heat_fig, heat_ax[s_i], title="Heat Map Loss %s" %(subject_id), xlabel="Dataset idx", ylabel="Dataset idx")

H_fig, H_ax = plt.subplots(len(trial_dict), 2)
subspace = {}
subspace['Training'] = {}
subspace['Testing'] = {}

for s_i, subject_id in enumerate(trial_dict):

    for trainer_id in trial_dict[subject_id]['info']['trainer_info']['Trainers Info']:
        if 'Trainer' in trial_dict[subject_id]['info']['trainer_info']['Trainers Info'][trainer_id]['Name']:
            subspace['Training'][subject_id] = trial_dict[subject_id]['info']['trainer_info']['Trainers Info'][trainer_id]['Dataset Info']['subspace']
        else:
            subspace['Testing'][subject_id] = trial_dict[subject_id]['info']['trainer_info']['Trainers Info'][trainer_id]['Dataset Info']['subspace']

for s_i, subject_id in enumerate(trial_dict):
    vectorized_loss_name = "tensor_loss_metric_%s.npy" %(subject_id)
    loss_path = os.path.join(trial_path, vectorized_loss_name)
    loss = np.load(loss_path)

    for sub_i, sub_type in enumerate(subspace):
        sub_mesh = np.meshgrid(subspace[sub_type][subject_id])
        D_map = np.sqrt(loss[sub_mesh]) + H[sub_mesh]
        Y = D_map.flatten()
        X = H[sub_mesh].flatten()
        pyml.plot.scatter_XY(X, Y, H_ax[s_i, sub_i], title=subject_id + sub_type, label=sub_type, xlabel="H", ylabel="D")


plt.show()



