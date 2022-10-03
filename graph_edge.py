import pyexml as pyml
import pyexlab as pylab
import os 
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import math
import sklearn


#Pulls in all losses from coach
def get_coach_loss(coach_dict):

    loss_dict = {}
    coach_trainers = coach_dict['Measurements']['Loss']

    for trainer_id in coach_trainers[0]:
        loss_dict[trainer_id] = []

    for loss_frame in coach_trainers:
        for trainer_id in loss_frame:
            loss_dict[trainer_id].append(loss_frame[trainer_id])

    return loss_dict

def swap_dict_keys(d1, d2):
    #List of dictionaries with
    inv_dict = {}
    for k1 in d1:
        for k2 in d2[k1]:
            inv_dict[k2] = {k1 : d2[k1][k2]}
    
    return inv_dict

def extract_model_states(subject, model_idx):
    
    model_dict = {}
    
    for trainer_id in subject['Info']['Trainer']['Trainers Info']:
        name = subject['Info']['Trainer']['Trainers Info'][trainer_id]['Name']
        if 'Model State' in subject['Info']['Trainer']['Trainers Info'][trainer_id] and 'Trainer' in name:
            model_dict[name] = subject['Info']['Trainer']['Trainers Info'][trainer_id]['Model State'][model_idx]
    
    return model_dict

def get_models(exp_dict, model_class):
    #Get models
    model_dict = {}
    for subj_id in exp_dict:
        model_dict[subj_id] = get_models_subject(exp_dict[subj_id], -1, model_class)

    return model_dict

def get_models_subject(subject, model_idx, model_class):

    subject_models = extract_model_states(subject, model_idx)
    model_list = []
    for model_id in subject_models:
        model = model_class
        model.load_state_dict(subject_models[model_id])
        subject_models[model_id] = model
        subject_models[model_id].eval()

    return subject_models

def retest_model(model, dataset, vectorized = False, alt_name = "Tester", loss_fn = torch.nn.MSELoss()):

    tester = pyml.trainers.Tester(model, dataset, loss_fn, alt_name=alt_name)

    if vectorized:
        return tester(epoch = 0, vectorized=True)
    else:
        return tester(epoch = 0)

def retest_subject(subject, dataset, vectorized = False):

    subject_models = get_models_subject(subject, -1, model_copy)

    loss = {}

    for trainer_id in subject:
        if "Tester" in trainer_id:
            loss[trainer_id] = retest_model(subject_models[trainer_id], dataset, vectorized, alt_name = "Metric Tester", loss_fn = torch.nn.MSELoss())

    return loss

def find_largest_k_component(adj_matrix):
    #for i in range()
    pass

def plot_edge_loss(exp_dict):

    loss_dict = {}

    for subject_id in exp_dict:
        loss_dict[subject_id] = get_coach_loss(exp_dict[subject_id])

    inv_dict = swap_dict_keys(exp_dict, { k1 : loss_dict[k1] for k1 in exp_dict })

    if len(inv_dict) == 1:
        loss_fig, loss_ax = plt.subplots(len(inv_dict), 1)
        loss_fig = [loss_fig]
        loss_ax = [loss_ax]
    else:
        loss_fig, loss_ax = plt.subplots(len(inv_dict), 1)

    for s_i, subject_id in enumerate(inv_dict):
        ax = loss_ax[s_i]
        for trainer_id in inv_dict[subject_id]:
            pyml.plot.plot_loss(inv_dict[subject_id][trainer_id], ax, title=subject_id, label=trainer_id, xlabel="Epoch", ylabel="Loss")

def MST_edge_graph(exp_dict, d_path, precompute):
    
    metric_dataset = pyml.datasets.MetricSpaceDataset(metric = None, space = np.load(d_path), precompute=precompute)

    loss_dict = {}

    #Get all model edges 
    for subject_id in exp_dict:
        
        loss_vector = retest_model(exp_dict[subject_id], metric_dataset, vectorized = True)

        #Get subspaces
        subspaces = {}

        for trainer_id in exp_dict[subject_id]:

            subspaces[trainer_id] = exp_dict[subject_id][trainer_id]['Dataset Info']['subspace']

            n = int(math.sqrt(len(loss_vector)))
            adj_matrix = np.zeros([n, n])

            adj_matrix[subspaces[trainer_id]] = loss_vector[subspaces[trainer_id]]
            
            MST_adj_matrix = csr_matrix(adj_matrix.toarray().astype(float))
            
def retest_validation(subj_dict, d_path, precompute, model_copy):

    domain_set = np.load(d_path)

    subject_models = get_models_subject(subj_dict, -1, model_copy)
    loss_vector = {}

    for trainer_id in subject_models:

        if 'Trainer' in trainer_id:
            node_validation_set = subj_dict['Info']['Validation Nodes']
            val_metric_dataset = pyml.datasets.MetricSpaceDataset(metric = None, space = domain_set, precompute = precompute, subspace=node_validation_set)
            loss_vector[trainer_id] = retest_model(subject_models[trainer_id], val_metric_dataset, vectorized = False)
    
    return loss_vector

def plot_validation_loss(exp_dict, d_path, precompute, model_copy):

    validation_loss = {}

    for subj_id in exp_dict:
        validation_loss[subj_id] = retest_validation(exp_dict[subj_id], d_path, precompute, model_copy)

    inv_dict = swap_dict_keys(exp_dict, { k1 : validation_loss[k1] for k1 in exp_dict })

    if len(inv_dict) == 1:
        loss_fig, loss_ax = plt.subplots(len(inv_dict), 1)
        loss_fig = [loss_fig]
        loss_ax = [loss_ax]
    else:
        loss_fig, loss_ax = plt.subplots(len(inv_dict), 1)

    for s_i, subject_id in enumerate(inv_dict):
        ax = loss_ax[s_i]
        for trainer_id in inv_dict[subject_id]:
            pyml.plot.plot_loss(inv_dict[subject_id][trainer_id], ax, title=subject_id, label=trainer_id, xlabel="Epoch", ylabel="Loss")

def get_vectorized_loss(exp_dict, model_class, H):

    model_dict = get_models(exp_dict, model_class)
    loss_dict = {}

    for s_i, subject_id in enumerate(exp_dict):

        vectorized_loss_name = "tensor_total_loss_metric_%s.npy" %(subject_id)
        loss_path = os.path.join(exp_dict[subject_id]['Path'], vectorized_loss_name)

        if os.path.exists(loss_path):
            loss = np.load(loss_path)
        else:
            metric_dataset = pyml.datasets.MetricSpaceDataset(metric = None, space = domain_set, precompute=H)
            tester = pyml.trainers.Tester(model_dict[subject_id]['Metric Trainer'], metric_dataset, torch.nn.MSELoss(), alt_name="Metric Tester")

            vectorized_test = tester(vectorized = True)
            
            n = len(domain_set)
            loss = np.zeros([n,n])
            loss = np.reshape(vectorized_test, [n,n])

            #for idx in range(n**2):
            #    i = int(idx / n)
            #    j = int(idx % n)
            #    loss[i,j] = vectorized_test[idx]
        
            np.save(loss_path, loss)

        loss_dict[subject_id] = loss

    return loss_dict

def get_D_vectorized(exp_dict, model_class, H):

    model_dict = get_models(exp_dict, model_class)
    loss_dict = {}

    for s_i, subject_id in enumerate(exp_dict):

        vectorized_loss_name = "tensor_total_D_metric_%s.npy" %(subject_id)
        loss_path = os.path.join(exp_dict[subject_id]['Path'], vectorized_loss_name)

        if os.path.exists(loss_path):
            loss = np.load(loss_path)
        else:
            metric_dataset = pyml.datasets.MetricSpaceDataset(metric = None, space = domain_set, precompute=H)
            loss = compute_D_vector(model_dict[subject_id]['Metric Trainer'], metric_dataset)
        
            np.save(loss_path, loss)

        loss_dict[subject_id] = loss

    return loss_dict

def plot_heat_map(exp_dict, model_class, H):

    #Get models
    loss_map = get_vectorized_loss(exp_dict, model_class, H)

    #Make figures
    heat_fig, heat_ax = plt.subplots(len(exp_dict), 4)
    if len(exp_dict) == 1:
        heat_ax = [heat_ax]
    

    for s_i, subj_id in enumerate(exp_dict):

        #exp_dict[subj_id]['Info']['Validation Nodes']
        #exp_dict[subj_id]['Info']['Info']['Training Nodes'] 
        #exp_dict[subj_id]['Info']['Info']['Training Edges'] 
        #exp_dict[subj_id]['Info']['Info']['Testing Edges'] 

        loss = np.array(loss_map[subj_id])
        
        
        #Total Heat Map
        pyml.plot.seismic_heat_map(loss, heat_fig, heat_ax[s_i][0], title="Heat Map Total Loss", xlabel="Dataset idx", ylabel="Dataset idx")

        #Training Heat Map
        train_heat = np.zeros(H.shape)
        train_edges_idx = np.unravel_index(exp_dict[subj_id]['Info']['Training Edges'], H.shape, order='C')
#        train_edges_idx[0] = np.array(exp_dict[subj_id]['Info']['Training Nodes'])[train_edges_idx[0]]
#        train_edges_idx[1] = np.array(exp_dict[subj_id]['Info']['Training Nodes'])[train_edges_idx[1]]
        t_edge = []
        t_edge.append(np.concatenate((np.array(train_edges_idx[0]), np.array(train_edges_idx[1]))))
        t_edge.append(np.concatenate((np.array(train_edges_idx[1]), np.array(train_edges_idx[0]))))
        train_heat[t_edge] = loss[t_edge]

        pyml.plot.seismic_heat_map(train_heat, heat_fig, heat_ax[s_i][1], title="Heat Map Training Set Loss ", xlabel="Dataset idx", ylabel="Dataset idx")

        #Testing Heat Map
        test_heat = np.zeros(H.shape)
        test_edges_idx = np.unravel_index(exp_dict[subj_id]['Info']['Testing Edges'], H.shape, order='C')
        t_edge = []
        t_edge.append(np.concatenate((np.array(test_edges_idx[0]), np.array(test_edges_idx[1]))))
        t_edge.append(np.concatenate((np.array(test_edges_idx[1]), np.array(test_edges_idx[0]))))
        
        #test_edges_idx[0] = np.concatenate((t_edge[0], test_edges_idx[1]))
        #test_edges_idx[1] = np.concatenate((test_edges_idx[1], test_edges_idx[0]))
        test_heat[t_edge] = loss[t_edge]

        pyml.plot.seismic_heat_map(test_heat, heat_fig, heat_ax[s_i][2], title="Heat Map Testing Set Loss" , xlabel="Dataset idx", ylabel="Dataset idx")

        #Validation Heat Map
        valid_heat = np.zeros(H.shape)
        valid_edges_idx = np.meshgrid(exp_dict[subj_id]['Info']['Validation Nodes'])
        valid_heat[valid_edges_idx] = np.squeeze(loss[valid_edges_idx])

        pyml.plot.seismic_heat_map(valid_heat, heat_fig, heat_ax[s_i][3], title="Heat Map Validation Set Loss", xlabel="Dataset idx", ylabel="Dataset idx")

def plot_distance_comparison():

    H_fig, H_ax = plt.subplots(len(exp_dict), 2)

    for s_i, subject_id in enumerate(exp_dict):

        vectorized_loss_name = "tensor_loss_metric_%s.npy" %(subject_id)
        loss_path = os.path.join(exp_dict['Path'], vectorized_loss_name)
        loss = np.load(loss_path)

        for sub_i, sub_type in enumerate(subspace):
            sub_mesh = np.meshgrid(subspace[sub_type][subject_id])
            D_map = np.sqrt(loss[sub_mesh]) + H[sub_mesh]
            Y = D_map.flatten()
            X = H[sub_mesh].flatten()
            pyml.plot.scatter_XY(X, Y, H_ax[s_i, sub_i], title=subject_id + sub_type, label=sub_type, xlabel="H", ylabel="D")

def plot_density_plot(exp_dict, model_class, H):

    #Get models
    loss_map = get_D_vectorized(exp_dict, model_class, H)

    heat_fig, heat_ax = plt.subplots(len(exp_dict), 4)

    if len(exp_dict) == 1:
        heat_ax = [heat_ax]
    

    for s_i, subj_id in enumerate(exp_dict):

        #exp_dict[subj_id]['Info']['Validation Nodes']
        #exp_dict[subj_id]['Info']['Info']['Training Nodes'] 
        #exp_dict[subj_id]['Info']['Info']['Training Edges'] 
        #exp_dict[subj_id]['Info']['Info']['Testing Edges'] 

        loss = np.array(loss_map[subj_id])
        
        #Total Heat Map
        Y = loss.flatten()
        X = H.flatten()
        pyml.plot.scatter_XY(X, Y, heat_ax[s_i][0], title="H vs. D Total", xlabel="H", ylabel="D")

        #Training Heat Map

        train_e = exp_dict[subj_id]['Info']['Training Edges']
        pyml.plot.scatter_XY(X[train_e], Y[train_e], heat_ax[s_i][1], title="H vs. D Train" , xlabel="H", ylabel="D")

        #Testing Heat Map
        test_e = exp_dict[subj_id]['Info']['Testing Edges']
        pyml.plot.scatter_XY(X[test_e], Y[test_e], heat_ax[s_i][2], title="H vs. D Test" , xlabel="H", ylabel="D")


        #Validation Heat Map
        valid_edges_idx_2d = np.array(np.meshgrid(exp_dict[subj_id]['Info']['Validation Nodes']))
        valid_e = np.ravel_multi_index(valid_edges_idx_2d, (H.shape[0]**2))
        pyml.plot.scatter_XY(X[valid_e], Y[valid_e], heat_ax[s_i][3], title="H vs. D Validation", xlabel="H", ylabel="D")


def compute_D_vector(model, dataset):

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    vec = np.zeros(len(dataloader))

    with torch.no_grad():
        for batch_idx, samples in enumerate(dataloader):
            vec[batch_idx] = model(samples[0])

    return vec

def plot_embedding_graphs(exp_dict):
    pass

def plot_validation_neighborhood(exp_dict):

    eigen_embedding = sklearn.manifold.SpectralEmbedding(n_components=3)


    pass
            
exp_path = r"C:\Users\bwilson\OneDrive - QuEra Computing\Documents\research_code\ISMetric\experiments\EX01_Dynamic_Metric\trials\edge_metric\4x4\validation_nodes_experiment_2022_09_28_07_14_58"
exp_dict = pylab.fio.load_test_subject_dicts(exp_path)


for subj_id in exp_dict:
    exp_dict[subj_id]['Path'] = os.path.join(exp_path , exp_dict[subj_id]['Info']['Test ID'])

plot_edge_loss(exp_dict)

domain_path = r"C:\Users\bwilson\OneDrive - QuEra Computing\Documents\research_code\ISMetric\experiments\EX01_Dynamic_Metric\datasets\Lattice 4x4 Independent Set\IS_n314_m16.npy"
image_path = r"C:\Users\bwilson\OneDrive - QuEra Computing\Documents\research_code\ISMetric\experiments\EX01_Dynamic_Metric\datasets\Lattice 4x4 Independent Set\Lattice_n314_m16.npy"
precompute = np.load(r"C:\Users\bwilson\OneDrive - QuEra Computing\Documents\research_code\ISMetric\experiments\EX01_Dynamic_Metric\datasets\Lattice 4x4 Independent Set\IS_PairwiseMetric_HammingLattice_n314_m16.npy")

domain_set = np.load(domain_path)
image_set = np.load(image_path)

map_net = pyml.models.Simple_FFWDNet(domain_set.shape[1], image_set.shape[1])
metric_net = pyml.models.MetricNet(map_net, pyml.geometry.metrics.torch_metrics.Euclidean())

#plot_validation_loss(exp_dict, domain_path, precompute, metric_net)

plot_heat_map(exp_dict, metric_net, precompute)

plot_density_plot(exp_dict, metric_net, precompute)

#plot_embedding_graphs(exp_dict)

plt.show()