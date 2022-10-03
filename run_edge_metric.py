import pyexlab as pylab
import pyexml as pyml
import torch
import os
import copy
import numpy as np



def edge_metric_subject(domain_set_path, image_set_path, precompute):

    #Data sets
    domain_set = np.load(domain_set_path)
    image_set = np.load(image_set_path)

    num_nodes = len(domain_set)
    validation_ratio = 0.2
    validation_cutoff_idx = int(len(domain_set) * validation_ratio)

    node_subset = pyml.datasets.utils.split_indeces(num_nodes, validation_cutoff_idx, num_nodes)

    metric_dataset = pyml.datasets.MetricSpaceDataset(metric = None, space = domain_set, precompute = precompute, subspace=node_subset[1])

    num_edges = len(metric_dataset)
    train_test_ratio = 0.7
    subspace_ratio = 1.0

    total_datasize = int(num_edges * subspace_ratio)
    edge_cutoff_idx = int(total_datasize * train_test_ratio)

    #Split Indeces for datasets
    subset_idx = pyml.datasets.utils.split_indeces(num_edges, edge_cutoff_idx, total_datasize)

    metric_training_dataset = torch.utils.data.Subset(metric_dataset, subset_idx[0])
    metric_testing_dataset = torch.utils.data.Subset(metric_dataset, subset_idx[1])

    #Construct neural networks
    map_net = pyml.models.Simple_FFWDNet(domain_set.shape[1], image_set.shape[1])
    metric_net = pyml.models.MetricNet(map_net, pyml.geometry.metrics.torch_metrics.Euclidean())

    #Construct trainers
    metric_optimizer = torch.optim.SGD(metric_net.parameters(), lr=0.001)

    metric_trainer = pyml.trainers.Trainer(metric_net, metric_training_dataset, torch.nn.MSELoss(), metric_optimizer, 
                                            scheduler = torch.optim.lr_scheduler.ExponentialLR(metric_optimizer, gamma=0.9),
                                            alt_name= "Metric Trainer")
    metric_tester  = pyml.trainers.Tester(metric_net, metric_testing_dataset, torch.nn.MSELoss(), alt_name= "Metric Tester")

    #Construct the coach that invokes the trainers at certain epochs
    trainer_schedule = pyml.trainers.Modulus_Schedule([-1, -1])
    trainers = [metric_trainer, metric_tester]

    metric_coach = pyml.trainers.Coach(trainers=trainers, trainer_schedule=trainer_schedule)
    metric_subject = pyml.mlsubjects.NeuralNetSubject(metric_coach, alt_name="Edge Training Subject")

    metric_subject.test_dict['Info']['Validation Nodes'] = node_subset[0]
    metric_subject.test_dict['Info']['Training Nodes'] = node_subset[1]
    metric_subject.test_dict['Info']['Training Edges'] = subset_idx[0]
    metric_subject.test_dict['Info']['Testing Edges'] = subset_idx[1]


    return metric_subject

if __name__ == "__main__":

    domain_path = r"C:\Users\bwilson\OneDrive - QuEra Computing\Documents\research_code\ISMetric\experiments\EX01_Dynamic_Metric\datasets\Lattice 4x4 Independent Set\IS_n314_m16.npy"
    image_path = r"C:\Users\bwilson\OneDrive - QuEra Computing\Documents\research_code\ISMetric\experiments\EX01_Dynamic_Metric\datasets\Lattice 4x4 Independent Set\Lattice_n314_m16.npy"
    precompute = np.load(r"C:\Users\bwilson\OneDrive - QuEra Computing\Documents\research_code\ISMetric\experiments\EX01_Dynamic_Metric\datasets\Lattice 4x4 Independent Set\IS_PairwiseMetric_HammingLattice_n314_m16.npy")

    metric_subject = edge_metric_subject(domain_path, image_path, precompute)

    #Build experiment
    folder_save_path = r"C:\Users\bwilson\OneDrive - QuEra Computing\Documents\research_code\ISMetric\experiments\EX01_Dynamic_Metric\trials\edge_metric\4x4"
    experiment_name = "validation_nodes_experiment_" + pylab.fio.datetime_now_str()
    experiment_save_path = pylab.fio.makeDirectory(folder_save_path, experiment_name)
    experiment = pylab.Experiment(experiment_save_path, [metric_subject])

    #Run experiment
    experiment.run(epochs=100) #Check formatting of output of model to be batch conscious and type appropriate