import numpy as np
import pyexlab as pylab
import torch
exp_path = r"C:\Users\bwilson\OneDrive - QuEra Computing\Documents\research_code\ISMetric\experiments\EX01_Dynamic_Metric\trials\edge_metric\experiment_2022_09_16_17_08_13"
exp_dict = pylab.fio.load_test_subject_dicts(exp_path)

model_save_path = r"C:\Users\bwilson\OneDrive - QuEra Computing\Documents\research_code\ISMetric\experiments\EX01_Dynamic_Metric\trials\edge_metric\experiment_2022_09_16_17_08_13\Edge Training Subject_0\model_dict"

model_dict = exp_dict['Edge Training Subject_0']['info']['trainer_info']['Trainers Info'][0]['Model State'][-1]
torch.save(model_dict, model_save_path)
pass