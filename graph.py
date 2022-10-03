import pyexml as pyml
import pyexlab as pylab
import os 
import matplotlib.pyplot as plt

trial_path = r"C:\Users\bwilson\OneDrive - QuEra Computing\Documents\research_code\ISMetric\experiments\EX01_Dynamic_Metric\trials\dynamic_static_compare\4x4\experiment_2022_09_14_05_13_38"
trial_dict = pylab.fio.load_test_subject_dicts(trial_path)

loss_fig, loss_ax = plt.subplots(1, 1)

for subject_id in trial_dict.keys():
    data = trial_dict[subject_id]['time']['loss']
    print(data)
    #pyml.plot.plot_loss(data, loss_ax, title="Function Training Loss", label=subject_id, xlabel="Epoch", ylabel="Loss")

plt.show()



