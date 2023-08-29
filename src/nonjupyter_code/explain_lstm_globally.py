import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from captum.attr import IntegratedGradients


# Parameters:
NUMBER_OF_HIDDEN_UNITS = 64
LEARNING_RATE = 0.9
batch_size = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("available device: ", device)


# Classes:
class DKRZSeismologyLSTM(nn.Module):
    def __init__(self, num_hidden=NUMBER_OF_HIDDEN_UNITS):
        super(DKRZSeismologyLSTM, self).__init__()
        self.num_hidden = num_hidden
        self.layer1 = nn.LSTMCell(49, self.num_hidden, dtype=torch.float64).to(device) # we will feed in one sample at a time, and then zero out the h and c variables after all the train data points were seen.
        self.layer2 = nn.LSTMCell(self.num_hidden, self.num_hidden, dtype=torch.float64).to(device)
        self.out = nn.Linear(self.num_hidden, 1, dtype=torch.float64).to(device)

    def forward(self, x):
        #print("input x shape: ", x.size())
        num_batches = x.size(0)
        #print("num_batches", num_batches)
        output_list = []
        
        h_t1 = torch.zeros(num_batches, self.num_hidden, dtype=torch.float64).to(device)
        c_t1 = torch.zeros(num_batches, self.num_hidden, dtype=torch.float64).to(device)
        h_t2 = torch.zeros(num_batches, self.num_hidden, dtype=torch.float64).to(device)
        c_t2 = torch.zeros(num_batches, self.num_hidden, dtype=torch.float64).to(device)

        for input_split in x.split(1, dim=1):
            #print("input_split before:", input_split.size())
            input_split_reshaped = input_split.reshape(num_batches, 49)
            #print("input_split after:", input_split_reshaped.size())
            h_t1, c_t1 = self.layer1(input_split_reshaped, (h_t1, c_t1))
            h_t2, c_t2 = self.layer2(h_t1, (h_t2, c_t2))
            output = self.out(h_t2)
            output_list.append(output)

        outputs = torch.cat(output_list, dim=1)
        return outputs


# Dataset: 
df_train_minmax = pd.HDFStore("../../cloudcontainer/concatenated_experiments.hdf5", mode='r').select("alldata_mean_instead_of_NaN/train/minmax")
df_pressure_train_minmax = df_train_minmax["pressure"]
df_ae_train_minmax = df_train_minmax["ae"]

df_val_minmax = pd.HDFStore("../../cloudcontainer/concatenated_experiments.hdf5", mode='r').select("alldata_mean_instead_of_NaN/val/minmax")
df_pressure_val_minmax = df_val_minmax["pressure"]
df_ae_val_minmax = df_val_minmax["ae"]

df_test_minmax = pd.HDFStore("../../cloudcontainer/concatenated_experiments.hdf5", mode='r').select("alldata_mean_instead_of_NaN/test/minmax")
df_pressure_test_minmax = df_test_minmax["pressure"]
df_ae_test_minmax = df_test_minmax["ae"]

df_train_minmax = df_train_minmax.drop(["pressure"], axis=1)
df_val_minmax = df_val_minmax.drop(["pressure"], axis=1)
df_test_minmax = df_test_minmax.drop(["pressure"], axis=1)
#df_train_minmax = df_train_minmax.drop(["ae"], axis=1)
#df_val_minmax = df_val_minmax.drop(["ae"], axis=1)
#df_test_minmax = df_test_minmax.drop(["ae"], axis=1)
feature_names = df_train_minmax.columns.to_list()
print(f"Feature names: {feature_names}")

np_train_minmax = df_train_minmax.to_numpy()
np_pressure_train_minmax = df_pressure_train_minmax.to_numpy()
np_ae_train_minmax = df_ae_train_minmax.to_numpy()

np_val_minmax = df_val_minmax.to_numpy()
np_pressure_val_minmax = df_pressure_val_minmax.to_numpy()
np_ae_val_minmax = df_ae_val_minmax.to_numpy()

np_test_minmax = df_test_minmax.to_numpy()
np_pressure_test_minmax = df_pressure_test_minmax.to_numpy()
np_ae_test_minmax = df_ae_test_minmax.to_numpy()

train_series_len = len(np_pressure_train_minmax)
val_series_len = len(np_pressure_val_minmax)
test_series_len = len(np_pressure_test_minmax)

n_features = np_train_minmax.shape[1]
print("n_features=", n_features)

train_shift_step = int(train_series_len / batch_size)
val_shift_step = int(val_series_len / batch_size)
test_shift_step = int(test_series_len / batch_size)

shift_step = int(test_series_len / batch_size)
print("shift_step=", shift_step)

np_train_minmax_shifted = np.empty((batch_size, train_series_len, n_features), np.float64)
np_pressure_train_minmax_shifted = np.empty((batch_size, train_series_len), np.float64)
np_ae_train_minmax_shifted = np.empty((batch_size, train_series_len), np.float64)

np_val_minmax_shifted = np.empty((batch_size, val_series_len, n_features), np.float64)
np_pressure_val_minmax_shifted = np.empty((batch_size, val_series_len), np.float64)
np_ae_val_minmax_shifted = np.empty((batch_size, val_series_len), np.float64)

np_test_minmax_shifted = np.empty((batch_size, test_series_len, n_features), np.float64)
np_pressure_test_minmax_shifted = np.empty((batch_size, test_series_len), np.float64)
np_ae_test_minmax_shifted = np.empty((batch_size, test_series_len), np.float64)


for i in range(batch_size):
    rolling_amount = shift_step*i
    np_train_minmax_shifted[i] = np.roll(np_train_minmax, rolling_amount)
    np_pressure_train_minmax_shifted[i] = np.roll(np_pressure_train_minmax, rolling_amount)
    np_ae_train_minmax_shifted[i] = np.roll(np_ae_train_minmax, rolling_amount)

for i in range(batch_size):
    rolling_amount = shift_step*i
    np_val_minmax_shifted[i] = np.roll(np_val_minmax, rolling_amount)
    np_pressure_val_minmax_shifted[i] = np.roll(np_pressure_val_minmax, rolling_amount)
    np_ae_val_minmax_shifted[i] = np.roll(np_ae_val_minmax, rolling_amount)
    
for i in range(batch_size):
    rolling_amount = shift_step*i
    np_test_minmax_shifted[i] = np.roll(np_test_minmax, rolling_amount)
    np_pressure_test_minmax_shifted[i] = np.roll(np_pressure_test_minmax, rolling_amount)
    np_ae_test_minmax_shifted[i] = np.roll(np_ae_test_minmax, rolling_amount)


train_input_torch = torch.from_numpy(np_train_minmax_shifted[:, :-1,:]).to(device)
train_target_torch = torch.from_numpy(np_pressure_train_minmax_shifted[:, 1:]).to(device)

val_input_torch = torch.from_numpy(np_val_minmax_shifted[:, :-1,:]).to(device)
val_target_torch = torch.from_numpy(np_pressure_val_minmax_shifted[:, 1:]).to(device)

test_input_torch = torch.from_numpy(np_test_minmax_shifted[:, :-1,:]).to(device)
test_target_torch = torch.from_numpy(np_pressure_test_minmax_shifted[:, 1:]).to(device)

print(np_train_minmax_shifted.shape)
print(train_input_torch.shape)
print(train_target_torch.shape)


# Test model:
seismology_model = DKRZSeismologyLSTM().to(device)
criterion = nn.MSELoss()
seismology_model = torch.load("../../cloudcontainer/experiments_related/best_seismology_model.pth", map_location=torch.device(device))
with torch.no_grad():
    interemdiary_train_pred = seismology_model(train_input_torch)
    intermediary_train_loss = criterion(interemdiary_train_pred, train_target_torch)
    print("intermediary_train_loss: ", intermediary_train_loss.item())
    np_pred_cpu_train = interemdiary_train_pred.detach().cpu().numpy()[0] # index 0 here means basically the original unshifted dataset
    target_cpu_train = train_target_torch.detach().cpu().numpy()[0] # index 0 here means basically the original unshifted dataset

    intermediary_val_pred = seismology_model(val_input_torch)
    intermediary_val_loss = criterion(intermediary_val_pred, val_target_torch)
    print("intermediary_val_loss: ", intermediary_val_loss.item())
    np_pred_cpu_val = intermediary_val_pred.detach().cpu().numpy()[0] # index 0 here means basically the original unshifted dataset
    target_cpu_val = val_target_torch.detach().cpu().numpy()[0] # index 0 here means basically the original unshifted dataset

    intermediary_test_pred = seismology_model(test_input_torch)
    intermediary_test_loss = criterion(intermediary_test_pred, test_target_torch)
    print("intermediary_test_loss: ", intermediary_test_loss.item())
    np_pred_cpu_test = intermediary_test_pred.detach().cpu().numpy()[0] # index 0 here means basically the original unshifted dataset
    target_cpu_test = test_target_torch.detach().cpu().numpy()[0] # index 0 here means basically the original unshifted dataset

# use sklearn MSE metrics to compare with the baseline using the same exact implementation of the metric:
mse_train = metrics.mean_squared_error(target_cpu_train, np_pred_cpu_train)
mse_val = metrics.mean_squared_error(target_cpu_val, np_pred_cpu_val)
mse_test = metrics.mean_squared_error(target_cpu_test, np_pred_cpu_test)
print("train mse: ", mse_train)
print("val mse: ", mse_val)
print("test mse: ", mse_test)


# Plot test related:
fig, ax = plt.subplots(figsize=(18, 12))
ax.plot(np_pred_cpu_test, color="blue", label="Test prediction")
ax.plot(target_cpu_test, color="orange", label="Test ground truth")

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
text_box_str = f"test error={intermediary_test_loss.item()}" + f"\nval error={intermediary_val_loss.item()}" + f"\ntrain error={intermediary_train_loss.item()}"
ax.text(0.15, 0.98, text_box_str, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)

ax.set_title(f"Test results", fontsize=20)
ax.set_xlabel("Time Index", fontsize=16)
ax.set_ylabel("Pressure", fontsize=16)
plt.legend()
plt.savefig(f"../../cloudcontainer/experiments_related/LSTM_global_explanation_saved_material/test_fig.png")


# Plot train|val|test related:
fig, ax = plt.subplots(figsize=(18, 12))
x_data = range(len(np_pred_cpu_train)+len(np_pred_cpu_val)+len(np_pred_cpu_test))

ax.plot(x_data[:len(np_pred_cpu_train)], np_pred_cpu_train, color="blue", label="Train prediction")
ax.plot(x_data[:len(np_pred_cpu_train)], target_cpu_train, color="orange", label="Train ground truth")

ax.plot(x_data[len(np_pred_cpu_train):len(np_pred_cpu_train)+len(np_pred_cpu_val)], np_pred_cpu_val, color="black", label="Validation prediction")
ax.plot(x_data[len(np_pred_cpu_train):len(np_pred_cpu_train)+len(np_pred_cpu_val)], target_cpu_val, color="yellow", label="Validation ground truth")

ax.plot(x_data[len(np_pred_cpu_train)+len(np_pred_cpu_val):], np_pred_cpu_test, color="red", label="Test prediction")
ax.plot(x_data[len(np_pred_cpu_train)+len(np_pred_cpu_val):], target_cpu_test, color="gray", label="Test ground truth")

ax.axvline(x=train_series_len, color='g', lw=4)
ax.axvline(x=train_series_len+val_series_len, color='black', lw=4)

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
text_box_str = f"test error={mse_test}" + f"\nval error={mse_val}" + f"\ntrain error={mse_train}"
ax.text(0.35, 0.98, text_box_str, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)

ax.set_title(f"Train|Val|Test Results for LSTM", fontsize=20)
ax.set_xlabel("Time Index", fontsize=16)
ax.set_ylabel("Pressure", fontsize=16)
plt.legend()
plt.savefig(f"../../cloudcontainer/experiments_related/LSTM_global_explanation_saved_material/trainvaltest_fig.png")

plt.close('all')

#################################################################################################################################################

# Explainable AI section:

ig = IntegratedGradients(seismology_model)
baseline = torch.zeros_like(test_input_torch)
print(baseline.shape)


'''
# Graphical illustrations:
offset=16
neighborhood = 15
assert offset > neighborhood, f"Please make the offset larger than {neighborhood}"
num_examples_to_show = 2
for target_index in range(offset, test_series_len-offset, (test_series_len-2*offset) // num_examples_to_show):
    attributions, delta = ig.attribute(test_input_torch, target=target_index, return_convergence_delta=True)
    print("attributions shape: ", attributions.shape)
    attr_to_plot = attributions.detach().cpu().numpy()[0].T
    print(attr_to_plot.shape)
    
    # Bigger picture plot:
    for f in range(49):
        plt.plot(attr_to_plot[f])
    
    plt.title(f"Zoomed-out feature attributions for target time={target_index}")
    plt.show()

    # Zoomed-in plot:
    for f in range(49):
        plt.plot(range(target_index-neighborhood, target_index+neighborhood), attr_to_plot[f, target_index-neighborhood:target_index+neighborhood], label=feature_names[f])
    plt.title(f"Zoomed-in feature attributions for target time={target_index}")
    #plt.legend()
    plt.show()
    

# End of graphical illustrations
'''

# Graphical illustrations:
neighborhood=15
for target_index in range(0, test_series_len-1):
    attributions, delta = ig.attribute(test_input_torch, target=target_index, return_convergence_delta=True)
    print(f"Plots for target {target_index}: ", attributions.shape)
    attr_to_plot = attributions.detach().cpu().numpy()[0].T
    print(attr_to_plot.shape)
    
    # Bigger picture plot:
    fig = plt.figure(figsize=(16, 12))
    for f in range(49):
        plt.plot(attr_to_plot[f])
    
    plt.title(f"Zoomed-out feature attributions for target time={target_index}")
    plt.savefig(f"../../cloudcontainer/experiments_related/LSTM_global_explanation_saved_material/figures_for_individual_targets/{target_index}_zoomedout.png")
    plt.close() # avoids consuming to much RAM and the risk for the process to be killed by the OS

    # Zoomed-in plot:
    fig = plt.figure(figsize=(16, 12))
    for f in range(49):
        if target_index < neighborhood:
            plt.plot(range(0, target_index+neighborhood), attr_to_plot[f, 0:target_index+neighborhood], label=feature_names[f])
        elif neighborhood < target_index and target_index < test_series_len - neighborhood:    
            plt.plot(range(target_index-neighborhood, target_index+neighborhood), attr_to_plot[f, target_index-neighborhood:target_index+neighborhood], label=feature_names[f])
        elif target_index > test_series_len - neighborhood:
            plt.plot(range(test_series_len-neighborhood, target_index), attr_to_plot[f, test_series_len-neighborhood:target_index], label=feature_names[f])
        else:
            pass
    plt.title(f"Zoomed-in feature attributions for target time={target_index}")
    plt.savefig(f"../../cloudcontainer/experiments_related/LSTM_global_explanation_saved_material/figures_for_individual_targets/{target_index}_zoomedin.png")
    plt.close()

# End of graphical illustrations


cumulated_feature_attributions = torch.zeros((49,)).to(device)
for target_index in range(test_series_len-1):
    print("Test target index: ", target_index)
    attributions, delta = ig.attribute(test_input_torch, target=target_index, return_convergence_delta=True)

    target_sample = attributions[0, target_index, :]
    target_sample_to_plot = target_sample.detach().cpu().numpy()

    # Save a bar-plot for each individual target index:
    fig = plt.figure(figsize=(16, 12))
    plt.barh(feature_names, target_sample_to_plot)
    plt.xticks(rotation="vertical")
    plt.title(f"Feature attributions for target index {target_index}")
    plt.savefig(f"../../cloudcontainer/experiments_related/LSTM_global_explanation_saved_material/figures_for_individual_targets/{target_index}_bar.png")
    plt.close()

    # Sum up the feature attribution values for all targets:
    cumulated_feature_attributions = torch.add(cumulated_feature_attributions, target_sample)

print("cumulated attributions: ", cumulated_feature_attributions)

cumulated_feature_attributions_np = cumulated_feature_attributions.detach().cpu().numpy()
print(cumulated_feature_attributions_np.shape)

fig = plt.figure(figsize=(16, 12))
plt.barh(feature_names, cumulated_feature_attributions_np)
plt.xticks(rotation="vertical")
plt.title("Cumulated feature attributions over the entire test dataset")
plt.savefig(f"../../cloudcontainer/experiments_related/LSTM_global_explanation_saved_material/cumulated_feature_attributions.png")