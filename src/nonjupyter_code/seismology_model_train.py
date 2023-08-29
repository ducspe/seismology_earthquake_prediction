import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os


# Functions
def optimization_func():
    optimizer.zero_grad()
    model_output = seismology_model(train_input_torch) # notice that this allows us to add the entire data
    loss = criterion(model_output, train_target_torch)
    print("loss=", loss)
    loss.backward()
    return loss

# Classes
class DKRZSeismologyLSTM(nn.Module):
    def __init__(self, num_hidden):
        super(DKRZSeismologyLSTM, self).__init__()
        self.num_hidden = num_hidden
        self.layer1 = nn.LSTMCell(49, self.num_hidden, dtype=torch.float64).to(device) # we will feed in one sample at a time, and then zero out the h and c variables after all the train data points were seen.
        self.layer2 = nn.LSTMCell(self.num_hidden, self.num_hidden, dtype=torch.float64).to(device)
        self.out = nn.Linear(self.num_hidden, 1, dtype=torch.float64).to(device)

    def forward(self, x):
        print("input x shape: ", x.size())
        num_batches = x.size(0)
        print("num_batches", num_batches)
        output_list = []
        
        h_t1 = torch.zeros(num_batches, self.num_hidden, dtype=torch.float64).to(device)
        c_t1 = torch.zeros(num_batches, self.num_hidden, dtype=torch.float64).to(device)
        h_t2 = torch.zeros(num_batches, self.num_hidden, dtype=torch.float64).to(device)
        c_t2 = torch.zeros(num_batches, self.num_hidden, dtype=torch.float64).to(device)

        for input_split in x.split(1, dim=1):
            #print("input_split before:", input_split.size())
            input_split_reshaped = input_split.reshape(num_batches, 49)
            #print("input_split after:", input_split_reshaped.size())
            #print("input split type: ", input_split.type())
            h_t1, c_t1 = self.layer1(input_split_reshaped, (h_t1, c_t1))
            h_t2, c_t2 = self.layer2(h_t1, (h_t2, c_t2))
            output = self.out(h_t2)
            output_list.append(output)

        outputs = torch.cat(output_list, dim=1)
        #print("outputs: ", outputs)
        return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-units", required=True, type=int)
    parser.add_argument("--lr", required=True, type=float)
    parser.add_argument("--n-train-steps", required=True, type=int)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--lbfgs-iterations", required=True, type=int)

    args = parser.parse_args()

    # Hyperparameters
    NUMBER_OF_HIDDEN_UNITS = args.n_units # 64
    LEARNING_RATE = args.lr # 0.9
    num_train_steps = args.n_train_steps # 30
    batch_size = args.batch_size # 1
    LBFGS_MAX_ITERATIONS = args.lbfgs_iterations # 10

    results_location = f"../../cloudcontainer/experiments_related/nonjupyter_LSTM_train_saved_material/exp_units{NUMBER_OF_HIDDEN_UNITS}_lr{LEARNING_RATE}_lbfgs{LBFGS_MAX_ITERATIONS}"
    
    os.makedirs(results_location, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("available device: ", device)


    # Load the dataset that was created using the scripts from the "preprocessing" folder.
    df_train_minmax = pd.HDFStore("../../cloudcontainer/concatenated_experiments.hdf5", mode='r').select("alldata_mean_instead_of_NaN/train/minmax").astype('float64')
    df_pressure_train_minmax = df_train_minmax["pressure"]
    df_ae_train_minmax = df_train_minmax["ae"]

    df_val_minmax = pd.HDFStore("../../cloudcontainer/concatenated_experiments.hdf5", mode='r').select("alldata_mean_instead_of_NaN/val/minmax").astype('float64')
    df_pressure_val_minmax = df_val_minmax["pressure"]
    df_ae_val_minmax = df_val_minmax["ae"]

    df_test_minmax = pd.HDFStore("../../cloudcontainer/concatenated_experiments.hdf5", mode='r').select("alldata_mean_instead_of_NaN/test/minmax").astype('float64')
    df_pressure_test_minmax = df_test_minmax["pressure"]
    df_ae_test_minmax = df_test_minmax["ae"]

    df_train_minmax = df_train_minmax.drop(["pressure"], axis=1)
    df_val_minmax = df_val_minmax.drop(["pressure"], axis=1)
    df_test_minmax = df_test_minmax.drop(["pressure"], axis=1)
    #df_train_minmax = df_train_minmax.drop(["ae"], axis=1)
    #df_val_minmax = df_val_minmax.drop(["ae"], axis=1)
    #df_test_minmax = df_test_minmax.drop(["ae"], axis=1)

    np_train_minmax = df_train_minmax.to_numpy()
    np_pressure_train_minmax = df_pressure_train_minmax.to_numpy()
    np_ae_train_minmax = df_ae_train_minmax.to_numpy()

    np_val_minmax = df_val_minmax.to_numpy()
    np_pressure_val_minmax = df_pressure_val_minmax.to_numpy()
    np_ae_val_minmax = df_ae_val_minmax.to_numpy()

    np_test_minmax = df_test_minmax.to_numpy()
    np_pressure_test_minmax = df_pressure_test_minmax.to_numpy()
    np_ae_test_minmax = df_ae_test_minmax.to_numpy()


    # Do some extra manipulation that was not in the original preprocessing. 
    # More concretely, the time series that were loaded above will be shifted circularly. This will allow this LBFGS-based algorithm to run using batches.
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


    # Convert to torch tensors:
    train_input_torch = torch.from_numpy(np_train_minmax_shifted[:, :-1,:]).to(device)
    train_target_torch = torch.from_numpy(np_pressure_train_minmax_shifted[:, 1:]).to(device)

    val_input_torch = torch.from_numpy(np_val_minmax_shifted[:, :-1,:]).to(device)
    val_target_torch = torch.from_numpy(np_pressure_val_minmax_shifted[:, 1:]).to(device)

    test_input_torch = torch.from_numpy(np_test_minmax_shifted[:, :-1,:]).to(device)
    test_target_torch = torch.from_numpy(np_pressure_test_minmax_shifted[:, 1:]).to(device)

    print(np_train_minmax_shifted.shape)
    print(train_input_torch.shape)
    print(train_target_torch.shape)


    # Model
    seismology_model = DKRZSeismologyLSTM(num_hidden=NUMBER_OF_HIDDEN_UNITS).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.LBFGS(seismology_model.parameters(), lr=LEARNING_RATE, max_iter=LBFGS_MAX_ITERATIONS)


    # Training
    for step in range(num_train_steps):
        print("Step ", step + 1)
        optimizer.step(optimization_func)
        
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
            
            # Create figure:
            fig, ax = plt.subplots(figsize=(16, 10))
            x_data = range(len(np_pred_cpu_train)+len(np_pred_cpu_val))
            ax.plot(x_data[:len(np_pred_cpu_train)], np_pred_cpu_train, color="blue", label="Train prediction")
            ax.plot(x_data[:len(np_pred_cpu_train)], target_cpu_train, color="orange", label="Train ground truth")
            ax.plot(x_data[len(np_pred_cpu_train):len(np_pred_cpu_train)+len(np_pred_cpu_val)], np_pred_cpu_val, color="black", label="Validation prediction")
            ax.plot(x_data[len(np_pred_cpu_train):len(np_pred_cpu_train)+len(np_pred_cpu_val)], target_cpu_val, color="red", label="Validation ground truth")
            ax.set_title(f"Train Epoch {step}", fontsize=20)
            ax.set_xlabel("Time Index", fontsize=16)
            ax.set_ylabel("Pressure", fontsize=16)
            ax.legend()
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            text_box_str = f"val error={intermediary_val_loss.item()}" + f"\ntrain error={intermediary_train_loss.item()}"
            ax.text(0.35, 0.98, text_box_str, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
            
            # Save figure:
            fig.savefig(f"{results_location}/afterstep_{step}")
            plt.close()
            
            # Save model:
            torch.save(seismology_model, f"{results_location}/trained_seismology_model_afterstep_{step}.pth")