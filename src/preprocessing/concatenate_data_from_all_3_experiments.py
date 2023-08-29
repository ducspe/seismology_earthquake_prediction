import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler, StandardScaler

wgrn04_hdf5_path = "../../cloudcontainer/wgrn04_ae_features_and_pressure.hdf5"
wgrn05_hdf5_path = "../../cloudcontainer/wgrn05_ae_features_and_pressure.hdf5"
wgrn07_hdf5_path = "../../cloudcontainer/wgrn07_ae_features_and_pressure.hdf5"

experiment4 = pd.HDFStore(wgrn04_hdf5_path, mode='r')
experiment5 = pd.HDFStore(wgrn05_hdf5_path, mode='r')
experiment7 = pd.HDFStore(wgrn07_hdf5_path, mode='r')

experiment4_df = experiment4.select("wgrn04")
# Remove the meaningless tail:
experiment4_df = experiment4_df[:14400]
experiment5_df = experiment5.select("wgrn05")
experiment7_df = experiment7.select("wgrn07")

print("exp 4: ", experiment4_df.head())
print("exp 5: ", experiment5_df.head())
print("exp 7: ", experiment7_df.head())

print("exp 4: ", experiment4_df.shape)
print("exp 5: ", experiment5_df.shape)
print("exp 7: ", experiment7_df.shape)

exp4_cols = sorted(experiment4_df.columns.tolist())
exp5_cols = sorted(experiment5_df.columns.tolist())
exp7_cols = sorted(experiment7_df.columns.tolist())

assert exp4_cols == exp5_cols
assert exp5_cols == exp7_cols

frames = [experiment4_df, experiment5_df, experiment7_df]
concatenated_dfs = pd.concat(frames).reset_index()
concatenated_dfs.index.name = "timeindex"
concatenated_dfs = concatenated_dfs.drop(["index", "time"], axis=1)

concatenated_dfs_mean_instead_of_NaN = concatenated_dfs.fillna(concatenated_dfs.mean()) 
concatenated_dfs_median_instead_of_NaN = concatenated_dfs.fillna(concatenated_dfs.median()) 
concatenated_dfs_zero_instead_of_NaN = concatenated_dfs.fillna(0) 

# Split into train/val/test:
train_len = math.floor(len(concatenated_dfs)*0.8)
val_len = math.floor(len(concatenated_dfs)*0.1)
test_len = math.floor(len(concatenated_dfs)*0.1)

train_df_zero = concatenated_dfs_zero_instead_of_NaN[:train_len]
val_df_zero = concatenated_dfs_zero_instead_of_NaN[train_len:train_len + val_len]
test_df_zero = concatenated_dfs_zero_instead_of_NaN[train_len + val_len:]

train_df_mean = concatenated_dfs_mean_instead_of_NaN[:train_len]
val_df_mean = concatenated_dfs_mean_instead_of_NaN[train_len:train_len + val_len]
test_df_mean = concatenated_dfs_mean_instead_of_NaN[train_len + val_len:]

train_df_median = concatenated_dfs_median_instead_of_NaN[:train_len]
val_df_median = concatenated_dfs_median_instead_of_NaN[train_len:train_len + val_len]
test_df_median = concatenated_dfs_median_instead_of_NaN[train_len + val_len:]

# Create minmax normalized splits:
minmaxscaler = MinMaxScaler()
minmaxscaler.fit(train_df_zero)
train_df_zero_minmax = pd.DataFrame(minmaxscaler.transform(train_df_zero), index=train_df_zero.index, columns=train_df_zero.columns)
val_df_zero_minmax = pd.DataFrame(minmaxscaler.transform(val_df_zero), index=val_df_zero.index, columns=val_df_zero.columns)
test_df_zero_minmax = pd.DataFrame(minmaxscaler.transform(test_df_zero), index=test_df_zero.index, columns=test_df_zero.columns)

train_df_mean_minmax = pd.DataFrame(minmaxscaler.transform(train_df_mean), index=train_df_mean.index, columns=train_df_mean.columns)
val_df_mean_minmax = pd.DataFrame(minmaxscaler.transform(val_df_mean), index=val_df_mean.index, columns=val_df_mean.columns)
test_df_mean_minmax = pd.DataFrame(minmaxscaler.transform(test_df_mean), index=test_df_mean.index, columns=test_df_mean.columns)

train_df_median_minmax = pd.DataFrame(minmaxscaler.transform(train_df_median), index=train_df_median.index, columns=train_df_median.columns)
val_df_median_minmax = pd.DataFrame(minmaxscaler.transform(val_df_median), index=val_df_median.index, columns=val_df_median.columns)
test_df_median_minmax = pd.DataFrame(minmaxscaler.transform(test_df_median), index=test_df_median.index, columns=test_df_median.columns)

# Create Z-normalization splits:
zscaler = StandardScaler()
zscaler.fit(train_df_zero)
train_df_zero_zscale = pd.DataFrame(zscaler.transform(train_df_zero), index=train_df_zero.index, columns=train_df_zero.columns)
val_df_zero_zscale = pd.DataFrame(zscaler.transform(val_df_zero), index=val_df_zero.index, columns=val_df_zero.columns)
test_df_zero_zscale = pd.DataFrame(zscaler.transform(test_df_zero), index=test_df_zero.index, columns=test_df_zero.columns)

train_df_mean_zscale = pd.DataFrame(zscaler.transform(train_df_mean), index=train_df_mean.index, columns=train_df_mean.columns)
val_df_mean_zscale = pd.DataFrame(zscaler.transform(val_df_mean), index=val_df_mean.index, columns=val_df_mean.columns)
test_df_mean_zscale = pd.DataFrame(zscaler.transform(test_df_mean), index=test_df_mean.index, columns=test_df_mean.columns)

train_df_median_zscale = pd.DataFrame(zscaler.transform(train_df_median), index=train_df_median.index, columns=train_df_median.columns)
val_df_median_zscale = pd.DataFrame(zscaler.transform(val_df_median), index=val_df_median.index, columns=val_df_median.columns)
test_df_median_zscale = pd.DataFrame(zscaler.transform(test_df_median), index=test_df_median.index, columns=test_df_median.columns)

# Save all data frames:
concatenated_dfs.to_hdf("../../cloudcontainer/concatenated_experiments.hdf5", key='alldata_with_NaN', mode='w')

concatenated_dfs_zero_instead_of_NaN.to_hdf("../../cloudcontainer/concatenated_experiments.hdf5", key='alldata_zero_instead_of_NaN', mode='a')
train_df_zero.to_hdf("../../cloudcontainer/concatenated_experiments.hdf5", key='alldata_zero_instead_of_NaN/train', mode='a')
val_df_zero.to_hdf("../../cloudcontainer/concatenated_experiments.hdf5", key='alldata_zero_instead_of_NaN/val', mode='a')
test_df_zero.to_hdf("../../cloudcontainer/concatenated_experiments.hdf5", key='alldata_zero_instead_of_NaN/test', mode='a')
train_df_zero_minmax.to_hdf("../../cloudcontainer/concatenated_experiments.hdf5", key='alldata_zero_instead_of_NaN/train/minmax', mode='a')
val_df_zero_minmax.to_hdf("../../cloudcontainer/concatenated_experiments.hdf5", key='alldata_zero_instead_of_NaN/val/minmax', mode='a')
test_df_zero_minmax.to_hdf("../../cloudcontainer/concatenated_experiments.hdf5", key='alldata_zero_instead_of_NaN/test/minmax', mode='a')
train_df_zero_zscale.to_hdf("../../cloudcontainer/concatenated_experiments.hdf5", key='alldata_zero_instead_of_NaN/train/zscale', mode='a')
val_df_zero_zscale.to_hdf("../../cloudcontainer/concatenated_experiments.hdf5", key='alldata_zero_instead_of_NaN/val/zscale', mode='a')
test_df_zero_zscale.to_hdf("../../cloudcontainer/concatenated_experiments.hdf5", key='alldata_zero_instead_of_NaN/test/zscale', mode='a')

concatenated_dfs_mean_instead_of_NaN.to_hdf("../../cloudcontainer/concatenated_experiments.hdf5", key='alldata_mean_instead_of_NaN', mode='a')
train_df_mean.to_hdf("../../cloudcontainer/concatenated_experiments.hdf5", key='alldata_mean_instead_of_NaN/train', mode='a')
val_df_mean.to_hdf("../../cloudcontainer/concatenated_experiments.hdf5", key='alldata_mean_instead_of_NaN/val', mode='a')
test_df_mean.to_hdf("../../cloudcontainer/concatenated_experiments.hdf5", key='alldata_mean_instead_of_NaN/test', mode='a')
train_df_mean_minmax.to_hdf("../../cloudcontainer/concatenated_experiments.hdf5", key='alldata_mean_instead_of_NaN/train/minmax', mode='a')
val_df_mean_minmax.to_hdf("../../cloudcontainer/concatenated_experiments.hdf5", key='alldata_mean_instead_of_NaN/val/minmax', mode='a')
test_df_mean_minmax.to_hdf("../../cloudcontainer/concatenated_experiments.hdf5", key='alldata_mean_instead_of_NaN/test/minmax', mode='a')
train_df_mean_zscale.to_hdf("../../cloudcontainer/concatenated_experiments.hdf5", key='alldata_mean_instead_of_NaN/train/zscale', mode='a')
val_df_mean_zscale.to_hdf("../../cloudcontainer/concatenated_experiments.hdf5", key='alldata_mean_instead_of_NaN/val/zscale', mode='a')
test_df_mean_zscale.to_hdf("../../cloudcontainer/concatenated_experiments.hdf5", key='alldata_mean_instead_of_NaN/test/zscale', mode='a')

concatenated_dfs_median_instead_of_NaN.to_hdf("../../cloudcontainer/concatenated_experiments.hdf5", key='alldata_median_instead_of_NaN', mode='a')
train_df_median.to_hdf("../../cloudcontainer/concatenated_experiments.hdf5", key='alldata_median_instead_of_NaN/train', mode='a')
val_df_median.to_hdf("../../cloudcontainer/concatenated_experiments.hdf5", key='alldata_median_instead_of_NaN/val', mode='a')
test_df_median.to_hdf("../../cloudcontainer/concatenated_experiments.hdf5", key='alldata_median_instead_of_NaN/test', mode='a')
train_df_median_minmax.to_hdf("../../cloudcontainer/concatenated_experiments.hdf5", key='alldata_median_instead_of_NaN/train/minmax', mode='a')
val_df_median_minmax.to_hdf("../../cloudcontainer/concatenated_experiments.hdf5", key='alldata_median_instead_of_NaN/val/minmax', mode='a')
test_df_median_minmax.to_hdf("../../cloudcontainer/concatenated_experiments.hdf5", key='alldata_median_instead_of_NaN/test/minmax', mode='a')
train_df_median_zscale.to_hdf("../../cloudcontainer/concatenated_experiments.hdf5", key='alldata_median_instead_of_NaN/train/zscale', mode='a')
val_df_median_zscale.to_hdf("../../cloudcontainer/concatenated_experiments.hdf5", key='alldata_median_instead_of_NaN/val/zscale', mode='a')
test_df_median_zscale.to_hdf("../../cloudcontainer/concatenated_experiments.hdf5", key='alldata_median_instead_of_NaN/test/zscale', mode='a')

# Close opened files:
experiment4.close()
experiment5.close()
experiment7.close()