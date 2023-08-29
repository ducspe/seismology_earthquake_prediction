import pandas as pd
import numpy as np
import math

exp7_pressure_df = pd.read_csv("../../cloudcontainer/Parameters_downloaded_on_jan5/wgrn07/ae_wgr07_150_axial.x_mtsraw.csv")
exp7_pressure_df.rename(columns={"t_dax":"T_DAX_pressure"}, inplace=True)
exp7_pressure_df = exp7_pressure_df[["T_DAX_pressure", "Ch_1_Kraft_1000kN"]]
print(exp7_pressure_df.head())

wgrn07_features_path = "../../cloudcontainer/wgrn07_features.hdf5"
exp7_precomputed_features_df = pd.HDFStore(wgrn07_features_path, mode='r').select("wgrn07")
exp7_precomputed_features_df.rename(columns={"T_DAX": "T_DAX_features"}, inplace=True)
print("precomputed features: ", exp7_precomputed_features_df.shape)

# Perform ceiling operation, with a 0.5 step size:
ceiled_pressure_time = exp7_pressure_df['T_DAX_pressure'].apply(lambda x: math.modf(x)[1] + 0.5 if math.modf(x)[0] < 0.5 else math.modf(x)[1] + 1)
exp7_pressure_df['pressure_ceiled_time'] = ceiled_pressure_time

exp7_pressure_df_aggregated_df = exp7_pressure_df.groupby('pressure_ceiled_time').agg({
    'Ch_1_Kraft_1000kN':['max']
}).reset_index()
print("pressure after aggregation: ", exp7_pressure_df_aggregated_df)

precomputed_features_and_pressure_df = pd.merge(exp7_pressure_df_aggregated_df, exp7_precomputed_features_df, how='inner', left_on="pressure_ceiled_time", right_on="T_DAX_features")
precomputed_features_and_pressure_df = precomputed_features_and_pressure_df.drop(['T_DAX_features'], axis=1)
precomputed_features_and_pressure_df.columns.values[0] = "time"
precomputed_features_and_pressure_df.columns.values[1] = "pressure"
print(precomputed_features_and_pressure_df.head())
print(precomputed_features_and_pressure_df.shape)

precomputed_features_and_pressure_df.to_hdf("../../cloudcontainer/wgrn07_features_and_pressure.hdf5", key='wgrn07', mode='w')