import pandas as pd
import numpy as np
import math

exp4_pressure_df = pd.read_csv("../../cloudcontainer/Parameters_downloaded_on_jan5/wgrn04/ae_wgn04_150_axial.x_mtsraw.csv")
exp4_pressure_df.rename(columns={"t_dax": "T_DAX_pressure"}, inplace=True)
exp4_pressure_df = exp4_pressure_df[["T_DAX_pressure", "Ch_1_Kraft_1000kN"]]

wgrn04_features_path = "../../cloudcontainer/wgrn04_features.hdf5"
exp4_precomputed_features_df = pd.HDFStore(wgrn04_features_path, mode='r').select("wgrn04")
exp4_precomputed_features_df.rename(columns={"T_DAX": "T_DAX_features"}, inplace=True)
print("precomputed features: ", exp4_precomputed_features_df.shape)

# Perform ceiling operation, with a 0.5 step size:
ceiled_pressure_time = exp4_pressure_df['T_DAX_pressure'].apply(lambda x: math.modf(x)[1] + 0.5 if math.modf(x)[0] < 0.5 else math.modf(x)[1] + 1)
exp4_pressure_df['pressure_ceiled_time'] = ceiled_pressure_time
print(exp4_pressure_df.head())

exp4_pressure_df_aggregated_df = exp4_pressure_df.groupby('pressure_ceiled_time').agg({
    'Ch_1_Kraft_1000kN':['max']
}).reset_index()
print("pressure after aggregation: ", exp4_pressure_df_aggregated_df)

precomputed_features_and_pressure_df = pd.merge(exp4_pressure_df_aggregated_df, exp4_precomputed_features_df, how='inner', left_on="pressure_ceiled_time", right_on="T_DAX_features")
precomputed_features_and_pressure_df = precomputed_features_and_pressure_df.drop(['T_DAX_features'], axis=1)
precomputed_features_and_pressure_df.columns.values[0] = "time"
precomputed_features_and_pressure_df.columns.values[1] = "pressure"
print(precomputed_features_and_pressure_df.head())
print(precomputed_features_and_pressure_df.shape)

precomputed_features_and_pressure_df.to_hdf("../../cloudcontainer/wgrn04_features_and_pressure.hdf5", key='wgrn04', mode='w')