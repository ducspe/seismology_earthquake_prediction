import pandas as pd
import numpy as np
import math

exp4_ae_df = pd.read_csv("../../cloudcontainer/Parameters_downloaded_on_jan5/wgrn04/ae_wgn04_150_axial.x_ae_01.csv")

print(exp4_ae_df.head())

pressure_and_features_path = "../../cloudcontainer/wgrn04_features_and_pressure.hdf5"
pressure_and_features_df = pd.HDFStore(pressure_and_features_path, mode="r").select("wgrn04")
print(pressure_and_features_df.head())

# I treat the acoustic emission/ae as a feature, so I will do the floor operation with a 0.5 step, just like I did inside "process_wgrn05_data.py".
# When I merged with the pressure table inside experiment4_merge_with_pressure_table.py, I did a ceiling operation so that I have:
# past features -> to predict future pressure.
floored_ae_time = exp4_ae_df["t_dax0"].apply(lambda x: math.modf(x)[1] if math.modf(x)[0] < 0.5 else math.modf(x)[1] + 0.5)

exp4_ae_df["ae_floored_time"] = floored_ae_time

exp4_ae_df_aggregated = exp4_ae_df.groupby("ae_floored_time").agg({
    'adjamp':['mean']
}).reset_index()

print("ae after aggregation: ", exp4_ae_df_aggregated)

ae_features_and_pressure_df = pd.merge(exp4_ae_df_aggregated, pressure_and_features_df, how="inner", left_on="ae_floored_time", right_on="time")
ae_features_and_pressure_df = ae_features_and_pressure_df.drop(['time'], axis=1)
ae_features_and_pressure_df.columns.values[0] = "time"
ae_features_and_pressure_df.columns.values[1] = "ae"
print(ae_features_and_pressure_df.shape)
print(ae_features_and_pressure_df.head())

ae_features_and_pressure_df.to_hdf("../../cloudcontainer/wgrn04_ae_features_and_pressure.hdf5", key='wgrn04', mode='w')