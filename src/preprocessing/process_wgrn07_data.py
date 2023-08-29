import pandas as pd
import numpy as np
import h5py
import os
import functools as ft

# https://stackoverflow.com/questions/44883175/how-to-list-all-datasets-in-h5py-file
def get_dataset_keys(f):
    keys = []
    f.visit(lambda key : keys.append(key) if isinstance(f[key], h5py.Dataset) else None)
    return keys

if __name__ == "__main__":

    features_dict = {}

    wgrn07_base_path = "../../cloudcontainer/Parameters_downloaded_on_jan5/wgrn07/alt"

    # List all file names in alt directory:
    all_files_list = os.listdir(wgrn07_base_path)
    print("All files: ", all_files_list)

    for found_file in all_files_list:
        file_path = f"{wgrn07_base_path}/{found_file}"
        # filter out directories and continue this for loop only with files.
        if os.path.isfile(file_path):
            file = h5py.File(file_path)
        else:
            continue

        all_datasets_list = get_dataset_keys(file)

        for dataset_path in all_datasets_list:
            feature_file = h5py.File(file_path)[dataset_path]
            features_dict[dataset_path] =  pd.DataFrame(feature_file).T
            features_dict[dataset_path].columns = feature_file.attrs["columns"].split('|')
            # Rename columns at index 1. This will be useful to differentiate between column names after all dataframes will be merged.
            features_dict[dataset_path].columns.values[1] = f"{dataset_path}_{features_dict[dataset_path].columns[1]}"

    filtered_columns_05sec_dict = {key:features_dict[key].iloc[:, [0,1]] for key in features_dict.keys() if "0.5" in key}

    # Next I need to merge the dataframes at each key using:
    # https://stackoverflow.com/questions/23668427/pandas-three-way-joining-multiple-dataframes-on-columns

    dfs = [filtered_columns_05sec_dict[key] for key in filtered_columns_05sec_dict.keys()]
    df_merged = ft.reduce(lambda left, right: pd.merge(left, right, on='T_DAX'), dfs)

    df_final = df_merged
    print(df_final.shape)
    print(df_final.head())

    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_hdf.html
    df_final.to_hdf("../../cloudcontainer/wgrn07_features.hdf5", key='wgrn07', mode='w')