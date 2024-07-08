import pandas as pd
import numpy as np
import torch
import h5py

from monai.transforms import MapTransform

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier


class LoadHdf5d(MapTransform):
    def __init__(self, keys, image_only=False):
        """
        "image" contains patient_IDs
        "path_to_h5" contains folder path

        """
        super().__init__(keys)
        self.image_only = image_only

    def __call__(self, data):
        d = dict(data)
        filename = d["path_to_h5"]
        patient_ID = d["image"]
        hdf_obj2 = h5py.File(filename, "r")

        if self.image_only:
            d["image"] = hdf_obj2[patient_ID]["image"][:]
        else:
            meta = dict(hdf_obj2[patient_ID].attrs)
            d["image"] = (hdf_obj2[patient_ID]["image"][:], meta)  # image  # metadata

        hdf_obj2.close()

        return d


class ClipCT(MapTransform):
    """
    Convert labels to multi channels based on hecktor classes:
    label 1 is the tumor
    label 2 is the lymph node

    """

    def __init__(
        self, keys, min=-1024, max=1024, window_center=False, window_width=False
    ):
        """
        "image" contains patient_IDs
        "path_to_h5" contains folder path

        """
        super().__init__(keys)
        self.min = min
        self.max = max
        self.window_center = window_center
        self.window_width = window_width

        if self.window_center and self.window_width:
            self.min = window_center - window_width // 2  # minimum HU level
            self.max = window_center + window_width // 2  # maximum HU level

    def __call__(self, data):
        d = dict(data)
        d["image"] = np.clip(d["image"], self.min, self.max)
        return d


class ClipCTHecktor(MapTransform):
    """
    Convert labels to multi channels based on hecktor classes:
    label 1 is the tumor
    label 2 is the lymph node

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key == "ct":
                d[key] = torch.clip(d[key], min=-200, max=200)
            elif key == "pt":
                d[key] = torch.clip(d[key], d[key].min(), 5)
        return d
    
    
def imputer(data_train, impute_mode, impute_percent, seed, idx_train, cfg):
    cols_with_X = data_train.columns
    
    if impute_mode == "mode":
        for i in cols_with_X:
            if impute_percent != 30:
                data_train[i] = data_train[i].sample(frac=impute_percent / 100)

            data_train[i] = data_train[i].fillna(data_train[i].mode()[0])

    elif impute_mode == "mice":
        for i in cols_with_X:
            if impute_percent != 30:
                data_train[i] = data_train[i].sample(frac=impute_percent / 100)

        data_train = data_train.replace({"X": np.nan})

        data_train[cols_with_X] = data_train[cols_with_X].apply(
            lambda series: pd.Series(
                LabelEncoder().fit_transform(series[series.notnull()]),
                index=series[series.notnull()].index,
            )
        )

        # # MICE imputation
        imputer = IterativeImputer(random_state=seed)
        imputer.fit(data_train.loc[idx_train])  # shud do for train ONLY

        data_train[cols_with_X] = imputer.transform(data_train[cols_with_X])
        data_train[cols_with_X] = data_train[cols_with_X].round()

    elif impute_mode == "missforest":
        for i in cols_with_X:  
            if impute_percent != 30:
                data_train[i] = data_train[i].sample(frac=impute_percent / 100)

        data_train = data_train.replace({"X": np.nan})

        data_train[cols_with_X] = data_train[cols_with_X].apply(
            lambda series: pd.Series(
                LabelEncoder().fit_transform(series[series.notnull()]),
                index=series[series.notnull()].index,
            )
        )

        imputer = IterativeImputer(
            estimator=RandomForestClassifier(),
            initial_strategy="most_frequent",
            max_iter=10,
            random_state=seed,
        )

        # fit to train data only
        imputer.fit(data_train[cols_with_X].loc[idx_train])

        # transform full df (train + val)
        data_train[cols_with_X] = imputer.transform(data_train[cols_with_X])
        data_train[cols_with_X] = data_train[cols_with_X].round()

    elif impute_mode == "knn":
        for i in cols_with_X:
            if impute_percent != 30:
                data_train[i].loc[idx_train] = (
                    data_train[i]
                    .loc[idx_train]
                    .sample(frac=impute_percent / 100, random_state=0)
                )

        data_train = data_train.replace({"X": np.nan})

        data_train[cols_with_X] = data_train[cols_with_X].apply(
            lambda series: pd.Series(
                LabelEncoder().fit_transform(series[series.notnull()]),
                index=series[series.notnull()].index,
            )
        )

        imputer = KNNImputer(n_neighbors=1)

        # fit to train data only
        imputer.fit(data_train[cols_with_X].loc[idx_train])

        # transform full df (train + val)
        data_train[cols_with_X] = imputer.transform(data_train[cols_with_X])
        data_train[cols_with_X] = data_train[cols_with_X].round()

    print('data_train', data_train.columns)
    if cfg.dataset_name == "lung":
        if "MALE" in data_train["gender"].unique():
            data_train.loc[len(data_train.index)] = ["MALE", "X", "X", "X", "X"]
        else:
            data_train.loc[len(data_train.index)] = [0, "X", "X", "X", "X"]

    return data_train


def one_hot_encode(data_train, one_hot_cols, cfg):
    data_train = data_train[one_hot_cols]

    # handle nan values
    data_train = data_train.fillna("X")

    if cfg.dataset_name == "lung":
        # do NOT drop_first in columns with nan/unknown values/X
        data_train = pd.get_dummies(
            data_train, columns=one_hot_cols, drop_first=False
        )

    elif cfg.dataset_name == "hecktor":        
        for i in data_train.columns:
            print(i, data_train[i].value_counts())

        data_train = pd.get_dummies(
            data_train, columns=one_hot_cols, drop_first=False
        )

    # drop cols with X
    cols_with_X = [i for i in data_train.columns if "_X" in i]
    data_train = data_train.drop(columns=cols_with_X)

    return data_train
