import pandas as pd
import numpy as np
import copy

from utils import *
from monai.transforms import (
    AsDiscrete,
    Activations,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Compose,
    RandRotate90d,
    Resized,
    MapTransform,
    ScaleIntensityd,
    RandBiasFieldd,
    RandGaussianSmoothd,
    RandAffined,
    RandRotated,
    Rand3DElasticd,
    RandZoomd,
    RandGaussianNoised,
    RandGibbsNoised,
    ShiftIntensityd,
    RandShiftIntensityd,
    RandGaussianSharpend,
    AdjustContrastd,
    RandAdjustContrastd,
    CenterSpatialCropd,
    SpatialPadd,
    Spacingd,
    NormalizeIntensityd,
    ConcatItemsd,
    RandFlipd,
    SpatialCropd,
)

PATH_TO_H5 = f"data/train_lung_preprocess_s300_350_350_h5v2_swapxy.h5"


def make_chaimeleon_data(time_col="survival_time_months", event_col="event", id_col="patient_id"):
    df_clinical_and_imaging = pd.read_csv(f"data/main_lung.csv")
    df_clinical_and_imaging = df_clinical_and_imaging.rename(
        columns={"Unnamed: 0": "patient_id"}
    )

    image_meta_cols = [
        "x_dim",
        "y_dim",
        "z_dim",
        "x_pixdim_spacing",
        "y_pixdim_spacing",
        "z_pixdim_spacing",
    ]
    df_image_meta = df_clinical_and_imaging[image_meta_cols]
    df_clinical = df_clinical_and_imaging.drop(columns=image_meta_cols)

    df_clinical = df_clinical.rename(
        columns={time_col: "time",
                 event_col: "event"}
    )

    num_uncensored = df_clinical["event"].sum()
    num_censored = len(df_clinical) - num_uncensored

    print(f"Number of uncensored patients: {num_uncensored}")
    print(f"Number of censored patients: {num_censored}")
    print(f"Total number of patients: {len(df_clinical)}")
    print("")

    percent_censored = np.round(num_censored / len(df_clinical) * 100, 2)
    percent_uncensored = np.round(num_uncensored / len(df_clinical) * 100, 2)

    print(f"Percent of censored patients: {percent_censored}%")
    print(f"Percent of uncensored patients: {percent_uncensored}%")

    # handle nan values
    df_clinical = df_clinical.replace({"Unknown": "X", "cNX": "X", "cTX": "X"})

    df_TNM_combo = copy.deepcopy(df_clinical)

    # T-stage
    df_TNM_combo = df_TNM_combo.replace(
        {
            "cT1": 1,
            "cT1a": 1,
            "cT1b": 1,
            "cT1c": 1,
            "cT2": 2,
            "cT2a": 2,
            "cT2b": 2,
            "cT3": 3,
            "cT4": 4,
            "X": "X",
        }
    )
    print(df_TNM_combo["clinical_category"].value_counts())
    print("")

    # N-stage
    df_TNM_combo = df_TNM_combo.replace(
        {"cN0": 0, "cN1": 1, "cN2": 2, "cN3": 3, "X": "X"}
    )
    print(df_TNM_combo["regional_nodes_category"].value_counts())
    print("")

    # M-stage
    df_TNM_combo = df_TNM_combo.replace(
        {"cM0": 0, "cM1": 1, "cM1a": 1, "cM1b": 1, "cM1c": 1, "X": "X"}
    )
    print(df_TNM_combo["metastasis_category"].value_counts())

    data_train = copy.deepcopy(df_TNM_combo)

    data_train.index = data_train[id_col]

    data_train = data_train.drop(columns=[id_col, "age"])

    return df_clinical, data_train


def lung_ct_transforms():
    # Define transforms
    xyz_dim = [112, 112, 130]
    # xyz_dim = [128, 128, 156]
    ClipCT_range = [-1000, 200]
    ClipCT_window_center_width = [False, False]

    prob = 0.25
    train_transforms = Compose(
        [
            LoadHdf5d(keys=["image", "path_to_h5"], image_only=True),
            ClipCT(
                keys="image",
                min=ClipCT_range[0],
                max=ClipCT_range[1],
                window_center=ClipCT_window_center_width[0],
                window_width=ClipCT_window_center_width[1],
            ),
            ScaleIntensityd(keys="image", minv=0, maxv=1),
            NormalizeIntensityd(keys="image"),
            Resized(
                keys="image",
                spatial_size=(xyz_dim[0], xyz_dim[1], xyz_dim[2]),
                mode="trilinear",
            ),
            RandRotate90d(keys="image", prob=prob),
            RandGaussianSmoothd(
                keys="image",
                sigma_x=[0.25, 1.5],
                sigma_y=[0.25, 1.5],
                sigma_z=[0.25, 1.5],
                prob=prob,
            ),
            RandAffined(
                keys="image",
                rotate_range=5,
                shear_range=0.5,
                translate_range=25,
                prob=prob,
            ),
            RandRotated(
                keys="image", range_x=0.1, range_y=0.1, range_z=0.1, prob=prob
            ),
            Rand3DElasticd(
                keys="image",
                sigma_range=(0.5, 1.5),
                magnitude_range=(0.5, 1.5),
                rotate_range=5,
                shear_range=0.5,
                translate_range=25,
                prob=prob,
            ),
            RandZoomd(keys="image", min=0.9, max=1.1, prob=prob),
            RandGaussianNoised(keys="image", mean=0.1, std=0.25, prob=prob),
            RandShiftIntensityd(keys="image", offsets=0.2, prob=prob),
            RandGaussianSharpend(
                keys="image",
                sigma1_x=[0.5, 1.0],
                sigma1_y=[0.5, 1.0],
                sigma1_z=[0.5, 1.0],
                sigma2_x=[0.5, 1.0],
                sigma2_y=[0.5, 1.0],
                sigma2_z=[0.5, 1.0],
                alpha=[10.0, 30.0],
                prob=prob,
            ),
            RandAdjustContrastd(keys="image", gamma=2.0, prob=prob),
        ],
    )

    val_transforms = Compose(
        [
            LoadHdf5d(keys=["image", "path_to_h5"], image_only=True),
            ClipCT(
                keys="image",
                min=ClipCT_range[0],
                max=ClipCT_range[1],
                window_center=ClipCT_window_center_width[0],
                window_width=ClipCT_window_center_width[1],
            ),
            ScaleIntensityd(keys="image", minv=0, maxv=1),
            NormalizeIntensityd(keys="image"),
            Resized(
                keys="image",
                spatial_size=(xyz_dim[0], xyz_dim[1], xyz_dim[2]),
                mode="trilinear",
            ),
        ]
    )

    return train_transforms, val_transforms