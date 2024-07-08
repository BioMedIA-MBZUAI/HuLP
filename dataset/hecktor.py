import pandas as pd
import numpy as np
import copy

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
from utils import *


def make_hecktor_data(time_col="time", event_col="event", id_col="PatientID_x"):
    # preprocess hecktor data
    df_clinical = pd.read_csv("splits/hecktor2021_ehr_imgpath_from_hecktor2022.csv")

    df_clinical["PatientID_x"] = [
        f"{i}-{j:03}" for i, j in zip(df_clinical["Center"], df_clinical["PID"])
    ]
    df_clinical = df_clinical.rename(
        columns={"Gender (1=M,0=F)": "Gender", 
                 "HPV status (0=-, 1=+)": "HPV_status",
                 time_col: "time",
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

    df = copy.deepcopy(df_clinical)

    # combine TNM
    df = df.replace(
        {
            "T-stage": {"T1": 1, "T2": 2, "T3": 3, "T4": 4, "T4a": 4, "T4b": 4},
            "N-stage": {
                "N0": 0,
                "N1": 1,
                "N2": 2,
                "N2a": 2,
                "N2b": 2,
                "N2c": 2,
                "N3": 3,
            },
            "M-stage": {"M0": 0, "M1": 1},
        }
    )

    for i in ["T-stage", "N-stage", "M-stage"]:
        print(df[i].value_counts())
        print("")

    print(df.isna().sum())
    print(np.round(df.isna().sum() / len(df) * 100, 2))

    data_train = copy.deepcopy(df)

    data_train.index = data_train[id_col]
    # data_centers_train = data_train["Center"]

    print(data_train)

    data_train = data_train.drop(
        columns=[id_col, "PID", "Center", "Age", "Path"]
    )

    return df_clinical, data_train


def hecktor_ct_pet_transforms():
    # Define transforms
    xyz_dim = [176, 176, 144]

    prob = 0.25
    train_transforms = Compose(
        [
            LoadImaged(
                keys=["ct", "pt"], ensure_channel_first=True, image_only=False
            ),
            Orientationd(keys=["ct", "pt"], axcodes="PLS"),
            SpatialPadd(
                keys=["ct", "pt"],
                spatial_size=(xyz_dim[0], xyz_dim[1], xyz_dim[2]),
                method="symmetric",
            ),
            CenterSpatialCropd(
                keys=["ct", "pt"], roi_size=(xyz_dim[0], xyz_dim[1], xyz_dim[2])
            ),
            ClipCTHecktor(keys=["ct", "pt"]),
            Resized(
                # keys=["ct", "pt"], spatial_size=(128, 128, 112), mode="trilinear"
                keys=["ct", "pt"], spatial_size=(112, 112, 130), mode="trilinear"
            ),
            ConcatItemsd(keys=["pt", "ct"], name="image"),
            NormalizeIntensityd(keys=["image"], channel_wise=True),
            RandFlipd(
                keys=["image"],
                spatial_axis=[0],
                prob=prob,
            ),
            RandFlipd(
                keys=["image"],
                spatial_axis=[1],
                prob=prob,
            ),
            RandFlipd(
                keys=["image"],
                spatial_axis=[2],
                prob=prob,
            ),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["ct", "pt"], ensure_channel_first=True),
            Orientationd(keys=["ct", "pt"], axcodes="PLS"),
            SpatialPadd(
                keys=["ct", "pt"],
                spatial_size=(xyz_dim[0], xyz_dim[1], xyz_dim[2]),
                method="symmetric",
            ),
            CenterSpatialCropd(
                keys=["ct", "pt"], roi_size=(xyz_dim[0], xyz_dim[1], xyz_dim[2])
            ),
            ClipCTHecktor(keys=["ct", "pt"]),
            Resized(
                keys=["ct", "pt"], spatial_size=(128, 128, 112), mode="trilinear"
            ),
            ConcatItemsd(keys=["pt", "ct"], name="image"),
            NormalizeIntensityd(keys=["image"], channel_wise=True),
        ]
    )

    return train_transforms, val_transforms