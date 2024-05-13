import logging
import os
import sys
import shutil

# from .utils import loader
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer

import monai
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import DataLoader, ImageDataset, CSVSaver
from monai.utils import set_determinism

from pycox.evaluation import EvalSurv

from einops import rearrange

# https://docs.monai.io/en/0.3.0/transforms.html
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

from monai.data import decollate_batch
from monai.metrics import ROCAUCMetric
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier

from datetime import datetime
from torch.optim.lr_scheduler import OneCycleLR

# from optimizer import NovoGrad

from sklearn.metrics import brier_score_loss, roc_auc_score

# from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index

sys.path.append("torchmtlr")
from torchmtlr import (
    MTLR,
    mtlr_neg_log_likelihood,
    mtlr_survival,
    mtlr_survival_at_times,
    mtlr_risk,
    mtlr_survival_bin_neg_log_likelihood,  # MR ADDED
)
from torchmtlr.utils import make_time_bins, encode_survival

# from coral_pytorch.losses import corn_loss
# from coral_pytorch.dataset import corn_label_from_logits, levels_from_labelbatch

import copy
import wandb
import argparse
import random
import json

from utils import (
    LoadHdf5d,
    ClipCT,
    ClipCTHecktor,
    deephit_encode_survival,
    predict_surv_df,
    prognosis_ranking_loss,
    brier_score_at_times,
    roc_auc_at_times,
    compute_metrics,
    deephit_loss,
    deepmtlr_loss,
)
from prognosis_models import PrognosisModel, Dual_MTLR

parser = argparse.ArgumentParser()

parser.add_argument("--aws", type=str, default="False", help="If using AWS SageMaker")
parser.add_argument("--fold", type=int, default=0, help="")
parser.add_argument("--seed", type=int, default=42, help="")
parser.add_argument("--image_only", type=str, default="True", help="")
parser.add_argument("--impute", type=str, default="None", help="")
parser.add_argument(
    "--model",
    type=str,
    default="PrognosisModel",
    choices=["PrognosisModel", "Dual_MTLR"],
)
parser.add_argument(
    "--impute_percent", type=int, default=30, help="value between 0 and 100"
)
parser.add_argument(
    "--mtlr_loss", type=str, default="deephit", choices=["deephit", "original_mtlr"]
)
parser.add_argument(
    "--data_dir", type=str, default="/l/users/muhammad.ridzuan/CHAIMELEON"
)
parser.add_argument(
    "--dataset_name", type=str, default="lung", choices=["lung", "hecktor"]
)
parser.add_argument("--output_dir", type=str, help="path to checkpoint and results")

cfg = parser.parse_args()
SEED = cfg.seed
fold = cfg.fold
impute = cfg.impute
impute_percent = cfg.impute_percent

if cfg.aws == "True":
    cfg.data_dir = "s3://mbz-hpc-aws-phd/AROARU6TOWKRSSJLLLFHK:Mai.Kassem@mbzuai.ac.ae/datasets/hecktor2022_2.5x2.5x3"
    cfg.output_dir = None

if cfg.image_only == "True":
    image_only = True
elif cfg.image_only == "False":
    image_only = False

if cfg.model == "Dual_MTLR":
    dual_mtlr = True
else:
    dual_mtlr = False
PATH_TO_H5 = f"{cfg.data_dir}/data/train_lung_preprocess_s300_350_350_h5v2_swapxy.h5"

set_determinism(SEED)
# Setting the seed
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
# if cfg.cuda:
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

####### init ##########
random_intervention_thresh = False
#######################

########################### LOADER ###########################
pin_memory = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.backends.cuda.matmul.allow_tf32 = False # in PyTorch 1.12 and later.
# torch.backends.cudnn.allow_tf32 = True
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
print_config()

current_datetime = datetime.now()
current_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")
current_datetime = current_datetime[2:]
print(current_datetime)

if cfg.dataset_name == "lung":
    df_clinical_and_imaging = pd.read_csv(f"{cfg.data_dir}/data/main_lung.csv")
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
        columns={"survival_time_months": "time", "event": "event"}
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

    data_train.index = data_train["patient_id"]
    data_event_train = data_train["event"]  # .values
    data_time_train = data_train["time"]  # .values
    eval_times = np.quantile(
        data_train.loc[data_train["event"] == 1, "time"], [0.25, 0.5, 0.75]
    ).astype("int")

    data_train = data_train.drop(columns=["patient_id", "time", "event", "age"])

elif cfg.dataset_name == "hecktor":
    # preprocess hecktor data
    if cfg.aws == "True":
        df_clinical = pd.read_csv("splits/hecktor2021_ehr_imgpath_from_hecktor2022.csv")
    else:
        df_clinical = pd.read_csv(
            "/l/users/muhammad.ridzuan/Survival_Analysis/Dataset/HECKTOR2021/hecktor2021_ehr_imgpath_from_hecktor2022.csv"
        )
    df_clinical["PatientID_x"] = [
        f"{i}-{j:03}" for i, j in zip(df_clinical["Center"], df_clinical["PID"])
    ]
    df_clinical = df_clinical.rename(
        columns={"Gender (1=M,0=F)": "Gender", "HPV status (0=-, 1=+)": "HPV_status"}
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

    # handle nan values
    df = df.fillna("X")

    data_train = copy.deepcopy(df)

    data_train.index = data_train["PatientID_x"]
    data_time_train = data_train["time"]
    data_event_train = data_train["event"]
    data_centers_train = data_train["Center"]

    print(data_train)

    eval_times = np.quantile(
        data_train.loc[data_train["event"] == 1, "time"], [0.25, 0.5, 0.75]
    ).astype("int")

    data_train = data_train.drop(
        columns=["time", "event", "PatientID_x", "PID", "Center", "Age", "Path"]
    )

dropout = 0
pre_prognosis = "concept"  # "concept" #"concept" #"class" #"X"
prognosis = "relu_lin"  # "relu_lin_sig" #"relu_mtlr_sig"
soft_prob = 1.0  # 0.9
concept_act = "leakyrelu"

if cfg.model == "PrognosisModel":
    intervention_cem = True
    if intervention_cem:
        random_intervention_thresh = 0.25  # 0.7 #0.25
else:
    intervention_cem = False

prog_loss_wt = 5  # 2 #1 #5
num_bins = None  # 20

ordinal_loss = "X"  # "X" #"corn"
mtlr_loss = cfg.mtlr_loss  # "prog_corn_rank" #"deephit" #None
mtlr_bin_alpha = None
mtlr_bin_penalize_censored = None
sig = False
number_warmup_epochs = 5
free_concept_pred = False
emb_size = 64
vSubmit = False  # "Yes"
if "mtlr" in prognosis and "deephit" not in mtlr_loss:
    mtlr_loss = "original_mtlr"

    if mtlr_loss == "bin_loss":
        mtlr_bin_alpha = 0.3  # lower alpha gives more weight to bin loss
        mtlr_bin_penalize_censored = True

    elif mtlr_loss == "original_mtlr":
        mtlr_bin_alpha = 0.0

lrs = [1e-3]
for lr in lrs:
    batch_size = 32

    if vSubmit:
        current_datetime = f"{current_datetime}_vSubmit_dense121_250ep_deephit_nocomboSmoking_dropX_CosineAnneal_ValcorrDataSwapxyV2_negVal_wConceptAct_actualProperCEM_AdamW_{num_bins}bins"  # reproduce_231214_103241_withDropout" #231214_103241" #231213_150735" #231214_102755"
    else:
        current_datetime = f"{current_datetime}_5CV_imputewGender_allNaNinValSplit_{impute}_{impute_percent}_concept_{mtlr_loss}_seed{SEED}_fold{fold}"

    root_dir = f"{cfg.output_dir}/lung_exps/prognosis/MICCAI/results/impute_concept_deephit_5CV/{current_datetime}"

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    if cfg.dataset_name == "lung":
        # Define transforms
        xyz_dim = [112, 112, 130]
        # xyz_dim = [128, 128, 156]
        ClipCT_range = [-1000, 200]
        ClipCT_window_center_width = [False, False]

        prob = 0.25  # 0.15
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
    elif cfg.dataset_name == "hecktor":
        # Define transforms
        xyz_dim = [176, 176, 144]
        ClipCT_range = [False, False]
        ClipCT_window_center_width = [False, False]

        prob = 0.25  # 0.15
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
                    keys=["ct", "pt"], spatial_size=(128, 128, 112), mode="trilinear"
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

    patientIDs = data_train.index.to_list()
    print("patientIDs", len(patientIDs))

    # test_size = 0.15
    # train_id, val_id = train_test_split(patientIDs, stratify=data_event_train, test_size=test_size, random_state=SEED)

    if cfg.dataset_name == "lung":
        with open(
            # lung_exps/prognosis/chaimeleon_allNaNinVal_lungCancer_5CV_patientSplit.json
            f"{cfg.data_dir}/lung_exps/prognosis/chaimeleon_lungCancer_5CV_patientSplit.json",
            "r",
        ) as f:
            patientSplit = json.load(f)
        train_id = patientSplit[f"Fold{fold}"]["train"]
        val_id = patientSplit[f"Fold{fold}"]["val"]

    elif cfg.dataset_name == "hecktor":
        with open(
            f"splits/hecktor2021_5CV_patientPathSplit_seed0.json",
            "r",
        ) as f:
            patientSplit = json.load(f)
        ct_train = patientSplit[f"Fold{fold}"]["train"]
        ct_val = patientSplit[f"Fold{fold}"]["val"]

        train_id = [i.split("/")[-1].split("_")[0] for i in ct_train]
        val_id = [i.split("/")[-1].split("_")[0] for i in ct_val]

    print("train_id", len(train_id))
    print("val_id", len(val_id))

    if cfg.dataset_name == "lung":
        ct_train = train_id
        ct_train = [i for i in ct_train if "case_0346" not in i]
        ct_val = val_id
        ct_val = [i for i in ct_val if "case_0346" not in i]

        if impute == "mode":
            # for i in ["smoking_status", "clinical_category", "regional_nodes_category", "metastasis_category"]: #df_clinical.columns[5:]:
            for i in df_clinical.columns[4:]:
                if impute_percent != 30:
                    data_train[i] = data_train[i].sample(frac=impute_percent / 100)

                data_train[i] = data_train[i].fillna(data_train[i].mode()[0])

        elif impute == "mice":
            cols_with_X = df_clinical.columns[4:].tolist()
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
            imputer = IterativeImputer(random_state=SEED)
            imputer.fit(data_train.loc[ct_train])  # shud do for train ONLY

            data_train[cols_with_X] = imputer.transform(data_train[cols_with_X])
            data_train[cols_with_X] = data_train[cols_with_X].round()

        elif impute == "missforest":
            cols_with_X = df_clinical.columns[4:].tolist()
            for i in cols_with_X:  # df_clinical.columns[5:]:
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
                random_state=SEED,
            )

            # fit to train data only
            imputer.fit(data_train[cols_with_X].loc[ct_train])

            # transform full df (train + val)
            data_train[cols_with_X] = imputer.transform(data_train[cols_with_X])
            data_train[cols_with_X] = data_train[cols_with_X].round()

        elif impute == "knn":
            cols_with_X = df_clinical.columns[4:].tolist()
            for i in cols_with_X:
                if impute_percent != 30:
                    data_train[i].loc[ct_train] = (
                        data_train[i]
                        .loc[ct_train]
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
            imputer.fit(data_train[cols_with_X].loc[ct_train])

            # transform full df (train + val)
            data_train[cols_with_X] = imputer.transform(data_train[cols_with_X])
            data_train[cols_with_X] = data_train[cols_with_X].round()

        else:
            # for i in ["smoking_status", "clinical_category", "regional_nodes_category", "metastasis_category"]: #df_clinical.columns[5:]:
            for i in df_clinical.columns[4:]:
                if impute_percent != 30:
                    data_train[i] = data_train[i].sample(frac=impute_percent / 100)
            data_train = data_train.fillna("X")

        if cfg.dataset_name == "lung":
            if "MALE" in data_train["gender"].unique():
                data_train.loc[len(data_train.index)] = ["MALE", "X", "X", "X", "X"]
            else:
                data_train.loc[len(data_train.index)] = [0, "X", "X", "X", "X"]

            print(data_train)

            drop_one_col_from_each_cat = False  # if False, [male, female] stays as [male, female], not just [male] or [female] with one col dropped
            penalize_missing_data = False  # ignores missing data if False

            if drop_one_col_from_each_cat:
                data_train = pd.get_dummies(
                    data_train, columns=["gender"], drop_first=True
                )
                # do NOT drop_first in columns with nan/unknown values/X
                cols_with_X = [
                    "smoking_status",
                    "clinical_category",
                    "regional_nodes_category",
                    "metastasis_category",
                ]
                data_train = pd.get_dummies(
                    data_train, columns=cols_with_X, drop_first=False
                )
                # drop X cols only
                data_train = data_train.drop(
                    columns=[i for i in data_train.columns if "_X" in i]
                )
            else:
                data_train = pd.get_dummies(
                    data_train, columns=["gender"], drop_first=False
                )
                # do NOT drop_first in columns with nan/unknown values/X
                cols_with_X = [
                    "smoking_status",
                    "clinical_category",
                    "regional_nodes_category",
                    "metastasis_category",
                ]
                data_train = pd.get_dummies(
                    data_train, columns=cols_with_X, drop_first=False
                )

            if not penalize_missing_data:
                missing_val_placeholder = 99
                cols_with_X = [i for i in data_train.columns if "_X" in i]

                for i in cols_with_X:
                    data_train[i][
                        data_train[i] == 1
                    ] = missing_val_placeholder  # placeholder for missing data

            num_classes = len(data_train.columns)
            if not penalize_missing_data:
                num_classes = len(data_train.columns) - len(
                    [i for i in data_train.columns if "_X" in i]
                )
            print("penalize_missing_data", penalize_missing_data)
            print("num_classes", num_classes)

        label_train = []
        time_train = []
        event_train = []
        for i in ct_train:
            label_train.append(data_train.loc[i].tolist())
            time_train.append(data_time_train.loc[i])
            event_train.append(data_event_train.loc[i])

        label_val = []
        time_val = []
        event_val = []
        for i in ct_val:
            label_val.append(data_train.loc[i].tolist())
            time_val.append(data_time_train.loc[i])
            event_val.append(data_event_train.loc[i])

        # data_train, data_test = train_test_split(data, test_size=.25, random_state=42)
        time_bins = make_time_bins(
            data_time_train.values, event=data_event_train.values
        )  # note: time bins are created based on full dataset (not just training!)
        num_time_bins = len(time_bins)

        train_files = [
            {
                "image": os.path.basename(ct_in).replace(".nii.gz", ""),
                "path_to_h5": PATH_TO_H5,
                "label": gtvt_in,
                "time": time_in,
                "event": event_in,
            }
            for ct_in, gtvt_in, time_in, event_in in zip(
                ct_train, label_train, time_train, event_train
            )
        ]

        val_files = [
            {
                "image": os.path.basename(ct_in).replace(".nii.gz", ""),
                "path_to_h5": PATH_TO_H5,
                "label": gtvt_in,
                "time": time_in,
                "event": event_in,
            }
            for ct_in, gtvt_in, time_in, event_in in zip(
                ct_val, label_val, time_val, event_val
            )
        ]
    elif cfg.dataset_name == "hecktor":
        cols = [
            "Gender",
            "T-stage",
            "N-stage",
            "M-stage",
            "HPV_status",
            "Chemotherapy",
        ]
        data_train = data_train[cols]
        if impute == "mode":
            for i in cols:
                if impute_percent != 30:
                    data_train[i] = data_train[i].sample(frac=impute_percent / 100)

                data_train[i] = data_train[i].fillna(data_train[i].mode()[0])

        non_missing_binary_one_hot_cols = ["Gender", "M-stage", "Chemotherapy"]

        non_missing_multiclass_one_hot_cols = ["T-stage", "N-stage"]

        one_hot_cols = [
            "HPV_status",
        ]

        # drop columns
        # data_train = pd.get_dummies(df, columns=non_missing_binary_one_hot_cols, drop_first=True)
        data_train = pd.get_dummies(
            data_train, columns=non_missing_binary_one_hot_cols, drop_first=False
        )

        data_train = pd.get_dummies(data_train, columns=one_hot_cols, drop_first=False)
        data_train = pd.get_dummies(
            data_train, columns=non_missing_multiclass_one_hot_cols, drop_first=False
        )
        cols_with_X = [i for i in data_train.columns if "X" in i]
        data_train[cols_with_X] = data_train[cols_with_X].replace({1: 99})

        drop_one_col_from_each_cat = False  # if False, [male, female] stays as [male, female], not just [male] or [female] with one col dropped
        penalize_missing_data = False  # ignores missing data if False

        data_train = data_train[
            [  # gt, output
                "Gender_F",
                "Gender_M",  # [0:2]; [0:2]
                "Chemotherapy_0",
                "Chemotherapy_1",  # [2:4]; [2:4]
                "HPV_status_0.0",
                "HPV_status_1.0",
                "HPV_status_X",  # [13:16]; [13:15]
                "T-stage_1",
                "T-stage_2",
                "T-stage_3",
                "T-stage_4",  # [16:20]; [16:20]
                "N-stage_0",
                "N-stage_1",
                "N-stage_2",
                "N-stage_3",  # [20:24]; [20:24]
                "M-stage_0",
                "M-stage_1",
            ]
        ]
        if not penalize_missing_data:
            missing_val_placeholder = 99
            cols_with_X = [i for i in data_train.columns if "_X" in i]

            for i in cols_with_X:
                data_train[i][
                    data_train[i] == 1
                ] = missing_val_placeholder  # placeholder for missing data

        num_classes = len(data_train.columns)
        if not penalize_missing_data:
            num_classes = len(data_train.columns) - len(
                [i for i in data_train.columns if "_X" in i]
            )
        print("penalize_missing_data", penalize_missing_data)
        print("num_classes", num_classes)
        label_train = []
        time_train = []
        event_train = []
        for i in train_id:
            label_train.append(data_train.loc[i].tolist())
            time_train.append(data_time_train.loc[i])
            event_train.append(data_event_train.loc[i])

        label_val = []
        time_val = []
        event_val = []
        for i in val_id:
            label_val.append(data_train.loc[i].tolist())
            time_val.append(data_time_train.loc[i])
            event_val.append(data_event_train.loc[i])

        # data_train, data_test = train_test_split(data, test_size=.25, random_state=42)
        time_bins = make_time_bins(
            data_time_train.values, event=data_event_train.values
        )  # note: time bins are created based on full dataset (not just training!)
        num_time_bins = len(time_bins)

        train_files = [
            {
                "ct": ct_in,
                "pt": ct_in.replace("ct", "pt"),
                "label": gtvt_in,
                "time": time_in,
                "event": event_in,
            }
            for ct_in, gtvt_in, time_in, event_in in zip(
                ct_train, label_train, time_train, event_train
            )
        ]

        val_files = [
            {
                "ct": ct_in,
                "pt": ct_in.replace("ct", "pt"),
                "label": gtvt_in,
                "time": time_in,
                "event": event_in,
            }
            for ct_in, gtvt_in, time_in, event_in in zip(
                ct_val, label_val, time_val, event_val
            )
        ]

    if vSubmit:
        train_files += val_files
        val_files = train_files

    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    print(f"Length of training data: {len(train_ds)}")
    print(f'Train image shape {train_ds[0]["image"].shape}')

    # create a validation & testing data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    print(f"Length of validation data: {len(val_ds)}")
    print(f'Validation Image shape {val_ds[0]["image"].shape}')

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=pin_memory,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=pin_memory,
        persistent_workers=True,
    )

    if cfg.dataset_name == "lung":
        in_channels = 1
    elif cfg.dataset_name == "hecktor":
        in_channels = 2

    if cfg.model == "PrognosisModel":
        model = PrognosisModel(
            spatial_dims=3,
            in_channels=in_channels,
            emb_size=emb_size,
            out_channels=len(data_train.columns),
            int_cem=intervention_cem,
            pre_prognosis=pre_prognosis,
            prognosis=prognosis,
            num_time_bins=num_time_bins,
            image_only=image_only,
            ordinal_loss=ordinal_loss,
            free_concept_pred=free_concept_pred,
            missing_val_placeholder=missing_val_placeholder,
            soft_prob=soft_prob,
            random_intervention_thresh=random_intervention_thresh,
            dropout=dropout,
            device=device,
        )
    elif cfg.model == "Dual_MTLR":
        model = Dual_MTLR(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=len(data_train.columns),
            num_time_bins=num_time_bins,
        )
    model.to(device)

    ce_loss_function = torch.nn.CrossEntropyLoss(reduction="mean")
    if ordinal_loss == "corn":
        # ordinal_loss_function = corn_loss
        pass

    y_trans_2 = Compose([AsDiscrete(to_onehot=2)])
    y_trans_3 = Compose([AsDiscrete(to_onehot=3)])
    y_trans_4 = Compose([AsDiscrete(to_onehot=4)])
    y_trans_5 = Compose([AsDiscrete(to_onehot=5)])

    y_pred_others_trans = Compose([Activations(softmax=True)])

    # start a typical PyTorch training
    val_interval = 1
    auc_metric = ROCAUCMetric()

    best_metric = -1
    best_metric_epoch = -1
    epoch_lossues = []
    metricues = {
        "auc": [],
        "sensitivity": [],
        "specificity": [],
        "balanced_accuracy": [],
        "weighted_score": [],
    }
    writer = SummaryWriter()
    max_epochs = 100
    # start a new wandb run to track this script
    run = wandb.init(
        # set the wandb project where this run will be logged
        project="IPEM_" + cfg.dataset_name,
        name=current_datetime,
        # track hyperparameters and run metadata
        config={
            # "PATH_TO_H5": PATH_TO_H5.split("/")[-1],
            "vSubmit": vSubmit,
            "seed": SEED,
            "fold": fold,
            "impute": impute,
            "impute_percent": impute_percent,
            "learning_rate": lr,
            # "architecture": "UNet",
            "epochs": max_epochs,
            "num_classes": num_classes,
            "penalize_missing_data": penalize_missing_data,
            "drop_one_col_from_each_cat": drop_one_col_from_each_cat,
            "batch_size": batch_size,
            "model": cfg.model,
            "pre_prognosis": pre_prognosis,
            "prognosis": prognosis,
            "mtlr_loss": mtlr_loss,
            "mtlr_bin_alpha": mtlr_bin_alpha,
            "mtlr_bin_penalize_censored": mtlr_bin_penalize_censored,
            "num_time_bins": num_time_bins,
            # "sigmoid": sig,
            "image_only": image_only,
            "ordinal_loss": ordinal_loss,
            "emb_size": emb_size,
            "dropout": dropout,
            "prog_loss_wt": prog_loss_wt,
            "num_bins": num_bins,
            "intervention_cem": intervention_cem,
            "random_intervention_thresh": random_intervention_thresh,
            "concept_act": concept_act,
            "xyz_dim": xyz_dim,
            "ClipCT_range": ClipCT_range,
            "ClipCT_window_center_width": ClipCT_window_center_width,
            "soft_prob": soft_prob,
            "num_warmup_epochs": number_warmup_epochs,
            "free_concept_pred": free_concept_pred,
            "aug_prob": prob,
            # "xy_size": imsize,
            # "z_size": zsize,
        },
        notes=root_dir,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    train_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=0, last_epoch=-1
    )

    def warmup(current_step: int):
        return 1 / (10 ** (float(number_warmup_epochs - current_step)))

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup)

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup_scheduler, train_scheduler], [number_warmup_epochs]
    )

    scaler = torch.cuda.amp.GradScaler()

    best_metric = -1
    best_metric_train = -1
    best_ci = -1
    best_ci_train = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    values = []
    metric_values = {}
    metric_values["f1"] = []
    metric_values["accuracy"] = []
    metric_values["precision"] = []
    metric_values["recall"] = []
    metric_values["classification_report"] = []
    metric_values["roc_auc"] = []
    y_labels = []
    y_preds = []

    for epoch in tqdm(range(max_epochs)):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")

        model.train()

        epoch_loss = 0
        step = 0

        y_pred_prog = torch.tensor([], dtype=torch.float32, device=device)  #
        y_time = torch.tensor([], dtype=torch.long, device=device)  # true label time
        y_event = torch.tensor([], dtype=torch.long, device=device)  # true label event
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["image"].to(device), torch.stack(
                batch_data["label"], axis=1
            ).float().to(device)
            times, events = batch_data["time"].to(device), batch_data["event"].to(
                device
            )
            # labels = rearrange(labels, 'c b -> b c')
            y_prog = encode_survival(
                times.cpu().numpy(), events.cpu().numpy(), time_bins
            ).to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                if dual_mtlr:
                    outputs, outputs_prog = model(inputs, gt_class=labels)
                elif intervention_cem:
                    outputs, outputs_prog = model(inputs, gt_class=labels, train=True)
                else:
                    outputs, outputs_prog = model(inputs, gt_class=None)

                if outputs_prog is not None:
                    y_pred_prog = torch.cat([y_pred_prog, outputs_prog], dim=0)
                    y_time = torch.cat([y_time, times], dim=0)
                    y_event = torch.cat([y_event, events], dim=0)

                    if mtlr_loss == "bin_loss":
                        prog_loss = mtlr_survival_bin_neg_log_likelihood(
                            y_prog,
                            outputs_prog.as_tensor(),
                            model,
                            C1=1.0,
                            average=True,
                            alpha=mtlr_bin_alpha,  # lower alpha gives more weight to bin loss
                            bin_penalty_type="abs",
                            penalize_censored=mtlr_bin_penalize_censored,
                        )
                    elif mtlr_loss == "original_mtlr":
                        # prog_loss = mtlr_neg_log_likelihood(
                        #     y_prog,
                        #     outputs_prog.as_tensor(),
                        #     model,
                        #     C1=1.0,
                        #     average=True,
                        # )
                        y_prog = deephit_encode_survival(
                            times.cpu().numpy(), events.cpu().numpy(), time_bins
                        ).to(device)
                        y_prog = y_prog.argmax(dim=1)
                        prog_loss = deepmtlr_loss(
                            outputs_prog.as_tensor(), y_prog, events
                        )
                    elif mtlr_loss == "prog_corn_rank":
                        prog_loss = prognosis_ranking_loss(
                            outputs_prog, times, events, num_time_bins, device=device
                        )
                    elif mtlr_loss == "deephit":
                        try:
                            y_prog = deephit_encode_survival(
                                times.cpu().numpy(), events.cpu().numpy(), time_bins
                            ).to(device)
                            y_prog = y_prog.argmax(dim=1)
                            prog_loss = deephit_loss(
                                outputs_prog.as_tensor(), y_prog, events
                            )  # deephit_loss(scores, labels, censors)
                        except Exception as e:
                            print("Train deephit loss failed")
                            print("times", times.shape)
                            print("events", events.shape)
                            print("time_bins", time_bins.shape)
                            breakpoint()

                loss = 0
                if (
                    outputs is not None
                ):  # 15 classes outputs; 19 classes labels (includes _X/unknowns)

                    # if not penalize_missing_data:
                    if cfg.dataset_name == "lung":
                        missing_val_placeholder = torch.tensor(missing_val_placeholder)

                        missing_val_idx = (
                            labels == missing_val_placeholder
                        )  # .nonzero().flatten()

                        missing_gender_idx = missing_val_idx[:, 0:2]  # gen1, gen2
                        missing_smoking_idx = missing_val_idx[
                            :, 2:6
                        ]  # smk1, smk2, smk3, smkX
                        missing_t_stage_idx = missing_val_idx[
                            :, 6:11
                        ]  # t1, t2, t3, t4, tX
                        missing_n_stage_idx = missing_val_idx[
                            :, 11:16
                        ]  # n0, n1, n2, n3, nX
                        missing_m_stage_idx = missing_val_idx[:, 16:]  # m0, m1, mX

                        # Check for True values along axis 1 (columns)
                        missing_gender_idx = torch.any(missing_gender_idx, dim=1)
                        missing_smoking_idx = torch.any(missing_smoking_idx, dim=1)
                        missing_t_stage_idx = torch.any(missing_t_stage_idx, dim=1)
                        missing_n_stage_idx = torch.any(missing_n_stage_idx, dim=1)
                        missing_m_stage_idx = torch.any(missing_m_stage_idx, dim=1)

                        # 15 classes outputs;
                        outputs_gender = outputs[:, 0:2].as_tensor()[
                            ~missing_gender_idx
                        ]  # gen1, gen2
                        outputs_smoking = outputs[:, 2:5].as_tensor()[
                            ~missing_smoking_idx
                        ]  # smk1, smk2, smk3
                        outputs_t_stage = outputs[:, 5:9].as_tensor()[
                            ~missing_t_stage_idx
                        ]  # t1, t2, t3, t4
                        outputs_n_stage = outputs[:, 9:13].as_tensor()[
                            ~missing_n_stage_idx
                        ]  # n0, n1, n2, n3
                        outputs_m_stage = outputs[:, 13:15].as_tensor()[
                            ~missing_m_stage_idx
                        ]  # m0, m1

                        # 19 classes labels (includes _X/unknowns); these idx is after removing _X cols
                        labels_gender = labels[:, 0:2][
                            ~missing_gender_idx
                        ]  # gen1, gen2
                        labels_smoking = labels[:, 2:5][
                            ~missing_smoking_idx
                        ]  # smk1, smk2, smk3 (remove smkX)
                        labels_t_stage = labels[:, 6:10][
                            ~missing_t_stage_idx
                        ]  # t1, t2, t3, t4 (remove tX)
                        labels_n_stage = labels[:, 11:15][
                            ~missing_n_stage_idx
                        ]  # n0, n1, n2, n3 (remove nX)
                        labels_m_stage = labels[:, 16:18][
                            ~missing_m_stage_idx
                        ]  # m0, m1 (remove mX)

                        # binary classification
                        gender_loss = ce_loss_function(outputs_gender, labels_gender)
                        # smoking_loss = ce_loss_function(outputs_smoking, labels_smoking)
                        m_stage_loss = ce_loss_function(outputs_m_stage, labels_m_stage)

                        if ordinal_loss == "corn":
                            pass

                        else:
                            smoking_loss = ce_loss_function(
                                outputs_smoking, labels_smoking
                            )
                            t_stage_loss = ce_loss_function(
                                outputs_t_stage, labels_t_stage
                            )
                            n_stage_loss = ce_loss_function(
                                outputs_n_stage, labels_n_stage
                            )

                        # else:
                        #     gender_loss = ce_loss_function(outputs[:, 0:2], labels[:, 0:2])
                        #     smoking_loss = ce_loss_function(outputs[:, 2:5], labels[:, 2:5])
                        #     t_stage_loss = ce_loss_function(
                        #         outputs[:, 5:9], labels[:, 6:10]
                        #     )
                        #     n_stage_loss = ce_loss_function(
                        #         outputs[:, 9:13], labels[:, 11:15]
                        #     )
                        #     m_stage_loss = ce_loss_function(
                        #         outputs[:, 13:15], labels[:, 16:18]
                        #     )

                        loss += (
                            gender_loss
                            + smoking_loss
                            + t_stage_loss
                            + n_stage_loss
                            + m_stage_loss
                        )

                    elif cfg.dataset_name == "hecktor":
                        missing_val_placeholder = torch.tensor(missing_val_placeholder)

                        missing_val_idx = (
                            labels == missing_val_placeholder
                        )  # .nonzero().flatten()

                        missing_gender_idx = missing_val_idx[:, 0:2]  # gen1, gen2
                        missing_chemotherapy_idx = missing_val_idx[:, 2:4]
                        missing_hpv_idx = missing_val_idx[:, 4:7]
                        missing_t_stage = missing_val_idx[:, 7:11]
                        missing_n_stage = missing_val_idx[:, 11:15]
                        missing_m_stage = missing_val_idx[:, 15:17]

                        # Check for True values along axis 1 (columns)
                        missing_gender_idx = torch.any(missing_gender_idx, dim=1)
                        missing_chemotherapy_idx = torch.any(
                            missing_chemotherapy_idx, dim=1
                        )
                        missing_hpv_idx = torch.any(missing_hpv_idx, dim=1)
                        missing_t_stage = torch.any(missing_t_stage, dim=1)
                        missing_n_stage = torch.any(missing_n_stage, dim=1)
                        missing_m_stage = torch.any(missing_m_stage, dim=1)

                        outputs_gender = outputs[:, 0:2].as_tensor()[
                            ~missing_gender_idx
                        ]  # gen1, gen2
                        outputs_chemotherapy = outputs[:, 2:4].as_tensor()[
                            ~missing_chemotherapy_idx
                        ]
                        outputs_hpv = outputs[:, 4:6].as_tensor()[~missing_hpv_idx]
                        outputs_t_stage = outputs[:, 6:10].as_tensor()[~missing_t_stage]
                        outputs_n_stage = outputs[:, 10:14].as_tensor()[
                            ~missing_n_stage
                        ]
                        outputs_m_stage = outputs[:, 14:16].as_tensor()[
                            ~missing_m_stage
                        ]

                        labels_gender = labels[:, 0:2][
                            ~missing_gender_idx
                        ]  # gen1, gen2
                        labels_chemotherapy = labels[:, 2:4][~missing_chemotherapy_idx]
                        labels_hpv = labels[:, 4:6][~missing_hpv_idx]
                        labels_t_stage = labels[:, 7:11][~missing_t_stage]
                        labels_n_stage = labels[:, 11:15][~missing_n_stage]
                        labels_m_stage = labels[:, 15:17][~missing_m_stage]

                        gender_loss = ce_loss_function(outputs_gender, labels_gender)
                        chemotherapy_loss = ce_loss_function(
                            outputs_chemotherapy, labels_chemotherapy
                        )
                        hpv_loss = ce_loss_function(outputs_hpv, labels_hpv)
                        t_stage_loss = ce_loss_function(outputs_t_stage, labels_t_stage)
                        n_stage_loss = ce_loss_function(outputs_n_stage, labels_n_stage)
                        m_stage_loss = ce_loss_function(outputs_m_stage, labels_m_stage)

                        loss += (
                            gender_loss
                            + chemotherapy_loss
                            + hpv_loss
                            + t_stage_loss
                            + n_stage_loss
                            + m_stage_loss
                        )

                if outputs_prog is not None:
                    loss += prog_loss_wt * prog_loss

                # loss.backward()
                # optimizer.step()
                # scheduler.step()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scale = scaler.get_scale()
                scaler.update()
                skip_lr_sched = scale != scaler.get_scale()
                if not skip_lr_sched:
                    scheduler.step()
                epoch_len = len(train_ds) // train_loader.batch_size
                if loss.item() == loss.item():
                    epoch_loss += loss.item()
                    stepp = step
                else:
                    # epoch_len -= 1
                    stepp = step - 1
                print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
                writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)

        if outputs_prog is not None:
            if "deephit" in mtlr_loss or "mtlr" in mtlr_loss:
                surv = predict_surv_df(
                    y_pred_prog.detach(), time_bins.cpu().numpy(), loss_type=mtlr_loss
                )
                ev = EvalSurv(
                    surv,
                    y_time.detach().cpu().numpy(),
                    y_event.detach().cpu().numpy(),
                    censor_surv="km",
                )
                train_ci = ev.concordance_td("antolini")

            elif "mtlr" in mtlr_loss:
                surv = predict_surv_df(
                    y_pred_prog.detach(), time_bins.cpu().numpy(), loss_type="mtlr"
                )
                ev = EvalSurv(
                    surv,
                    y_time.detach().cpu().numpy(),
                    y_event.detach().cpu().numpy(),
                    censor_surv="km",
                )
                train_ci = ev.concordance_td("antolini")

            # else:
            #     tr_pred_survival = mtlr_survival(y_pred_prog.detach()).cpu().numpy()
            #     tr_pred_risk = mtlr_risk(y_pred_prog.detach()).cpu().numpy()
            #     train_ci = concordance_index(
            #         y_time.detach().cpu().numpy(),
            #         -tr_pred_risk,
            #         event_observed=y_event.detach().cpu().numpy(),
            #     )

            wandb.log(
                {
                    "trn_concordance_idx": train_ci,
                },
                step=epoch + 1,
            )

        # epoch_loss /= step
        epoch_loss /= stepp
        epoch_lossues.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                y_pred = torch.tensor(
                    [], dtype=torch.float32, device=device
                )  # classification pred
                y_pred_prog = torch.tensor([], dtype=torch.float32, device=device)  #
                y = torch.tensor(
                    [], dtype=torch.long, device=device
                )  # true label classification
                y_time = torch.tensor(
                    [], dtype=torch.long, device=device
                )  # true label time
                y_event = torch.tensor(
                    [], dtype=torch.long, device=device
                )  # true label event
                for val_data in val_loader:
                    val_images, val_labels = (
                        val_data["image"].to(device),
                        torch.stack(val_data["label"], axis=1).float().to(device),
                    )
                    val_times, val_events = (
                        val_data["time"].to(device),
                        val_data["event"].to(device),
                    )
                    y_prog = encode_survival(
                        val_times.cpu().numpy(), val_events.cpu().numpy(), time_bins
                    ).to(device)

                    if dual_mtlr:
                        val_outputs, val_outputs_prog = model(
                            val_images, gt_class=val_labels
                        )
                    elif intervention_cem or impute != "None":
                        val_outputs, val_outputs_prog = model(
                            val_images, gt_class=val_labels, train=False
                        )
                    else:
                        val_outputs, val_outputs_prog = model(val_images, gt_class=None)

                    y_pred_prog = torch.cat([y_pred_prog, val_outputs_prog], dim=0)
                    y_time = torch.cat([y_time, val_times], dim=0)
                    y_event = torch.cat([y_event, val_events], dim=0)

                    if val_outputs is not None:
                        y_pred = torch.cat([y_pred, val_outputs], dim=0)
                    y = torch.cat([y, val_labels], dim=0)

                val_pred_survival = mtlr_survival(y_pred_prog).cpu().numpy()
                val_pred_risk = mtlr_risk(y_pred_prog).cpu().numpy()

                if "deephit" in mtlr_loss or "mtlr" in mtlr_loss:
                    val_pred_survival_at_times = mtlr_survival_at_times(
                        y_pred_prog, time_bins[:-1], eval_times
                    )
                elif "prog_corn" not in mtlr_loss:
                    val_pred_survival_at_times = mtlr_survival_at_times(
                        y_pred_prog, time_bins, eval_times
                    )

                if "prog_corn" not in mtlr_loss:
                    val_bs = brier_score_at_times(
                        y_time.cpu().numpy(),
                        val_pred_survival_at_times,
                        y_event.cpu().numpy(),
                        eval_times,
                    )
                    val_auc = roc_auc_at_times(
                        y_time.cpu().numpy(),
                        val_pred_survival_at_times,
                        y_event.cpu().numpy(),
                        eval_times,
                    )

                if "deephit" in mtlr_loss or "mtlr" in mtlr_loss:
                    try:
                        y_prog = deephit_encode_survival(
                            val_times.cpu().numpy(), val_events.cpu().numpy(), time_bins
                        ).to(device)
                    except:
                        print("Val deephit loss failed")
                        print("times", val_times.shape)
                        print("events", val_events.shape)
                        print("time_bins", time_bins.shape)
                        breakpoint()

                    surv = predict_surv_df(
                        y_pred_prog.detach(),
                        time_bins.cpu().numpy(),
                        loss_type=mtlr_loss,
                    )
                    ev = EvalSurv(
                        surv,
                        y_time.detach().cpu().numpy(),
                        y_event.detach().cpu().numpy(),
                        censor_surv="km",
                    )
                    val_ci = ev.concordance_td("antolini")

                else:
                    val_ci = concordance_index(
                        y_time.cpu().numpy(),
                        -val_pred_risk,
                        event_observed=y_event.cpu().numpy(),
                    )

                if val_outputs is not None:
                    # if not penalize_missing_data:
                    if cfg.dataset_name == "lung":
                        missing_val_idx = (
                            y == missing_val_placeholder
                        )  # .nonzero().flatten()

                        # temporary - remove all rows with X in prediction...
                        missing_val_idx = torch.any(missing_val_idx, dim=1)
                        y_pred = y_pred.as_tensor()[~missing_val_idx]
                        y = y[~missing_val_idx]

                        labels_gender = y[:, 0:2]
                        labels_smoking = y[:, 2:5]
                        labels_t_stage = y[:, 6:10]
                        labels_n_stage = y[:, 11:15]
                        labels_m_stage = y[:, 16:18]

                        if ordinal_loss == "corn":
                            pass

                        else:
                            # THE CORRECT ONE SHOULD BE THIS
                            (
                                y_pred_gender,
                                y_pred_smoking,
                                y_pred_t_stage,
                                y_pred_n_stage,
                                y_pred_m_stage,
                            ) = (
                                y_pred[:, 0:2],
                                y_pred[:, 2:5],
                                y_pred[:, 5:9],
                                y_pred[:, 9:13],
                                y_pred[:, 13:15],
                            )

                            y_pred_gender = [
                                y_trans_2(y_pred_others_trans(item).argmax())
                                for item in y_pred_gender
                            ]
                            y_pred_gender = torch.stack(y_pred_gender, dim=0)

                            y_pred_smoking = [
                                y_trans_3(y_pred_others_trans(item).argmax())
                                for item in y_pred_smoking
                            ]
                            y_pred_smoking = torch.stack(y_pred_smoking, dim=0)

                            y_pred_t_stage = [
                                y_trans_4(y_pred_others_trans(item).argmax())
                                for item in y_pred_t_stage
                            ]
                            y_pred_t_stage = torch.stack(y_pred_t_stage, dim=0)

                            y_pred_n_stage = [
                                y_trans_4(y_pred_others_trans(item).argmax())
                                for item in y_pred_n_stage
                            ]
                            y_pred_n_stage = torch.stack(y_pred_n_stage, dim=0)

                            y_pred_m_stage = [
                                y_trans_2(y_pred_others_trans(item).argmax())
                                for item in y_pred_m_stage
                            ]
                            y_pred_m_stage = torch.stack(y_pred_m_stage, dim=0)

                        y = torch.cat(
                            (
                                labels_gender,
                                labels_smoking,
                                labels_t_stage,
                                labels_n_stage,
                                labels_m_stage,
                            ),
                            dim=1,
                        )

                        y_pred = torch.cat(
                            (
                                y_pred_gender,
                                y_pred_smoking,
                                y_pred_t_stage,
                                y_pred_n_stage,
                                y_pred_m_stage,
                            ),
                            dim=1,
                        )

                    elif cfg.dataset_name == "hecktor":
                        missing_val_idx = (
                            y == missing_val_placeholder
                        )  # .nonzero().flatten()

                        # temporary - remove all rows with X in prediction...
                        missing_val_idx = torch.any(missing_val_idx, dim=1)
                        y_pred = y_pred.as_tensor()[~missing_val_idx]
                        y = y[~missing_val_idx]

                        labels_gender = y[:, 0:2]
                        labels_chemotherapy = y[:, 2:4]
                        labels_hpv = y[:, 4:6]
                        labels_t_stage = y[:, 7:11]
                        labels_n_stage = y[:, 11:15]
                        labels_m_stage = y[:, 15:17]

                        if ordinal_loss == "corn":
                            pass

                        else:
                            # THE CORRECT ONE SHOULD BE THIS
                            (
                                y_pred_gender,
                                y_pred_chemotherapy,
                                y_pred_hpv,
                                y_pred_t_stage,
                                y_pred_n_stage,
                                y_pred_m_stage,
                            ) = (
                                y_pred[:, 0:2],
                                y_pred[:, 2:4],
                                y_pred[:, 4:6],
                                y_pred[:, 6:10],
                                y_pred[:, 10:14],
                                y_pred[:, 14:16],
                            )

                            y_pred_gender = [
                                y_trans_2(y_pred_others_trans(item).argmax())
                                for item in y_pred_gender
                            ]
                            y_pred_gender = torch.stack(y_pred_gender, dim=0)

                            y_pred_chemotherapy = [
                                y_trans_2(y_pred_others_trans(item).argmax())
                                for item in y_pred_chemotherapy
                            ]
                            y_pred_chemotherapy = torch.stack(
                                y_pred_chemotherapy, dim=0
                            )

                            y_pred_hpv = [
                                y_trans_2(y_pred_others_trans(item).argmax())
                                for item in y_pred_hpv
                            ]
                            y_pred_hpv = torch.stack(y_pred_hpv, dim=0)

                            y_pred_t_stage = [
                                y_trans_4(y_pred_others_trans(item).argmax())
                                for item in y_pred_t_stage
                            ]
                            y_pred_t_stage = torch.stack(y_pred_t_stage, dim=0)

                            y_pred_n_stage = [
                                y_trans_4(y_pred_others_trans(item).argmax())
                                for item in y_pred_n_stage
                            ]
                            y_pred_n_stage = torch.stack(y_pred_n_stage, dim=0)

                            y_pred_m_stage = [
                                y_trans_2(y_pred_others_trans(item).argmax())
                                for item in y_pred_m_stage
                            ]
                            y_pred_m_stage = torch.stack(y_pred_m_stage, dim=0)

                        y = torch.cat(
                            (
                                labels_gender,
                                labels_chemotherapy,
                                labels_hpv,
                                labels_t_stage,
                                labels_n_stage,
                                labels_m_stage,
                            ),
                            dim=1,
                        )

                        y_pred = torch.cat(
                            (
                                y_pred_gender,
                                y_pred_chemotherapy,
                                y_pred_hpv,
                                y_pred_t_stage,
                                y_pred_n_stage,
                                y_pred_m_stage,
                            ),
                            dim=1,
                        )

                    y_labels.append(y)
                    y_preds.append(y_pred)

                    ## DON'T DO [1:-1] bc m-stage has missing data col (so sum class 0+1 is NOT 1)...
                    try:
                        _f1 = f1_score(
                            y.cpu().numpy()[:, 1:],
                            y_pred.cpu().numpy()[:, 1:],
                            average="macro",
                        )
                        _accuracy = accuracy_score(
                            y.cpu().numpy()[:, 1:], y_pred.cpu().numpy()[:, 1:]
                        )
                        _precision = precision_score(
                            y.cpu().numpy()[:, 1:],
                            y_pred.cpu().numpy()[:, 1:],
                            average="macro",
                        )
                        _recall = recall_score(
                            y.cpu().numpy()[:, 1:],
                            y_pred.cpu().numpy()[:, 1:],
                            average="macro",
                        )
                        _roc_auc = roc_auc_score(
                            y.cpu().numpy()[:, 1:],
                            y_pred.cpu().numpy()[:, 1:],
                            average="macro",
                        )
                    except:
                        _f1 = -1
                        _accuracy = -1
                        _precision = -1
                        _recall = -1
                        _roc_auc = -1

                    metric_values["f1"].append(_f1)
                    metric_values["accuracy"].append(_accuracy)
                    metric_values["precision"].append(_precision)
                    metric_values["recall"].append(_recall)
                    metric_values["roc_auc"].append(_roc_auc)

                    y_decollated = decollate_batch(y, detach=False)
                    y_pred_decollated = decollate_batch(y_pred, detach=False)

                    try:
                        auc_metric(y_pred_decollated, y_decollated)
                        result = auc_metric.aggregate()
                        auc_metric.reset()
                    except:
                        result = -1

                    if cfg.dataset_name == "lung":
                        del (
                            y_pred_gender,
                            y_pred_smoking,
                            y_pred_t_stage,
                            y_pred_n_stage,
                            y_pred_m_stage,
                            val_outputs,
                            val_outputs_prog,
                            val_data,
                            val_images,
                            val_labels,
                            val_times,
                            val_events,
                            y_prog,
                            val_pred_survival,
                            val_pred_risk,
                            val_pred_survival_at_times,
                        )
                    elif cfg.dataset_name == "hecktor":
                        del (
                            y_pred_gender,
                            y_pred_chemotherapy,
                            y_pred_hpv,
                            y_pred_t_stage,
                            y_pred_n_stage,
                            y_pred_m_stage,
                            val_outputs,
                            val_outputs_prog,
                            val_data,
                            val_images,
                            val_labels,
                            val_times,
                            val_events,
                            y_prog,
                            val_pred_survival,
                            val_pred_risk,
                            val_pred_survival_at_times,
                        )

                    values.append(result)

                    if val_ci > best_metric:
                        best_avg_auc = result
                        best_metric = val_ci
                        best_metric_epoch = epoch + 1
                        try:
                            best_auc = roc_auc_score(
                                y.cpu().numpy()[:, 1:],
                                y_pred.cpu().numpy()[:, 1:],
                                average="macro",
                            )
                        except:
                            best_auc = -1
                        torch.save(
                            model.state_dict(),
                            os.path.join(
                                root_dir, f"best_metric_dense121_CEM_lr{lr}.pth"
                            ),
                        )
                        # artifact = wandb.Artifact("model_1", type="model")
                        # artifact.add_file(
                        #     os.path.join(
                        #         root_dir, f"best_metric_dense121_CEM_lr{lr}.pth"
                        #     )
                        # )
                        # run.log_artifact(artifact)
                        torch.save(
                            optimizer.state_dict(),
                            os.path.join(root_dir, f"best_opt_dense121_lr{lr}.pth"),
                        )
                        # artifact = wandb.Artifact("model_2", type="model")
                        # artifact.add_file(
                        #     os.path.join(root_dir, f"best_opt_dense121_lr{lr}.pth")
                        # )
                        # run.log_artifact(artifact)
                        torch.save(
                            scheduler.state_dict(),
                            os.path.join(root_dir, f"best_sch_dense121_lr{lr}.pth"),
                        )
                        # artifact = wandb.Artifact("model_3", type="model")
                        # artifact.add_file(
                        #     os.path.join(root_dir, f"best_sch_dense121_lr{lr}.pth")
                        # )
                        # run.log_artifact(artifact)
                        print("saved new best metric model")
                    print(
                        f"current epoch: {epoch + 1} current AUC: {result:.4f} current CI: {val_ci:.4f} train CI: {train_ci:.4f}"
                        # f" current accuracy: {acc_metric:.4f}"
                        f" best CI: {best_metric:.4f}"
                        f" at epoch: {best_metric_epoch}"
                    )
                    try:
                        print(
                            roc_auc_score(
                                y.cpu().numpy()[:, 1:],
                                y_pred.cpu().numpy()[:, 1:],
                                average=None,
                            )
                        )
                    except:
                        pass

                    if cfg.dataset_name == "lung":
                        wandb.log(
                            {
                                "f1": _f1,
                                "accuracy": _accuracy,
                                "precision": _precision,
                                "recall": _recall,
                                "roc_auc": _roc_auc,
                                "best_auc": best_auc,
                                "gender_loss": gender_loss,
                                "smoking_loss": smoking_loss,
                                "t_stage_loss": t_stage_loss,
                                "n_stage_loss": n_stage_loss,
                                "m_stage_loss": m_stage_loss,
                                "prog_loss": prog_loss,
                                "train_loss": loss,
                                "val_concordance_idx": val_ci,
                                **{
                                    f"val_bs_{t}": val_bs[i]
                                    for i, t in enumerate(eval_times)
                                },
                                **{
                                    f"val_auc_{t}": val_auc[i]
                                    for i, t in enumerate(eval_times)
                                },
                            },
                            step=epoch + 1,
                        )
                    elif cfg.dataset_name == "hecktor":
                        wandb.log(
                            {
                                "f1": _f1,
                                "accuracy": _accuracy,
                                "precision": _precision,
                                "recall": _recall,
                                "roc_auc": _roc_auc,
                                "best_auc": best_auc,
                                "gender_loss": gender_loss,
                                "chemotherapy_loss": chemotherapy_loss,
                                "hpv_loss": hpv_loss,
                                "t_stage_loss": t_stage_loss,
                                "n_stage_loss": n_stage_loss,
                                "m_stage_loss": m_stage_loss,
                                "prog_loss": prog_loss,
                                "train_loss": loss,
                                "val_concordance_idx": val_ci,
                                **{
                                    f"val_bs_{t}": val_bs[i]
                                    for i, t in enumerate(eval_times)
                                },
                                **{
                                    f"val_auc_{t}": val_auc[i]
                                    for i, t in enumerate(eval_times)
                                },
                            },
                            step=epoch + 1,
                        )

                    torch.save(
                        model.state_dict(),
                        os.path.join(root_dir, f"last_metric_dense121_CEM_lr{lr}.pth"),
                    )
                    # artifact = wandb.Artifact("model_4", type="model")
                    # artifact.add_file(
                    #     os.path.join(root_dir, f"last_metric_dense121_CEM_lr{lr}.pth")
                    # )
                    # run.log_artifact(artifact)
                    torch.save(
                        optimizer.state_dict(),
                        os.path.join(root_dir, f"last_opt_dense121_CEM_lr{lr}.pth"),
                    )
                    # artifact = wandb.Artifact("model_5", type="model")
                    # artifact.add_file(
                    #     os.path.join(root_dir, f"last_opt_dense121_CEM_lr{lr}.pth")
                    # )
                    # run.log_artifact(artifact)
                    torch.save(
                        scheduler.state_dict(),
                        os.path.join(root_dir, f"last_sch_dense121_CEM_lr{lr}.pth"),
                    )
                    # artifact = wandb.Artifact("model_6", type="model")
                    # artifact.add_file(
                    #     os.path.join(root_dir, f"last_sch_dense121_CEM_lr{lr}.pth")
                    # )
                    # run.log_artifact(artifact)

                    ######### FOR FULL SUBMISSION ###############
                    if train_ci > best_metric_train:
                        best_metric_train = train_ci
                        try:
                            best_auc_train = roc_auc_score(
                                y.cpu().numpy()[:, 1:],
                                y_pred.cpu().numpy()[:, 1:],
                                average="macro",
                            )
                        except:
                            best_auc_train = -1
                        torch.save(
                            model.state_dict(),
                            os.path.join(
                                root_dir, f"best_train_metric_dense121_CEM_lr{lr}.pth"
                            ),
                        )
                        # artifact = wandb.Artifact("model_7", type="model")
                        # artifact.add_file(
                        #     os.path.join(
                        #         root_dir, f"best_train_metric_dense121_CEM_lr{lr}.pth"
                        #     )
                        # )
                        # run.log_artifact(artifact)
                        torch.save(
                            optimizer.state_dict(),
                            os.path.join(
                                root_dir, f"best_train_opt_dense121_lr{lr}.pth"
                            ),
                        )
                        # artifact = wandb.Artifact("model_8", type="model")
                        # artifact.add_file(
                        #     os.path.join(
                        #         root_dir, f"best_train_opt_dense121_lr{lr}.pth"
                        #     ),
                        # )
                        # run.log_artifact(artifact)
                        torch.save(
                            scheduler.state_dict(),
                            os.path.join(
                                root_dir, f"best_train_sch_dense121_lr{lr}.pth"
                            ),
                        )
                        # artifact = wandb.Artifact("model_9", type="model")
                        # artifact.add_file(
                        #     os.path.join(
                        #         root_dir, f"best_train_sch_dense121_lr{lr}.pth"
                        #     ),
                        # )
                        # run.log_artifact(artifact)

                    ######### FOR FULL SUBMISSION ###############

                else:
                    best_auc = None
                    if val_ci > best_metric:
                        best_metric = val_ci
                        best_metric_epoch = epoch + 1
                        torch.save(
                            model.state_dict(),
                            os.path.join(
                                root_dir, f"best_metric_dense121Ori_lr{lr}.pth"
                            ),
                        )
                        # artifact = wandb.Artifact("model_10", type="model")
                        # artifact.add_file(
                        #     os.path.join(
                        #         root_dir, f"best_metric_dense121Ori_lr{lr}.pth"
                        #     ),
                        # )
                        # run.log_artifact(artifact)
                        torch.save(
                            optimizer.state_dict(),
                            os.path.join(root_dir, f"best_opt_dense121_lr{lr}.pth"),
                        )
                        # artifact = wandb.Artifact("model_11", type="model")
                        # artifact.add_file(
                        #     os.path.join(root_dir, f"best_opt_dense121_lr{lr}.pth"),
                        # )
                        # run.log_artifact(artifact)
                        torch.save(
                            scheduler.state_dict(),
                            os.path.join(root_dir, f"best_sch_dense121_lr{lr}.pth"),
                        )
                        # artifact = wandb.Artifact("model_12", type="model")
                        # artifact.add_file(
                        #     os.path.join(root_dir, f"best_sch_dense121_lr{lr}.pth"),
                        # )
                        # run.log_artifact(artifact)
                        print("saved new best metric model")
                    print(
                        f"current epoch: {epoch + 1} current CI: {val_ci:.4f} train CI: {train_ci:.4f}"
                        # f" current accuracy: {acc_metric:.4f}"
                        f" best CI: {best_metric:.4f}"
                        f" at epoch: {best_metric_epoch}"
                    )

                    wandb.log(
                        {
                            "prog_loss": prog_loss,
                            "train_loss": loss,
                            "val_concordance_idx": val_ci,
                            **{
                                f"val_bs_{t}": val_bs[i]
                                for i, t in enumerate(eval_times)
                            },
                            **{
                                f"val_auc_{t}": val_auc[i]
                                for i, t in enumerate(eval_times)
                            },
                        },
                        step=epoch + 1,
                    )

                    torch.save(
                        model.state_dict(),
                        os.path.join(root_dir, f"last_metric_dense121Ori_lr{lr}.pth"),
                    )
                    # artifact = wandb.Artifact("model_13", type="model")
                    # artifact.add_file(
                    #     os.path.join(root_dir, f"last_metric_dense121Ori_lr{lr}.pth"),
                    # )
                    # run.log_artifact(artifact)
                    torch.save(
                        optimizer.state_dict(),
                        os.path.join(root_dir, f"last_opt_dense121_CEM_lr{lr}.pth"),
                    )
                    # artifact = wandb.Artifact("model_14", type="model")
                    # artifact.add_file(
                    #     os.path.join(root_dir, f"last_opt_dense121_CEM_lr{lr}.pth"),
                    # )
                    # run.log_artifact(artifact)
                    torch.save(
                        scheduler.state_dict(),
                        os.path.join(root_dir, f"last_sch_dense121_CEM_lr{lr}.pth"),
                    )
                    # artifact = wandb.Artifact("model_15", type="model")
                    # artifact.add_file(
                    #     os.path.join(root_dir, f"last_sch_dense121_CEM_lr{lr}.pth"),
                    # )
                    # run.log_artifact(artifact)

                    ######### FOR FULL SUBMISSION ###############
                    if train_ci > best_metric_train:
                        best_metric_train = train_ci
                        best_train_epoch = epoch + 1
                        try:
                            best_auc_train = roc_auc_score(
                                y.cpu().numpy()[:, 1:],
                                y_pred.cpu().numpy()[:, 1:],
                                average="macro",
                            )
                        except:
                            best_auc_train = -1
                        torch.save(
                            model.state_dict(),
                            os.path.join(
                                root_dir, f"best_train_metric_dense121_CEM_lr{lr}.pth"
                            ),
                        )
                        # artifact = wandb.Artifact("model_16", type="model")
                        # artifact.add_file(
                        #     os.path.join(
                        #         root_dir, f"best_train_metric_dense121_CEM_lr{lr}.pth"
                        #     ),
                        # )
                        # run.log_artifact(artifact)
                        torch.save(
                            optimizer.state_dict(),
                            os.path.join(
                                root_dir, f"best_train_opt_dense121_lr{lr}.pth"
                            ),
                        )
                        # artifact = wandb.Artifact("model_17", type="model")
                        # artifact.add_file(
                        #     os.path.join(
                        #         root_dir, f"best_train_opt_dense121_lr{lr}.pth"
                        #     ),
                        # )
                        # run.log_artifact(artifact)
                        torch.save(
                            scheduler.state_dict(),
                            os.path.join(
                                root_dir, f"best_train_sch_dense121_lr{lr}.pth"
                            ),
                        )
                        # artifact = wandb.Artifact("model_18", type="model")
                        # artifact.add_file(
                        #     os.path.join(
                        #         root_dir, f"best_train_sch_dense121_lr{lr}.pth"
                        #     ),
                        # )
                        # run.log_artifact(artifact)

                    ######### FOR FULL SUBMISSION ###############

        del y_pred, y_pred_prog, y, y_time, y_event

    print(
        f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}"
    )

    try:
        print("best_auc_val", best_auc)

        ######### FOR FULL SUBMISSION ###############
        print("best_ci_train at epoch", best_train_epoch, best_auc_train)
        ######### FOR FULL SUBMISSION ###############
    except:
        best_auc_train = -1

    try:
        # plot the AUC, sensitivity, specificity, balanced_accuracy, weighted_score
        plt.figure(figsize=(16, 12))
        plt.subplot(2, 3, 1)
        plt.title("AUC")
        plt.xlabel("epoch")
        plt.ylabel("auc")
        plt.plot(metricues["auc"], label="auc")
        plt.legend()
        plt.subplot(2, 3, 2)
        plt.title("Sensitivity")
        plt.xlabel("epoch")
        plt.ylabel("sensitivity")
        plt.plot(metricues["sensitivity"], label="sensitivity")
        plt.legend()
        plt.subplot(2, 3, 3)
        plt.title("Specificity")
        plt.xlabel("epoch")
        plt.ylabel("specificity")
        plt.plot(metricues["specificity"], label="specificity")
        plt.legend()
        plt.subplot(2, 3, 4)
        plt.title("balanced Accuracy")
        plt.xlabel("epoch")
        plt.ylabel("f1 score")
        plt.plot(metricues["balanced_accuracy"], label="balanced_accuracy")
        plt.legend()
        plt.subplot(2, 3, 5)
        plt.title("Weighted Score")
        plt.xlabel("epoch")
        plt.ylabel("weighted_score")
        plt.plot(metricues["weighted_score"], label="weighted_score")
        plt.legend()
        plt.show()
        # plt.savefig(f"{current_datetime}_metric.jpg")
        plt.savefig(os.path.join(root_dir, f"{current_datetime}_lr{lr}_metric.jpg"))
        plt.cla()
    except:
        plt.cla()
        pass

    wandb.finish()
    run.finish()
