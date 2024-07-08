import logging
import os
import sys
import shutil

import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve

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
    Compose,
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

from datetime import datetime
from torch.optim.lr_scheduler import OneCycleLR

from sklearn.metrics import brier_score_loss, roc_auc_score

from lifelines.utils import concordance_index

sys.path.append("torchmtlr")
from torchmtlr import (
    MTLR,
    mtlr_neg_log_likelihood,
    mtlr_survival,
    mtlr_survival_at_times,
    mtlr_risk,
)
from torchmtlr.utils import make_time_bins, encode_survival

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
from dataset.utils import *
from dataset.hecktor import *
from dataset.chaim import *
from prognosis_models import PrognosisModel, Dual_MTLR

parser = argparse.ArgumentParser()

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
    "--data_dir", type=str, help="path to data directory"
)
parser.add_argument(
    "--dataset_name", type=str, default="lung", choices=["lung", "hecktor"]
)
parser.add_argument(
    "--output_dir", type=str, help="path to checkpoint and results"
)

cfg = parser.parse_args()
SEED = cfg.seed
fold = cfg.fold
impute = cfg.impute
impute_percent = cfg.impute_percent

if cfg.image_only == "True":
    image_only = True
elif cfg.image_only == "False":
    image_only = False

if cfg.model == "Dual_MTLR":
    dual_mtlr = True
else:
    dual_mtlr = False

# Setting the seed
set_determinism(SEED)
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

pre_prognosis = "concept" #"class" #"X"
prognosis = "relu_lin"
concept_act = "leakyrelu"

if cfg.model == "PrognosisModel":
    intervention = True
    if intervention:
        random_intervention_thresh = 0.25
else:
    intervention = False

mtlr_loss = cfg.mtlr_loss  # "prog_corn_rank" #"deephit" #None
prog_loss_wt = 0.2
lr = 1e-3
batch_size = 32
emb_size = 64
number_warmup_epochs = 5
max_epochs = 100

if "mtlr" in prognosis and "deephit" not in mtlr_loss:
    mtlr_loss = "original_mtlr"

current_datetime = f"{current_datetime}_HuLP_{impute}_{impute_percent}_{mtlr_loss}_seed{SEED}_fold{fold}"
root_dir = f"{cfg.output_dir}/{current_datetime}"

if not os.path.exists(root_dir):
    os.makedirs(root_dir)

# TODO: move to utils
def create_multiclass_dict(data_train, one_hot_cols):
    i = 0
    multiclass_dict = {}
    for col in one_hot_cols:
        # num_unique_vals = data_train[col].nunique()
        num_unique_vals = len([i for i in data_train.columns if col in i])
        
        # TODO: make more dynamic to account for unordered columns
        multiclass_dict.update({col: slice(i, i + num_unique_vals)})
        i += num_unique_vals

    return multiclass_dict

if cfg.dataset_name == "lung":
    df_clinical, data_train = make_chaimeleon_data()
    in_channels = 1
    one_hot_cols = [
            "gender",
            "smoking_status",
            "clinical_category",
            "regional_nodes_category",
            "metastasis_category",
        ]

elif cfg.dataset_name == "hecktor":
    df_clinical, data_train = make_hecktor_data()
    in_channels = 2
    one_hot_cols = [
            "Gender",
            "Chemotherapy",
            "HPV_status",
            "T-stage",
            "N-stage",
            "M-stage",
        ]

data_time_train = data_train["time"]
data_event_train = data_train["event"]

time_bins = make_time_bins(
    data_time_train.values, event=data_event_train.values
)  # note: time bins are created based on full dataset (not just training!)
num_time_bins = len(time_bins)
    
eval_times = np.quantile(
    data_train.loc[data_train["event"] == 1, "time"], [0.25, 0.5, 0.75]
).astype("int")

data_train = data_train[one_hot_cols]

patientIDs = data_train.index.to_list()
print("patientIDs", len(patientIDs))

if cfg.dataset_name == "lung":
    train_transforms, val_transforms = lung_ct_transforms()

    with open(
        f"{cfg.data_dir}/lung_exps/prognosis/chaimeleon_lungCancer_5CV_patientSplit.json",
        "r",
    ) as f:
        patientSplit = json.load(f)
    train_id = patientSplit[f"Fold{fold}"]["train"]
    val_id = patientSplit[f"Fold{fold}"]["val"]

elif cfg.dataset_name == "hecktor":
    train_transforms, val_transforms = hecktor_ct_pet_transforms()

    with open(
        f"dataset/splits/hecktor2021_5CV_patientPathSplit_seed0.json",
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
    # ct_train = train_id
    train_id = [i for i in train_id if "case_0346" not in i] # TODO: remove case_0346
    # ct_val = val_id
    val_id = [i for i in val_id if "case_0346" not in i]

    if impute:
        data_train = imputer(data_train, impute, impute_percent, SEED, idx_train=train_id, cfg=cfg)
    else:
        for i in ['gender', 'smoking_status', 'clinical_category', 'regional_nodes_category', 'metastasis_category']: #df_clinical.columns[4:]:
            if impute_percent != 30:
                data_train[i] = data_train[i].sample(frac=impute_percent / 100)

    data_train = one_hot_encode(data_train, one_hot_cols, cfg)
    multiclass_dict = create_multiclass_dict(data_train, one_hot_cols)

    num_classes = len(data_train.columns) - len(
        [i for i in data_train.columns if "_X" in i]
    )
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

    train_files = [
        {
            "image": os.path.basename(ct_in).replace(".nii.gz", ""),
            "path_to_h5": PATH_TO_H5,
            "label": gtvt_in,
            "time": time_in,
            "event": event_in,
        }
        for ct_in, gtvt_in, time_in, event_in in zip(
            train_id, label_train, time_train, event_train
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
            val_id, label_val, time_val, event_val
        )
    ]
    
elif cfg.dataset_name == "hecktor":

    if impute:
        data_train = imputer(data_train, impute, impute_percent, SEED, idx_train=ct_train, cfg=cfg)
     
    data_train = one_hot_encode(data_train, one_hot_cols, cfg)
    multiclass_dict = create_multiclass_dict(data_train, one_hot_cols)
    
    num_classes = len(data_train.columns) - len(
        [i for i in data_train.columns if "_X" in i]
    )
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

if cfg.model == "PrognosisModel":
    model = PrognosisModel(
        spatial_dims=3,
        in_channels=in_channels,
        emb_size=emb_size,
        out_channels=len(data_train.columns),
        intervention=intervention,
        pre_prognosis=pre_prognosis,
        prognosis=prognosis,
        num_time_bins=num_time_bins,
        image_only=image_only,
        random_intervention_thresh=random_intervention_thresh,
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

y_pred_others_trans = Compose([Activations(softmax=True)])

# start a typical PyTorch training
val_interval = 1
auc_metric = ROCAUCMetric()

best_metric = -1
best_metric_epoch = -1
epoch_losses = []
metricues = {
    "auc": [],
    "sensitivity": [],
    "specificity": [],
    "balanced_accuracy": [],
    "weighted_score": [],
}

# start a new wandb run to track this script
run = wandb.init(
    # set the wandb project where this run will be logged
    project="IPEM_" + cfg.dataset_name,
    name=current_datetime,
    # track hyperparameters and run metadata
    config={
        "seed": SEED,
        "fold": fold,
        "impute": impute,
        "impute_percent": impute_percent,
        "learning_rate": lr,
        "epochs": max_epochs,
        "num_classes": num_classes,
        "batch_size": batch_size,
        "model": cfg.model,
        "pre_prognosis": pre_prognosis,
        "prognosis": prognosis,
        "mtlr_loss": mtlr_loss,
        "num_time_bins": num_time_bins,
        "image_only": image_only,
        "emb_size": emb_size,
        "prog_loss_wt": prog_loss_wt,
        "intervention": intervention,
        "random_intervention_thresh": random_intervention_thresh,
        "concept_act": concept_act,
        "num_warmup_epochs": number_warmup_epochs,
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
        y_prog = encode_survival(
            times.cpu().numpy(), events.cpu().numpy(), time_bins
        ).to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            if dual_mtlr:
                outputs, outputs_prog = model(inputs, gt_class=labels)
            elif intervention:
                outputs, outputs_prog = model(inputs, gt_class=labels, train=True, multiclass_dict=multiclass_dict)
            else:
                outputs, outputs_prog = model(inputs, gt_class=None)

            if outputs_prog is not None:
                y_pred_prog = torch.cat([y_pred_prog, outputs_prog], dim=0)
                y_time = torch.cat([y_time, times], dim=0)
                y_event = torch.cat([y_event, events], dim=0)

                if mtlr_loss == "original_mtlr":
                    y_prog = deephit_encode_survival(
                        times.cpu().numpy(), events.cpu().numpy(), time_bins
                    ).to(device)
                    y_prog = y_prog.argmax(dim=1)
                    prog_loss = deepmtlr_loss(
                        outputs_prog.as_tensor(), y_prog, events
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
            ): 
                
                # for each category
                for each_categ, slice_idx in multiclass_dict.items():
                    # find rows with missing vals
                    missing_val_idx = labels[:, slice_idx].sum(axis=1) == 0
                    
                    # skip loss calculation where there are missing values
                    outputs_categ = outputs[:, slice_idx].as_tensor()[~missing_val_idx]
                    labels_categ = labels[:, slice_idx][~missing_val_idx]
                    
                    loss_categ = ce_loss_function(outputs_categ, labels_categ)
                    loss += loss_categ
                    
                loss *= prog_loss_wt

            if outputs_prog is not None:
                loss += (1-prog_loss_wt) * prog_loss

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
                stepp = step - 1
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")

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

        wandb.log(
            {
                "trn_concordance_idx": train_ci,
            },
            step=epoch + 1,
        )

    epoch_loss /= stepp
    epoch_losses.append(epoch_loss)
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
                elif intervention or impute != "None":
                    val_outputs, val_outputs_prog = model(
                        val_images, gt_class=val_labels, train=False, multiclass_dict=multiclass_dict
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
                y_all = []
                y_pred_all = []
                # for each category
                for each_categ, slice_idx in multiclass_dict.items():
                    # find rows with missing vals
                    missing_val_idx = y[:, slice_idx].sum(axis=1) == 0
                    
                    # skip loss calculation where there are missing values
                    # gt_labels
                    labels_categ = y[:, slice_idx][~missing_val_idx]

                    # outputs
                    outputs_categ = y_pred[:, slice_idx].as_tensor()[~missing_val_idx]
                    
                    num_classes_per_categ = len(range(slice_idx.stop)[slice_idx])
                    y_transform = AsDiscrete(to_onehot=num_classes_per_categ) # TODO: bring outside loop...just define once
                    
                    outputs_categ = [
                        y_transform(y_pred_others_trans(item).argmax())
                        for item in outputs_categ
                    ]
                    outputs_categ = torch.stack(outputs_categ, dim=0)
                    
                    y_all.append(labels_categ)
                    y_pred_all.append(outputs_categ)

                y = torch.cat(y_all, dim=1)
                y_pred = torch.cat(y_pred_all, dim=1)

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
                            root_dir, f"best_metric_dense121_HuLP_lr{lr}.pth"
                        ),
                    )
                    # artifact = wandb.Artifact("model_1", type="model")
                    # artifact.add_file(
                    #     os.path.join(
                    #         root_dir, f"best_metric_dense121_HuLP_lr{lr}.pth"
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
                    os.path.join(root_dir, f"last_metric_dense121_HuLP_lr{lr}.pth"),
                )
                # artifact = wandb.Artifact("model_4", type="model")
                # artifact.add_file(
                #     os.path.join(root_dir, f"last_metric_dense121_HuLP_lr{lr}.pth")
                # )
                # run.log_artifact(artifact)
                torch.save(
                    optimizer.state_dict(),
                    os.path.join(root_dir, f"last_opt_dense121_HuLP_lr{lr}.pth"),
                )
                # artifact = wandb.Artifact("model_5", type="model")
                # artifact.add_file(
                #     os.path.join(root_dir, f"last_opt_dense121_HuLP_lr{lr}.pth")
                # )
                # run.log_artifact(artifact)
                torch.save(
                    scheduler.state_dict(),
                    os.path.join(root_dir, f"last_sch_dense121_HuLP_lr{lr}.pth"),
                )
                # artifact = wandb.Artifact("model_6", type="model")
                # artifact.add_file(
                #     os.path.join(root_dir, f"last_sch_dense121_HuLP_lr{lr}.pth")
                # )
                # run.log_artifact(artifact)

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
                    os.path.join(root_dir, f"last_opt_dense121_HuLP_lr{lr}.pth"),
                )
                # artifact = wandb.Artifact("model_14", type="model")
                # artifact.add_file(
                #     os.path.join(root_dir, f"last_opt_dense121_HuLP_lr{lr}.pth"),
                # )
                # run.log_artifact(artifact)
                torch.save(
                    scheduler.state_dict(),
                    os.path.join(root_dir, f"last_sch_dense121_HuLP_lr{lr}.pth"),
                )
                # artifact = wandb.Artifact("model_15", type="model")
                # artifact.add_file(
                #     os.path.join(root_dir, f"last_sch_dense121_HuLP_lr{lr}.pth"),
                # )
                # run.log_artifact(artifact)

    del y_pred, y_pred_prog, y, y_time, y_event

print(
    f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}"
)

wandb.finish()
run.finish()