import torch
import torch.nn as nn
import torch.nn.functional as F
import monai
import sys

sys.path.append("torchmtlr")
from torchmtlr import (
    MTLR,
)
        
class PrognosisModel(nn.Module):
    def __init__(
        self,
        spatial_dims=3,
        in_channels=1,
        emb_size=32,
        out_channels=18,
        intervention=False,
        prognosis=False,
        pre_prognosis="class",
        num_time_bins=0,
        image_only=False,
        random_intervention_thresh=0.25,
        device="cuda",
    ):
        super(PrognosisModel, self).__init__()
        num_embeddings = out_channels * emb_size
        self.out_channels = out_channels
        self.intervention = intervention
        self.random_intervention_thresh = random_intervention_thresh
        self.device = device

        self.encoder = monai.networks.nets.DenseNet121(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=num_embeddings,
        )

        self.concept_pred = []
        self.concept_act = []
        self.weightage = []
        self.classifier = []
        for i in range(self.out_channels):
            self.concept_pred.append(
                nn.Linear(in_features=num_embeddings, out_features=emb_size)
            )  # even
            self.concept_act.append(nn.LeakyReLU())  # even
            self.weightage.append(nn.Linear(in_features=emb_size, out_features=1))  # odd

            if self.intervention:
                self.classifier.append(
                    nn.Linear(in_features=int(emb_size / 2), out_features=1)
                )  # odd
            else:
                self.classifier.append(
                    nn.Linear(in_features=emb_size, out_features=1)
                )  # odd

        self.concept_pred = nn.ModuleList(self.concept_pred)
        self.concept_act = nn.ModuleList(self.concept_act)
        self.weightage = nn.ModuleList(self.weightage)
        self.classifier = nn.ModuleList(self.classifier)

        self.sigmoid = nn.Sigmoid()

        self.prognosis = prognosis
        self.pre_prognosis = pre_prognosis
        self.image_only = image_only

        if self.pre_prognosis == "class":
            num_prog_in = out_channels
        elif self.pre_prognosis == "concept":
            num_prog_in = num_embeddings

            if self.intervention:
                num_prog_in = int(num_prog_in / 2)

        if self.image_only:
            self.encoder = monai.networks.nets.DenseNet121(
                spatial_dims=spatial_dims, in_channels=in_channels, out_channels=256
            )
            num_prog_in = 256

        if self.prognosis == "regression":
            self.prognosis_layer = nn.Linear(in_features=num_prog_in, out_features=1)
        elif self.prognosis == "mtlr":
            self.prognosis_layer = MTLR(
                in_features=num_prog_in, num_time_bins=num_time_bins
            )
        elif self.prognosis == "relu_mtlr":
            self.prognosis_layer = nn.Sequential(
                nn.ReLU(),
                MTLR(in_features=num_prog_in, num_time_bins=num_time_bins),
            )
        elif self.prognosis == "relu_mtlr_sig":
            self.prognosis_layer = nn.Sequential(
                nn.ReLU(),
                MTLR(in_features=num_prog_in, num_time_bins=num_time_bins),
                nn.Sigmoid(),
            )
        elif self.prognosis == "relu_lin":
            self.prognosis_layer = nn.Sequential(
                nn.ReLU(),
                nn.Linear(in_features=num_prog_in, out_features=num_time_bins),
            )
        elif self.prognosis == "relu_lin_sig":
            self.prognosis_layer = nn.Sequential(
                nn.ReLU(),
                nn.Linear(in_features=num_prog_in, out_features=num_time_bins),
                nn.Sigmoid(),
            )

    def forward(self, x, gt_class=None, train=False, multiclass_dict=None):
        x = self.encoder(x)

        if self.image_only:
            prognosis_out = self.prognosis_layer(x)
            return None, prognosis_out
        
        class_pred_out = torch.tensor([], dtype=torch.float32, device=self.device)
        class_emb_out = torch.tensor([], dtype=torch.float32, device=self.device)

        gt_class2 = gt_class.float()
        for each_categ, slice_idx in multiclass_dict.items():
            # find rows with missing vals
            missing_val_idx = gt_class2[:, slice_idx].sum(axis=1) == 0

            for i in range(slice_idx.start, slice_idx.stop):
                x_emb = self.concept_pred[i](x)

                if self.intervention:
                    x_emb = self.concept_act[i](x_emb)
                    x_weightage = self.weightage[i](x_emb)
                    random_intervention = torch.rand(x.shape[0])
                    random_intervention_mask = (
                        random_intervention < self.random_intervention_thresh
                    )

                    probs = self.sigmoid(x_weightage)
                    probs = probs.as_tensor().float()

                    gt_class2[:, i : i + 1][missing_val_idx] = probs[missing_val_idx]
                        
                    if train:
                        # replace probs with gt_class values based on random_intervention probability
                        probs[random_intervention_mask.unsqueeze(-1)] = gt_class2[
                            random_intervention_mask, i
                        ]
                    else:
                        # else for val, replace probs completely with gt_class values
                        probs = gt_class2[:, i : i + 1]

                    x_emb_pos = x_emb[..., : int(x_emb.shape[-1] / 2)]
                    x_emb_neg = x_emb[..., int(x_emb.shape[-1] / 2) :]

                    x_emb = probs * x_emb_pos + (1 - probs) * x_emb_neg

                x_class = self.classifier[i](x_emb)

                class_pred_out = torch.cat([class_pred_out, x_class], dim=1)  # logits
                class_emb_out = torch.cat([class_emb_out, x_emb], dim=1)  # logits

        if self.pre_prognosis == "class":
            prog_in = class_pred_out
        elif self.pre_prognosis == "concept":
            prog_in = class_emb_out

        if self.prognosis:
            prognosis_out = self.prognosis_layer(prog_in)
            return class_pred_out, prognosis_out

        else:
            return class_pred_out, None


def conv_3d_block(in_c, out_c, act="relu", norm="bn", num_groups=8, *args, **kwargs):
    activations = nn.ModuleDict(
        [["relu", nn.ReLU(inplace=True)], ["lrelu", nn.LeakyReLU(0.1, inplace=True)]]
    )

    normalizations = nn.ModuleDict(
        [
            ["bn", nn.BatchNorm3d(out_c)],
            ["gn", nn.GroupNorm(int(out_c / num_groups), out_c)],
        ]
    )

    return nn.Sequential(
        nn.Conv3d(in_c, out_c, *args, **kwargs),
        normalizations[norm],
        activations[act],
    )


def flatten_layers(arr):
    return [i for sub in arr for i in sub]


class Dual_MTLR(nn.Module):
    """
    Adapted with modifications from https://github.com/numanai/BioMedIA-Hecktor2021/blob/main/src/models/deepmtlr_model.py
    An Ensemble Approach for Patient Prognosis of Head and Neck Tumor Using Multimodal Data (Saeed et al., 2021)
    """

    def __init__(
        self,
        spatial_dims=3,
        in_channels=1,
        num_time_bins=0,
        out_channels=18,
        k1=3,
        k2=5,
        n_dense=2,
        dense_factor=1,
        dropout=0.2,
    ):
        super().__init__()

        n_clin_var = out_channels
        num_embeddings = 256
        self.cnn = monai.networks.nets.DenseNet121(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=num_embeddings,
        )
        # self.cnn = nn.Sequential(#block 1
        #              conv_3d_block(1, 32, kernel_size=k1),
        #              conv_3d_block(32, 64, kernel_size=k2),
        #              nn.MaxPool3d(kernel_size=2, stride=2),

        #              #block 2
        #              conv_3d_block(64, 128, kernel_size=k1),
        #              conv_3d_block(128, 256, kernel_size=k2),
        #              nn.MaxPool3d(kernel_size=2, stride=2),
        #              nn.AdaptiveAvgPool3d(1),
        #              nn.Flatten()
        #         )

        if n_dense <= 0:
            # self.mtlr = MTLR(256 + n_clin_var, num_time_bins)
            self.mtlr = nn.Linear(256 + n_clin_var, num_time_bins)

        else:
            fc_layers = [
                [
                    nn.Linear(256 + n_clin_var, 512 * dense_factor),
                    nn.BatchNorm1d(512 * dense_factor),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            ]

            if n_dense > 1:
                fc_layers.extend(
                    [
                        [
                            nn.Linear(512 * dense_factor, 512 * dense_factor),
                            nn.BatchNorm1d(512 * dense_factor),
                            nn.ReLU(inplace=True),
                            nn.Dropout(dropout),
                        ]
                        for _ in range(n_dense - 1)
                    ]
                )

            fc_layers = flatten_layers(fc_layers)
            self.mtlr = nn.Sequential(
                *fc_layers,
                # MTLR(512 * dense_factor, num_time_bins),)
                nn.Linear(512 * dense_factor, num_time_bins),
            )

    def forward(self, x, gt_class):
        cnn = self.cnn(x)
        ftr_concat = torch.cat((cnn, gt_class), dim=1)
        prognosis_out = self.mtlr(ftr_concat)
        return None, prognosis_out
