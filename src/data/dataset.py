import os, sys
from copy import deepcopy
from typing import Literal

from matplotlib import pyplot as plt
from data.synthesis import synthesize_body_measurements

from utils.utils import normalize_to_root

current = os.path.realpath(__file__)
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(current)))
os.chdir(base_dir)
sys.path.append(os.path.join(base_dir, "src"))

from data.dipimu import dipimu_fnames
from data.amass import amass_fnames
from data.totalcapture import totalcapture_fnames
import torch
from tqdm import tqdm
import config
from utils.rotation_conversions import rotation_converter
from torch.utils import data
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
from dataclasses import dataclass, field
from utils.smpl import SMPLModel, get_smpl_model
from loguru import logger

dataset_variables = {
    "poses": {
        "dim": 72,
        "description": "SMPL local joint angles",
        "getter": lambda x: x.reshape(-1, 72)[2:],
        "is_sequence": True,
    },
    "poses_global": {
        "dim": 72,
        "description": "SMPL global joint angles",
        "getter": lambda x: x.reshape(-1, 72)[2:],
        "is_sequence": True,
    },
    "joints": {
        "dim": 72,
        "description": "global joint positions (without translation)",
        "getter": lambda x: x.reshape(-1, 72)[2:],
        "is_sequence": True,
    },
    "betas": {
        "dim": 10,
        "description": "shape parameters",
        "getter": lambda x: x,
        "is_sequence": False,
    },
    "accs": {
        "dim": 18,
        "description": "linear acceleration (global frame without gravity) of six sensors",
        "getter": lambda x: x.reshape(-1, 18)[1:-1],
        "is_sequence": True,
    },
    "oris": {
        "dim": 54,
        "description": "global orientation of six sensors",
        "getter": lambda x: x.reshape(-1, 54)[2:],
        "is_sequence": True,
    },
    "dists": {
        "dim": 15,
        "description": "distance among six sensors",
        "getter": lambda x: x.reshape(-1, 15)[2:],
        "is_sequence": True,
    },
    "los_proportions": {
        "dim": 15,
        "description": "line-of-sight proportions among six sensors",
        "getter": lambda x: x.reshape(-1, 15)[2:],
        "is_sequence": True,
    },
    "meas": {
        "name": "meas",
        "dim": 13,
        "description": "body measurements",
        "getter": lambda x: torch.tensor(
            [x[feature.name] for feature in config.BodyMeasurement]
        ),
        "is_sequence": False,
    },
    "trans": {
        "dim": 3,
        "description": "translation of the body (root joint)",
        "getter": lambda x: x.reshape(-1, 3)[2:],
        "is_sequence": True,
    },
    "gender": {
        "name": "gender",
        "dim": 1,
        "description": "the gender of the subject",
        "getter": lambda x: torch.tensor([1]) if x == "male" else torch.tensor([0]),
        "is_sequence": False,
    },
}


@dataclass
class Motion:
    smpl_model: SMPLModel
    poses: torch.Tensor
    rep: Literal["axis_angle", "quat", "matrix", "rotation_6d"]
    shape: torch.Tensor
    trans: torch.Tensor
    poses_global: torch.Tensor = field(init=False)
    joints: torch.Tensor = field(init=False)
    vertices: torch.Tensor = field(init=False)

    def __post_init__(self):
        self.process_pose_joint_vertex()

    @property
    def joints_with_trans(self):
        return self.smpl_model.get_joints_with_trans(self.joints, self.trans)

    @property
    def vertices_with_trans(self):
        return self.smpl_model.get_vertices_with_trans(self.vertices, self.trans)

    def process_pose_joint_vertex(self):
        """process pose, joint, vertex"""
        if self.rep == "matrix":
            pjv = self.smpl_model(self.poses, self.shape, pose2rot=False)
        elif self.rep == "axis_angle":
            pjv = self.smpl_model(self.poses, self.shape, pose2rot=True)
        else:
            poses_in_axis_angle = rotation_converter(self.poses, self.rep, "axis_angle")
            pjv = self.smpl_model(poses_in_axis_angle, self.shape, pose2rot=True)
        self.poses_global = pjv["poses"]
        self.joints = pjv["joints"]
        self.vertices = pjv["vertices"]



class ShapeDataset:
    def __init__(self) -> None:
        self.betas = []
        self.meas = []
        self.genders = []

        self.mean_meas = dict()
        self.mean_dists = dict()
        mean_vertices = get_smpl_model(0).get_zero_pose_joint_and_vertex()[1]
        self.mean_meas[0] = synthesize_body_measurements(mean_vertices, get_smpl_model(0).faces)
        self.mean_dists[0] = (
            mean_vertices[config.masks.amass_vertices][config.masks.y_idxs] - mean_vertices[config.masks.amass_vertices][config.masks.x_idxs]
        ).norm(dim=-1)
        mean_vertices = get_smpl_model(1).get_zero_pose_joint_and_vertex()[1]
        self.mean_meas[1] = synthesize_body_measurements(mean_vertices, get_smpl_model(1).faces)
        self.mean_dists[1] = (
            mean_vertices[config.masks.amass_vertices][config.masks.y_idxs] - mean_vertices[config.masks.amass_vertices][config.masks.x_idxs]
        ).norm(dim=-1)

    def load(self, name):
        if name == "amass":
            data = torch.load(
                os.path.join(
                    config.paths.data_dir_body_measurements,
                    "amass_exclude_totalcapture.pt",
                ), weights_only=False, map_location="cpu"
            )
        if name == "totalcapture":
            data = torch.load(
                os.path.join(
                    config.paths.data_dir_body_measurements,
                    "amass_include_totalcapture.pt",
                ), weights_only=False, map_location="cpu"
            )
        self.betas.append(data["Betas"])
        self.meas.append(data["Meas"])
        self.genders.append(data["Genders"])

    def preprocessing(self):
        self.betas = torch.cat(self.betas)
        self.meas = torch.cat(self.meas)
        self.genders = torch.cat(self.genders)
        n = self.betas.shape[0]
        vertices = torch.stack(
            [
                get_smpl_model(self.genders[i]).get_zero_pose_joint_and_vertex(
                    self.betas[i]
                )[1]
                for i in range(n)
            ]
        )[:, config.masks.amass_vertices]
        self.dists = (
            vertices[:, config.masks.y_idxs] - vertices[:, config.masks.x_idxs]
        ).norm(dim=-1)


    def convert_gender_to_int(self, gender):
        return 0 if gender == "female" or gender == 0 else 1

    def get_meas_by_gender(self, gender):
        return self.meas[self.genders == self.convert_gender_to_int(gender)].float()

    def get_dists_by_gender(self, gender):
        return self.dists[self.genders == self.convert_gender_to_int(gender)].float()

    def get_betas_by_gender(self, gender):
        return self.betas[self.genders == self.convert_gender_to_int(gender)].float()

    def get_mean_meas_by_gender(self, gender, to_tensor=False):
        return self.mean_meas[self.convert_gender_to_int(gender)] if not to_tensor else torch.tensor(list(self.mean_meas[self.convert_gender_to_int(gender)].values()))

    def get_mean_dists_by_gender(self, gender):
        return self.mean_dists[self.convert_gender_to_int(gender)]

    def plot_betas_and_meas(self, figname="figures/betas-meas.pdf"):
        keys = [item.name for item in config.BodyMeasurement]
        width_per_figure = 4
        height_per_figure = 4
        n_rows = 10  # betas
        n_columns = len(keys)

        fig, axes = plt.subplots(
            n_rows,
            n_columns,
            figsize=(width_per_figure * n_columns, height_per_figure * n_rows),
        )

        for i_betas in range(n_rows):
            for i_meas in range(n_columns):
                ax = axes[i_betas][i_meas]
                meas_idx = config.BodyMeasurement[keys[i_meas]].value
                X = self.betas[:, i_betas]
                y = self.meas[:, meas_idx]  # measurement
                ax.scatter(X, y, s=5)
                ax.set_xlabel(f"Betas [{i_betas}]")
                ax.set_ylabel(f"{keys[i_meas]}")

        plt.tight_layout()
        plt.savefig(figname, dpi=300, format="pdf")
        plt.close()


class Dataset(data.Dataset):
    """Dataset

    training:
        for each sequence, split it into multiple sequences with length `seq_length`
        (if the length of the sequence is less than seq_length, then the sequence is discarded)
        all sequences are stored in a torch.Tensor
    testing:
        for each sequence, sequence data is stored in a list, while other data is stored in a torch.Tensor
    """

    def __init__(self, load_attrs: list = None) -> None:
        super().__init__()
        self.load_attrs = []
        if load_attrs:
            for attr in load_attrs:
                if attr not in self.load_attrs:
                    self.load_attrs.append(attr)
                    self._init_attr(attr)
        else:
            for k, v in dataset_variables.items():
                self.load_attrs.append(k)
                self._init_attr(k)
        self.missing_variables = set()
        self.fnames = []

    def _init_attr(self, attr):
        """initialize an attribute
        :param str attr: attribute name
        """
        setattr(self, attr, [])

    def load(self, name):
        if name.startswith("amass"):
            # amass: all amass data
            # amass_-TotalCapture: amass data, excluding totalcapture
            # amass_+TotalCapture: amass data, only including totalcapture
            excludes = []
            includes = []
            for subname in name.split("_")[1:]:
                if subname.startswith("-"):
                    excludes.append(subname[1:])
                elif subname.startswith("+"):
                    includes.append(subname[1:])
            logger.info(f"including {includes}")
            logger.info(f"excluding {excludes}")
            fnames = amass_fnames(processed=True, excludes=excludes, includes=includes)
        if name == "totalcapture_test":
            fnames = totalcapture_fnames(processed=True)
        if name == "dipimu_train":
            fnames = dipimu_fnames(processed=True, subject_ids=list(range(1, 9)))
        if name == "dipimu_test":
            fnames = dipimu_fnames(processed=True, subject_ids=[9, 10])

        self.fnames.extend(fnames)

        for fname in fnames:
            data = torch.load(fname, weights_only=False, map_location="cpu")
            for attr in self.load_attrs:
                if attr in data:
                    getattr(self, attr).append(
                        dataset_variables[attr]["getter"](data[attr])
                    )
                else:
                    self.missing_variables.add(attr)

    def preprocessing(self, seq_length=0, normalization=True):
        """stack data into torch.Tensor if
        1. seq_length == 0 and the data is not a sequence
        2. seq_length > 0
        """
        assert seq_length >= 0, "seq_length must be non-negative integer"
        self.seq_length = seq_length

        if self.seq_length == 0:  # no need to split
            for attr in self.load_attrs:
                if not dataset_variables[attr]["is_sequence"]:
                    setattr(self, attr, torch.stack(getattr(self, attr)).squeeze())

            if (
                normalization
                and "accs" in self.load_attrs
                and "oris" in self.load_attrs
            ):
                normalized_accs, normalized_oris = [], []
                for accs, oris in zip(self.accs, self.oris):
                    _accs, _oris = normalize_to_root(accs, oris)
                    normalized_accs.append(_accs)
                    normalized_oris.append(_oris)
                self.accs, self.oris = normalized_accs, normalized_oris
            return

        seq_sizes = []
        for attr in self.load_attrs:
            try:
                if dataset_variables[attr]["is_sequence"]:
                    logger.info(f"preprocessing {attr}")
                    items = []
                    for seq in tqdm(getattr(self, attr)):
                        if seq.shape[0] < self.seq_length:
                            # skip short sequences
                            logger.warning(
                                f"Skip short sequence: {seq.shape[0]} < {self.seq_length}"
                            )
                            items.append(torch.empty(0, *seq.shape[1:]))
                        elif seq.shape[0] == self.seq_length:
                            items.append(seq[None])
                        else:
                            items.append(
                                torch.cat(
                                    [
                                        torch.stack(seq.split(self.seq_length)[:-1]),
                                        seq[None, -self.seq_length :],
                                    ],
                                    dim=0,
                                )
                            )
                    if seq_sizes == []:
                        seq_sizes = [_.shape[0] for _ in items]
                    setattr(self, attr, torch.cat(items))
            except Exception as e:
                logger.error(f"{attr}: {e}")
                logger.error(f"{attr} is not processed")
                logger.error(f"missing variables: {self.missing_variables}")

        for attr in self.load_attrs:
            if not dataset_variables[attr]["is_sequence"]:
                logger.info(f"preprocessing {attr}")
                items = []
                for i, value in tqdm(enumerate(getattr(self, attr))):
                    if seq_sizes[i] > 0:
                        items.append(value.expand((seq_sizes[i], -1)))
                setattr(self, attr, torch.cat(items).squeeze())

        if normalization and "accs" in self.load_attrs and "oris" in self.load_attrs:
            logger.info("normalizing accs, oris...")
            self.accs, self.oris = normalize_to_root(self.accs, self.oris)

    def _concatenate(self, attrs: list):
        try:
            return torch.cat([getattr(self, _) for _ in attrs], dim=-1)
        except:
            seq_sizes = torch.tensor([_.shape[0] for _ in getattr(self, attrs[0])])
            return unpad_sequence(
                torch.cat(
                    [pad_sequence(getattr(self, _), batch_first=True) for _ in attrs],
                    dim=-1,
                ),
                lengths=seq_sizes,
                batch_first=True,
            )

    def get(self, attrs):
        if attrs is None:
            return None
        return self._concatenate(attrs)

    def __len__(self):
        return len(self.poses)


class RNNDataset(Dataset):
    def __init__(self, X, X_h, y, device="cpu"):
        self.X = X.to(device)
        self.X_h = (
            X_h.to(device) if X_h is not None else torch.zeros(X.shape[0]).to(device)
        )
        self.y = y.to(device)
        self.input_size = self.X.shape[-1]
        self.output_size = self.y.shape[-1]
        self.input_hidden_size = 0 if X_h is None else self.X_h.shape[-1]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.X_h[index], self.y[index]
