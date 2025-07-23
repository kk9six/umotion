from omegaconf import OmegaConf
import torch
import torch.nn as nn

import config
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.common.utils.utils import setup_outputdir
from autogluon.core.utils.loaders import load_pkl
from autogluon.core.utils.savers import save_pkl
import os.path as osp


class ShapeEstimator:
    multi_predictor_file = "multilabel_predictor.pkl"

    def __init__(
        self,
        labels,
        path=None,
        problem_types=None,
        eval_metrics=None,
        consider_labels_correlation=True,
        **kwargs,
    ):
        if len(labels) < 2:
            raise ValueError(
                "MultilabelPredictor is only intended for predicting MULTIPLE labels (columns), use TabularPredictor for predicting one label (column)."
            )
        if (problem_types is not None) and (len(problem_types) != len(labels)):
            raise ValueError(
                "If provided, `problem_types` must have same length as `labels`"
            )
        if (eval_metrics is not None) and (len(eval_metrics) != len(labels)):
            raise ValueError(
                "If provided, `eval_metrics` must have same length as `labels`"
            )
        self.path = setup_outputdir(path, warn_if_exist=False)
        self.labels = labels
        self.consider_labels_correlation = consider_labels_correlation
        self.predictors = (
            {}
        )  # key = label, value = TabularPredictor or str path to the TabularPredictor for this label
        if eval_metrics is None:
            self.eval_metrics = {}
        else:
            self.eval_metrics = {labels[i]: eval_metrics[i] for i in range(len(labels))}
        problem_type = None
        eval_metric = None
        for i in range(len(labels)):
            label = labels[i]
            path_i = osp.join(self.path, "Predictor_" + str(label))
            if problem_types is not None:
                problem_type = problem_types[i]
            if eval_metrics is not None:
                eval_metric = eval_metrics[i]
            self.predictors[label] = TabularPredictor(
                label=label,
                problem_type=problem_type,
                eval_metric=eval_metric,
                path=path_i,
                **kwargs,
            )

    def fit(self, train_data, tuning_data=None, **kwargs):
        if isinstance(train_data, str):
            train_data = TabularDataset(train_data)
        if tuning_data is not None and isinstance(tuning_data, str):
            tuning_data = TabularDataset(tuning_data)
        train_data_og = train_data.copy()
        if tuning_data is not None:
            tuning_data_og = tuning_data.copy()
        else:
            tuning_data_og = None
        save_metrics = len(self.eval_metrics) == 0
        for i in range(len(self.labels)):
            label = self.labels[i]
            predictor = self.get_predictor(label)
            if not self.consider_labels_correlation:
                labels_to_drop = [l for l in self.labels if l != label]
            else:
                labels_to_drop = [
                    self.labels[j] for j in range(i + 1, len(self.labels))
                ]
            train_data = train_data_og.drop(labels_to_drop, axis=1)
            if tuning_data is not None:
                tuning_data = tuning_data_og.drop(labels_to_drop, axis=1)
            print(f"Fitting TabularPredictor for label: {label} ...")
            predictor.fit(train_data=train_data, tuning_data=tuning_data, **kwargs)
            self.predictors[label] = predictor.path
            if save_metrics:
                self.eval_metrics[label] = predictor.eval_metric
        self.save()

    def predict(self, data, **kwargs):
        return self._predict(data, as_proba=False, **kwargs)

    def predict_proba(self, data, **kwargs):
        return self._predict(data, as_proba=True, **kwargs)

    def evaluate(self, data, **kwargs):
        data = self._get_data(data)
        eval_dict = {}
        for label in self.labels:
            print(f"Evaluating TabularPredictor for label: {label} ...")
            predictor = self.get_predictor(label)
            eval_dict[label] = predictor.evaluate(data, **kwargs)
            if self.consider_labels_correlation:
                data[label] = predictor.predict(data, **kwargs)
        return eval_dict

    def save(self):
        for label in self.labels:
            if not isinstance(self.predictors[label], str):
                self.predictors[label] = self.predictors[label].path
        save_pkl.save(
            path=osp.join(self.path, self.multi_predictor_file), object=self
        )
        print(
            f"MultilabelPredictor saved to disk. Load with: MultilabelPredictor.load('{self.path}')"
        )

    @classmethod
    def load(cls, path):
        path = osp.expanduser(path)
        return load_pkl.load(path=osp.join(path, cls.multi_predictor_file))

    def get_predictor(self, label):
        predictor = self.predictors[label]
        if isinstance(predictor, str):
            return TabularPredictor.load(path=predictor, require_version_match=False)
        return predictor

    def _get_data(self, data):
        if isinstance(data, str):
            return TabularDataset(data)
        return data.copy()

    def _predict(self, data, as_proba=False, **kwargs):
        data = self._get_data(data)
        if as_proba:
            predproba_dict = {}
        for label in self.labels:
            predictor = self.get_predictor(label)
            if as_proba:
                predproba_dict[label] = predictor.predict_proba(
                    data, as_multiclass=True, **kwargs
                )
            data[label] = predictor.predict(data, **kwargs)
        if not as_proba:
            return data[self.labels]
        else:
            return predproba_dict


class FcBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super(FcBlock, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc_dim = 512

        self.net = nn.Sequential(
            nn.Linear(self.input_size, self.fc_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.fc_dim, self.fc_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.fc_dim, self.output_size),
        )

    def forward(self, x):
        return self.net(x)


class PoseEstimator(nn.Module):
    def __init__(
        self,
        input_size: int,
        input_hidden_size: int,
        output_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.4,
    ) -> None:
        """RNN model
        :param int input_dim: input dimension
        :param int input_hidden_size: input hidden dimension
        :param int output_dim: output dimension
        :param int hidden_dim: hidden dimension
        :param int n_layers: number of layers (default: 2)
        :param float dropout: dropout rate (default: 0.0)
        """
        super(PoseEstimator, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.rnn = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers,
            bidirectional=False,  # for online inference
            dropout=dropout,
            batch_first=True,  # batch size is the first dimension
        )
        self.pose_output = nn.Linear(hidden_size, output_size)
        self.logstd_output = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.h, self.c = None, None  # hidden states
        if input_hidden_size > 0:
            self.init_net = nn.Sequential(
                nn.Linear(input_hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size * num_layers),
                nn.ReLU(),
                nn.Linear(
                    hidden_size * num_layers, hidden_size * num_layers * 2  # h, c
                ),
            )

    def get_hidden_state(self, x_h=None):
        """
        get hidden states
        :param torch.Tensor x_h: (batch_size, output_size) or (output_size)

        :return: (h, c)
        """
        if x_h is not None:
            x_h = x_h.view(-1, x_h.shape[-1])  # (batch_size, output_size)
            self.h, self.c = self.init_net(x_h).view(
                2,  # (h, c)
                self.rnn.num_layers,
                x_h.shape[0],
                self.rnn.hidden_size,
            )  # (2, n_layers, batch_size, hidden_size)

        if self.h is not None and self.c is not None:
            return self.h, self.c

    def forward(self, x, x_h=None):
        """
        :param torch.Tensor x: (batch_size, seq_length, input_size) or (seq_length, input_size)
        :param torch.Tensor init_state: (output_size) or (batch_size, output_size)
        :return torch.Tensor: (batch_size, seq_length, output_size) or (seq_length, output_size)

        Notes:
        discarding short sequence does not negatively affect the performance
        """
        if x.dim() == 1:
            x = x.view(1, 1, x.shape[-1])
        else:
            x = x.view(-1, x.shape[-2], x.shape[-1])

        y, (self.h, self.c) = self.rnn(
            self.dropout(self.relu(self.linear1(x))),
            self.get_hidden_state(x_h),
        )
        pose = self.pose_output(y).squeeze()
        logstd = self.logstd_output(y).squeeze()
        return pose, logstd

def load_pose_estimator(model_name, return_cfg=False, device="cuda", checkpoint=None):
    dirpath = osp.join(config.paths.base_dir, "models", model_name)
    cfg = OmegaConf.load(osp.join(dirpath, "config.yaml"))
    pose_estimator = PoseEstimator(
        input_size=cfg.pose_estimator.input_size,
        input_hidden_size=cfg.pose_estimator.input_hidden_size,
        output_size=cfg.pose_estimator.output_size,
        hidden_size=cfg.pose_estimator.hidden_size,
        num_layers=cfg.pose_estimator.num_layers,
        dropout=cfg.pose_estimator.dropout,
    ).to(device)
    if checkpoint is not None:
        pose_estimator.load_state_dict(torch.load(osp.join(cfg.dirs.checkpoints, f"{checkpoint:04d}.pt")))
    else:
        pose_estimator.load_state_dict(torch.load(osp.join(cfg.dirs.pose_estimator, "model.pt")))
    pose_estimator.eval()
    if return_cfg:
        return pose_estimator, cfg
    return pose_estimator
