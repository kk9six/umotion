from functools import partial
import numpy as np
import torch, random, sys, os
import os.path as osp
from network.losses import PoseLoss
from network.network import ShapeEstimator,PoseEstimator
from utils.utils import plot_losses
from data.dataset import Dataset, RNNDataset, ShapeDataset
from loguru import logger
from utils.rotation_conversions import axis_angle_to_rotation_6d
import config
import pandas as pd

def seed_everything(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ShapeEstimatorTrainer:

    @classmethod
    def train(cls, cfg):
        seed_everything(cfg.seed)

        logger.remove()
        logger.add(
            osp.join(cfg.dirs.shape_estimator, "run.log"), enqueue=True, level="INFO"
        )
        logger.add(sys.stdout, colorize=True, enqueue=True, level="INFO")
        logger.info(f"Training ShapeEstimator at {cfg.dirs.shape_estimator}")

        logger.info("Loading data: AMASS")
        shape_data = ShapeDataset()
        shape_data.load("amass")
        logger.info("Preprocessing data")
        shape_data.preprocessing()

        for gender in [0, 1]:
            H = shape_data.get_meas_by_gender(gender)[:, 0].reshape(-1, 1)
            H -= torch.tensor(list(shape_data.mean_meas[gender].values()))[0]
            W = shape_data.get_meas_by_gender(gender)[:, 1].reshape(-1, 1)
            W -= torch.tensor(list(shape_data.mean_meas[gender].values()))[1]
            W /= 100
            D = shape_data.get_dists_by_gender(gender)[:, config.masks.D_idxs] - shape_data.mean_dists[gender][config.masks.D_idxs]
            HWD = {"H": H, "W": W, "D": D}
            X_train = torch.cat([HWD[_] for _ in cfg.shape_estimator.input_attrs], dim=-1)
            y_train = shape_data.get_betas_by_gender(gender)
            logger.info(
                f"Training: {'male' if gender == 1 else 'female'}"
            )
            train_data = pd.DataFrame(torch.cat([X_train, y_train], dim=1))
            labels = np.arange(train_data.shape[1])[-10:]
            problem_type = ["regression"] * 10
            eval_metric = ["root_mean_squared_error"] * 10
            save_path = osp.join(cfg.dirs.shape_estimator, f"model_{gender}")
            ShapeEstimator(
                labels=labels,
                problem_types=problem_type,
                eval_metrics=eval_metric,
                path=save_path,
            ).fit(train_data)


class PoseEstimatorTrainer:

    @staticmethod
    def augmentation_noise(x, noise_std):
        noise = torch.zeros_like(x)
        noise[..., 72:] = torch.randn_like(noise[..., 72:]) * noise_std
        return noise.to(x.device)

    @classmethod
    def train(cls, cfg):
        seed_everything(cfg.seed)

        logger.remove()
        logger.add(osp.join(cfg.dirs.pose_estimator, "run.log"), enqueue=True, level="INFO")
        logger.add(sys.stdout, colorize=True, enqueue=True, level="INFO")
        logger.info(f"Training PoseEstimator at {cfg.dirs.pose_estimator}")

        augment = (
            None
            if cfg.training.noise_range is None
            else partial(cls.augmentation_noise, noise_std=cfg.training.noise_range)
        )
        logger.info(f"Augmentation: {cfg.training.noise_range}")

        # Load data
        logger.info(f"Loading training data")
        motions = Dataset(
            load_attrs=cfg.pose_estimator.input_attrs
            + cfg.pose_estimator.input_hidden_attrs
            + cfg.pose_estimator.output_attrs
            + ["accs"]
        )
        for name in cfg.dataset.training_data:
            logger.info(f"- Loading {name}")
            motions.load(name)
        logger.info("Preprocessing all data")
        motions.preprocessing(cfg.training.seq_length, cfg.training.normalization)

        logger.info("Preprocessing X, X_h, y for training")
        logger.info(f"Target joints: {config.masks.target_joints}")
        logger.info(f"Ignore joints: {config.masks.ignored_joints}")
        X = motions.get(cfg.pose_estimator.input_attrs)[
            :, 1:
        ]  # ignore the first frame, which is used for initial hidden state
        X_h = motions.get(cfg.pose_estimator.input_hidden_attrs)[:, 0]
        y = motions.get(cfg.pose_estimator.output_attrs)[:, 1:]  # ignore the first frame
        y = axis_angle_to_rotation_6d(
            y.reshape(-1, cfg.training.seq_length - 1, 24, 3)[
                :, :, config.masks.target_joints
            ]
        ).flatten(-2)
        logger.info(f"X shape: {X.shape}")
        logger.info(f"X_h shape: {X_h.shape}")
        logger.info(f"y shape: {y.shape}")

        logger.info("Creating RNNDataset")
        train_data = RNNDataset(X, X_h, y, device="cuda")

        logger.info("Creating model")
        model = PoseEstimator(
            input_size=cfg.pose_estimator.input_size,
            input_hidden_size=cfg.pose_estimator.input_hidden_size,
            output_size=cfg.pose_estimator.output_size,
            hidden_size=cfg.pose_estimator.hidden_size,
            num_layers=cfg.pose_estimator.num_layers,
            dropout=cfg.pose_estimator.dropout,
        ).cuda()
        if cfg.pose_estimator.base_model_path is not None:
            model.load_state_dict(
                torch.load(
                    osp.join(
                        cfg.dirs.models,
                        cfg.pose_estimator.base_model_path,
                        "model.pt",
                    )
                )
            )
        critirion = PoseLoss(pretrain_epoch=cfg.training.pretrain_epoch).cuda()
        logger.info("Training model")
        losses, val_losses = cls._do_train(cfg, train_data, model, critirion, augment)
        logger.info("Saving model")
        torch.save(model.state_dict(), osp.join(cfg.dirs.pose_estimator, "model.pt"))
        logger.info("Saving losses")
        torch.save(
            {"losses": losses, "val_losses": val_losses},
            osp.join(cfg.dirs.pose_estimator, "losses.pt"),
        )
        plot_losses(
            losses,
            val_losses,
            cfg.training.val_step,
            cfg.training.num_epochs,
            osp.join(cfg.dirs.pose_estimator, "losses.pdf"),
        )

    @staticmethod
    def _do_train(cfg, train_data, model, critirion, augment=None):
        # Split dataset
        train_subset, val_subset = torch.utils.data.random_split(
            train_data, [cfg.training.train_ratio, cfg.training.val_ratio]
        )
        train_loader = torch.utils.data.DataLoader(
            train_subset, batch_size=cfg.training.batch_size, shuffle=False
        )
        val_loader = torch.utils.data.DataLoader(
            val_subset, batch_size=cfg.training.batch_size, shuffle=False
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
        losses = []
        val_losses = []

        for epoch in range(1, cfg.training.num_epochs + 1):
            # Training
            accumulator = {"count": 0, "loss": 0.0}
            model.train()
            for X, X_h, y in train_loader:
                optimizer.zero_grad(set_to_none=True)
                if augment:
                    y_pred, y_pred_logstd = model(X + augment(X), X_h)
                else:
                    y_pred, y_pred_logstd = model(X, X_h)
                loss = critirion(y_pred, y_pred_logstd, y, epoch)
                loss.backward()
                optimizer.step()
                accumulator["count"] += 1
                accumulator["loss"] += loss.item()
            losses.append(accumulator["loss"] / accumulator["count"])
            logger.info(f"Epoch {epoch} loss: {losses[-1]}")

            # Validation
            if (epoch % cfg.training.val_step == 0) or (epoch == cfg.training.num_epochs):  # validate every step epochs or at the end of training
                accumulator = {"count": 0, "loss": 0.0}
                for X, X_h, y in val_loader:
                    model.eval()
                    with torch.no_grad():
                        if augment:
                            y_pred, y_pred_logstd = model(X + augment(X), X_h)
                        else:
                            y_pred, y_pred_logstd = model(X, X_h)
                        loss = critirion(y_pred, y_pred_logstd, y, epoch)
                        accumulator["count"] += 1
                        accumulator["loss"] += loss.item()
                val_losses.append(accumulator["loss"] / accumulator["count"])
                logger.info(f"Epoch {epoch} val loss: {val_losses[-1]}")

            # Save model
            if (epoch % cfg.training.save_step == 0) or (epoch == cfg.training.num_epochs):
                torch.save(model.state_dict(), osp.join(cfg.dirs.checkpoints, f"{epoch:04d}.pt"))
                logger.info(f"Saved model at {epoch}")

        return losses, val_losses
