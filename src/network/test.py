import os, torch, json
import sys
import pandas as pd
import os.path as osp
from omegaconf import OmegaConf
from loguru import logger
from prettytable import PrettyTable
import config
from data.dataset import Dataset, Motion, ShapeDataset
from utils.smpl import get_smpl_model
from utils.rotation_conversions import rotation_6d_to_matrix, axis_angle_to_matrix
from network.network import PoseEstimator, ShapeEstimator
from network.eval import PoseEvaluator, ShapeEvaluator


class ShapeEstimatorTester:
    @classmethod
    def test(cls, dirpath):
        if not os.path.exists(dirpath):
            raise FileNotFoundError(f"Path {dirpath} not found")

        # Load parameters
        cfg = OmegaConf.load(osp.join(dirpath, "config.yaml"))

        logger.remove()
        logger.add(osp.join(cfg.dirs.shape_estimator, "run.log"), enqueue=True, level="INFO")
        logger.add(sys.stdout, colorize=True, enqueue=True, level="INFO")
        logger.info(f"Testing ShapeEstimator at {dirpath}")

        shape_data = ShapeDataset()
        shape_data.load("totalcapture")
        shape_data.preprocessing()

        predictors = {
            0: ShapeEstimator.load(os.path.join(dirpath, "model_0")),
            1: ShapeEstimator.load(os.path.join(dirpath, "model_1")),
        }

        preds, evals = dict(), dict()
        for gender in [0, 1]:
            logger.info(f"Testing ShapeEstimator with: gender {gender}")
            H = shape_data.get_meas_by_gender(gender)[:, 0].reshape(-1, 1)
            H -= torch.tensor(list(shape_data.mean_meas[gender].values()))[0]
            W = shape_data.get_meas_by_gender(gender)[:, 1].reshape(-1, 1)
            W -= torch.tensor(list(shape_data.mean_meas[gender].values()))[1]
            W /= 100
            D = (
                shape_data.get_dists_by_gender(gender)[:, config.masks.D_idxs]
                - shape_data.mean_dists[gender][config.masks.D_idxs]
            )
            HWD = {"H": H, "W": W, "D": D}
            X_test = torch.cat([HWD[_] for _ in cfg.shape_estimator.input_attrs], dim=-1)
            test_data = pd.DataFrame(X_test)
            y_pred = torch.tensor(predictors[gender].predict(test_data).to_numpy())
            y = shape_data.get_betas_by_gender(gender)
            preds[gender] = y_pred
            evals[gender] = ShapeEvaluator.eval(y_pred, y, gender)

        torch.save(preds, os.path.join(dirpath, "pred.pt"))
        torch.save(evals, os.path.join(dirpath, "eval.pt"))
        table = PrettyTable(
            field_names=["Gender", "E_mesh", "E_height", "E_weight", "E_cir", "E_dist"]
        )
        table.add_row(
            [
                "Female",
                f"{evals[0]['mesh_error'].mean().item():.4f}",
                f"{evals[0]['meas_error'][0].item():.4f}",
                f"{evals[0]['meas_error'][1].item():.4f}",
                f"{evals[0]['meas_error'][2:8].mean().item():.4f}",
                f"{evals[0]['D_error'].mean().item():.4f}",
            ]
        )
        table.add_row(
            [
                "Male",
                f"{evals[1]['mesh_error'].mean().item():.4f}",
                f"{evals[1]['meas_error'][0].item():.4f}",
                f"{evals[1]['meas_error'][1].item():.4f}",
                f"{evals[1]['meas_error'][2:8].mean().item():.4f}",
                f"{evals[1]['D_error'].mean().item():.4f}",
            ]
        )
        logger.info(f"Evaluation results:\n{table}")


class PoseEstimatorTester:
    @classmethod
    def predict_shape(cls, betas, gender, meas, shape_estimator_name):
        shape_data = ShapeDataset()
        smpl_model = get_smpl_model(gender)
        virtual_vertices = smpl_model.get_zero_pose_joint_and_vertex(betas)[1][
            config.masks.amass_vertices
        ]
        HW = meas[:2] - shape_data.get_mean_meas_by_gender(gender, to_tensor=True)[:2]
        HW[1] /= 100
        D = (
            virtual_vertices[config.masks.y_idxs]
            - virtual_vertices[config.masks.x_idxs]
        ).norm(dim=-1)[config.masks.D_idxs] - shape_data.get_mean_dists_by_gender(
            gender
        )[
            config.masks.D_idxs
        ]
        X = torch.cat([HW, D], dim=-1).unsqueeze(0)

        predictor = ShapeEstimator.load(
            os.path.join(
                config.paths.base_dir, "models", shape_estimator_name, f"model_{gender}"
            )
        )
        return torch.tensor(predictor.predict(pd.DataFrame(X)).to_numpy()).squeeze()

    @classmethod
    def test(cls, dirpath, shape_estimator_name, checkpoint=None):
        # Check if the model exists
        if not os.path.exists(dirpath):
            raise FileNotFoundError(f"Path {dirpath} not found")

        # Load parameters
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(osp.join(dirpath, "config.yaml"))

        logger.remove()
        logger.add(osp.join(cfg.dirs.pose_estimator, "run.log"), enqueue=True, level="INFO")
        logger.add(sys.stdout, colorize=True, enqueue=True, level="INFO")
        logger.info(f"Testing PoseEstimator at {dirpath}")

        logger.info("Loading model")
        # Load model
        pose_model = PoseEstimator(
            input_size=cfg.pose_estimator.input_size,
            input_hidden_size=cfg.pose_estimator.input_hidden_size,
            output_size=cfg.pose_estimator.output_size,
            hidden_size=cfg.pose_estimator.hidden_size,
            num_layers=cfg.pose_estimator.num_layers,
            dropout=cfg.pose_estimator.dropout,
        ).cuda()
        if checkpoint is not None:
            pose_model.load_state_dict(torch.load(osp.join(cfg.dirs.checkpoints, f"{checkpoint:04d}.pt")))
        else:
            pose_model.load_state_dict(torch.load(osp.join(cfg.dirs.pose_estimator, "model.pt")))
        pose_model.eval()

        for name in cfg.dataset.test_data:
            motions = Dataset(
                load_attrs=cfg.pose_estimator.input_attrs
                + cfg.pose_estimator.input_hidden_attrs
                + cfg.pose_estimator.output_attrs
                + ["betas", "gender", "meas", "accs"]
            )
            logger.info(f"Loading test data: {name}")
            motions.load(name)
            logger.info(f"Preprocessing test data: {name}")
            motions.preprocessing(normalization=cfg.training.normalization)

            # first frame is used for initial hidden state
            logger.info("Preprocessing X, X_h, y for testing")
            X = [_[1:].cuda() for _ in motions.get(cfg.pose_estimator.input_attrs)]
            X_h = [_[0].cuda() for _ in motions.get(cfg.pose_estimator.input_hidden_attrs)]
            y = [_[1:].cpu() for _ in motions.get(cfg.pose_estimator.output_attrs)]
            logger.info(f"Input attrs: {cfg.pose_estimator.input_attrs}")
            logger.info(f"Input hidden attrs: {cfg.pose_estimator.input_hidden_attrs}")
            logger.info(f"Output attrs: {cfg.pose_estimator.output_attrs}")
            evals = {
                "no_shape_vs_no_shape": {
                    metric: [] for metric in PoseEvaluator.metrics
                },  # beta_test = 0, beta_gt = 0
                "no_shape_vs_gt_shape": {
                    metric: [] for metric in PoseEvaluator.metrics
                },  # beta_test = 0, beta_gt = gt
                "pred_shape_vs_gt_shape": {
                    metric: [] for metric in PoseEvaluator.metrics
                },  # beta_test = pred_beta, beta_gt = gt
                "gt_shape_vs_gt_shape": {
                    metric: [] for metric in PoseEvaluator.metrics
                },  # beta_test = gt, beta_gt = gt
            }
            with torch.no_grad():
                for i, (x, x_h, y) in enumerate(zip(X, X_h, y)):
                    logger.info(f"Testing on {name} sequence {i+1}/{len(X)}")
                    y_pred = pose_model(x, x_h)[0].cpu()
                    n_frames = y_pred.shape[0]
                    y_pred = rotation_6d_to_matrix(y_pred.reshape(n_frames, -1, 6)).reshape(n_frames, -1, 3, 3)#.transpose(-1, -2)
                    U, S, Vh = torch.linalg.svd(y_pred)
                    y_pred_ortho = torch.einsum("bnij, bnjk -> bnik", U, Vh)
                    y = axis_angle_to_matrix(y.reshape(n_frames, -1, 3)).reshape(n_frames, -1, 3, 3)
                    y_pred = y.clone()
                    y_pred[:, config.masks.target_joints] = y_pred_ortho

                    smpl_model = get_smpl_model(motions.gender[i])
                    y_pred = smpl_model.get_local_pose_from_global_pose(y_pred)
                    y = smpl_model.get_local_pose_from_global_pose(y)

                    # no shape vs no shape
                    logger.info(f"y_pred shape: {y_pred.shape}")
                    logger.info(f"y shape: {y.shape}")
                    motion_pred = Motion(
                        smpl_model, y_pred, rep="matrix", shape=None, trans=None
                    )
                    motion_gt = Motion(
                        smpl_model, y, rep="matrix", shape=None, trans=None
                    )
                    err = PoseEvaluator.eval(motion_pred, motion_gt)
                    table = PrettyTable()
                    table.field_names = list(err.keys())
                    table.add_row(
                        [f"{err[key][0]:.4f} ± {err[key][1]:.4f}" for key in err.keys()]
                    )
                    logger.info(f"No shape vs No shape:\n{table}")
                    for key in PoseEvaluator.metrics:
                        evals["no_shape_vs_no_shape"][key].append(err[key][2])

                    # no shape vs gt shape
                    shape_gt = motions.betas[i]
                    motion_pred = Motion(
                        smpl_model, y_pred, rep="matrix", shape=None, trans=None
                    )
                    motion_gt = Motion(
                        smpl_model, y, rep="matrix", shape=shape_gt, trans=None
                    )
                    err = PoseEvaluator.eval(motion_pred, motion_gt)
                    table = PrettyTable()
                    table.field_names = list(err.keys())
                    table.add_row(
                        [f"{err[key][0]:.4f} ± {err[key][1]:.4f}" for key in err.keys()]
                    )
                    logger.info(f"No shape vs GT shape:\n{table}")
                    for key in PoseEvaluator.metrics:
                        evals["no_shape_vs_gt_shape"][key].append(err[key][2])

                    # pred shape vs gt shape
                    shape_pred = cls.predict_shape(
                        shape_gt, motions.gender[i], motions.meas[i], shape_estimator_name
                    )
                    motion_pred = Motion(
                        smpl_model, y_pred, rep="matrix", shape=shape_pred, trans=None
                    )
                    motion_gt = Motion(
                        smpl_model, y, rep="matrix", shape=shape_gt, trans=None
                    )
                    err = PoseEvaluator.eval(motion_pred, motion_gt)
                    table = PrettyTable()
                    table.field_names = list(err.keys())
                    table.add_row(
                        [f"{err[key][0]:.4f} ± {err[key][1]:.4f}" for key in err.keys()]
                    )
                    logger.info(f"Pred shape vs GT shape:\n{table}")
                    for key in PoseEvaluator.metrics:
                        evals["pred_shape_vs_gt_shape"][key].append(err[key][2])

                    # gt shape vs gt shape
                    motion_pred = Motion(
                        smpl_model, y_pred, rep="matrix", shape=shape_gt, trans=None
                    )
                    motion_gt = Motion(
                        smpl_model, y, rep="matrix", shape=shape_gt, trans=None
                    )
                    err = PoseEvaluator.eval(motion_pred, motion_gt)
                    table = PrettyTable()
                    table.field_names = list(err.keys())
                    table.add_row(
                        [f"{err[key][0]:.4f} ± {err[key][1]:.4f}" for key in err.keys()]
                    )
                    logger.info(f"GT shape vs GT shape:\n{table}")
                    for key in PoseEvaluator.metrics:
                        evals["gt_shape_vs_gt_shape"][key].append(err[key][2])

            with open(os.path.join(dirpath, f"eval_{name}.json"), "w") as f:
                json.dump(
                    {
                        key: {
                            metric: [
                                torch.cat(evals[key][metric]).mean().item(),
                                torch.cat(evals[key][metric]).std().item(),
                            ]
                            for metric in PoseEvaluator.metrics
                        }
                        for key in evals.keys()
                    },
                    f,
                )
