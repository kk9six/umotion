from data.dataset import Motion
from data.synthesis import synthesize_body_measurements
from utils.smpl import get_smpl_model
from utils.rotation_conversions import matrix_to_axis_angle, rotation_converter, radian_to_degree
import torch
import config


class ShapeEvaluator:

    metrics = ["betas_error", "mesh_error", "meas_error", "D_error"]

    @classmethod
    def eval(cls, betas_pred, betas_gt, gender):
        errs = {}
        smpl_model = get_smpl_model(gender)

        # 1. betas error
        errs["betas_error"] = (betas_pred - betas_gt).pow(2).mean(dim=0).sqrt()

        # 2. mesh error
        vertices_pred = torch.stack(
            [
                smpl_model.get_zero_pose_joint_and_vertex(betas)[1]
                for betas in betas_pred
            ]
        )
        vertices_gt = torch.stack(
            [smpl_model.get_zero_pose_joint_and_vertex(betas)[1] for betas in betas_gt]
        )
        errs["mesh_error"] = torch.norm(vertices_pred - vertices_gt, dim=-1).mean(dim=0)

        # 3. measurements error
        meas_pred = torch.tensor(
            [
                list(synthesize_body_measurements(vertices, smpl_model.faces).values())
                for vertices in vertices_pred
            ]
        )
        meas_gt = torch.tensor(
            [
                list(synthesize_body_measurements(vertices, smpl_model.faces).values())
                for vertices in vertices_gt
            ]
        )
        errs["meas_error"] = (meas_pred - meas_gt).pow(2).mean(dim=0).sqrt()

        # 4. D error
        virtual_sensors_pred = vertices_pred[:, config.masks.amass_vertices]
        D_pred = (
            virtual_sensors_pred[:, config.masks.y_idxs]
            - virtual_sensors_pred[:, config.masks.x_idxs]
        ).norm(dim=-1)[:, config.masks.D_idxs]
        virtual_sensors_gt = vertices_gt[:, config.masks.amass_vertices]
        D_gt = (
            virtual_sensors_gt[:, config.masks.y_idxs]
            - virtual_sensors_gt[:, config.masks.x_idxs]
        ).norm(dim=-1)[:, config.masks.D_idxs]
        errs["D_error"] = (D_pred - D_gt).pow(2).mean(dim=0).sqrt()

        return errs


class PoseEvaluator:

    metrics = [
        "local_joint_angular_error",
        "global_joint_angular_error",
        "global_joint_angular_error_16",
        "sip",
        "joint_positional_error",
        "joint_positional_error_16",
        "mesh_error",
        "D_error"
    ]

    @staticmethod
    def sip(
        motion_pred: Motion,
        motion_gt: Motion,
    ):
        rot_pred = motion_pred.poses_global[:, config.masks.sip_joints].reshape(
            -1, 3, 3
        )
        rot_gt = motion_gt.poses_global[:, config.masks.sip_joints].reshape(-1, 3, 3)
        errs = radian_to_degree(
            torch.norm(
                matrix_to_axis_angle(torch.matmul(rot_pred, rot_gt.transpose(1, 2))),
                dim=-1,
            )
        )
        return errs.mean().item(), errs.std().item(), errs

    @staticmethod
    def global_joint_angular_error(
        motion_pred: Motion,
        motion_gt: Motion,
    ):
        rot_pred = motion_pred.poses_global.reshape(-1, 3, 3)
        rot_gt = motion_gt.poses_global.reshape(-1, 3, 3)
        errs = radian_to_degree(
            torch.norm(
                matrix_to_axis_angle(torch.matmul(rot_pred, rot_gt.transpose(1, 2))),
                dim=-1,
            )
        )
        return errs.mean().item(), errs.std().item(), errs

    @staticmethod
    def global_joint_angular_error_16(
        motion_pred: Motion,
        motion_gt: Motion,
    ):
        """global joint angular error for 16 joints (except for wrist, ankle, foot, hand)"""
        rot_pred = motion_pred.poses_global[:, config.masks.target_joints].reshape(
            -1, 3, 3
        )
        rot_gt = motion_gt.poses_global[:, config.masks.target_joints].reshape(-1, 3, 3)
        errs = radian_to_degree(
            torch.norm(
                matrix_to_axis_angle(torch.matmul(rot_pred, rot_gt.transpose(1, 2))),
                dim=-1,
            )
        )
        return errs.mean().item(), errs.std().item(), errs

    @staticmethod
    def local_joint_angular_error(
        motion_pred: Motion,
        motion_gt: Motion,
    ):
        axis_angle_pred = motion_pred.poses.reshape(-1, 24, 3)[
            :, config.masks.target_joints
        ].reshape(-1, 3)
        axis_angle_gt = motion_gt.poses.reshape(-1, 24, 3)[
            :, config.masks.target_joints
        ].reshape(-1, 3)

        rot_pred = rotation_converter(
            axis_angle_pred, motion_pred.rep, "matrix"
        ).reshape(-1, 3, 3)
        rot_gt = rotation_converter(
            axis_angle_gt, motion_gt.rep, "matrix"
        ).reshape(-1, 3, 3)

        errs = radian_to_degree(
            torch.norm(
                matrix_to_axis_angle(torch.matmul(rot_pred, rot_gt.transpose(1, 2))),
                dim=-1,
            )
        )
        return errs.mean().item(), errs.std().item(), errs

    @staticmethod
    def joint_positional_error(motion_pred: Motion, motion_gt: Motion):
        js_pred = motion_pred.joints
        js_gt = motion_gt.joints
        errs = torch.norm(js_pred - js_gt, dim=-1)
        return errs.mean().item(), errs.std().item(), errs

    @staticmethod
    def joint_positional_error_16(motion_pred: Motion, motion_gt: Motion):
        js_pred = motion_pred.joints[:, config.masks.target_joints]
        js_gt = motion_gt.joints[:, config.masks.target_joints]
        errs = torch.norm(js_pred - js_gt, dim=-1)
        return errs.mean().item(), errs.std().item(), errs

    @staticmethod
    def mesh_error(
        motion_pred: Motion,
        motion_gt: Motion,
    ):
        vertices_pred = motion_pred.vertices
        vertices_gt = motion_gt.vertices
        errs = torch.norm(vertices_pred - vertices_gt, dim=-1)
        return errs.mean().item(), errs.std().item(), errs

    @staticmethod
    def D_error(motion_pred: Motion, motion_gt: Motion):
        vertices_pred = motion_pred.vertices[:, config.masks.amass_vertices]
        vertices_gt = motion_gt.vertices[:, config.masks.amass_vertices]
        D_pred = (
            vertices_pred[:, config.masks.y_idxs]
            - vertices_pred[:, config.masks.x_idxs]
        ).norm(dim=-1)[:, config.masks.D_idxs]
        D_gt = (
            vertices_gt[:, config.masks.y_idxs]
            - vertices_gt[:, config.masks.x_idxs]
        ).norm(dim=-1)[:, config.masks.D_idxs]
        errs = (D_pred - D_gt).abs()
        return errs.mean().item(), errs.std().item(), errs

    @classmethod
    def eval(
        cls,
        motion_pred,
        motion_gt,
    ):
        errs = {}
        for metric in cls.metrics:
            errs[metric] = getattr(cls, metric)(motion_pred, motion_gt)

        return errs
