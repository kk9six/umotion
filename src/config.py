import os
import os.path as osp
import torch
from enum import Enum, auto
from typing import Literal

TARGET_FPS = 60 # target fps
MIN_DURATION = 0.54 # minimum duration of a motion, in seconds, ignore motions shorter than this
EXT = "pt" # extension of processed files

datasets = Literal["amass", "dipimu", "totalcapture_real", "totalcapture_amass"]


# fmt: off
class BodyMeasurement(Enum):
    HEIGHT            = 0 # height
    WEIGHT            = auto()  # mass in kg
    CHEST             = auto()  # circumference
    WAIST             = auto()  # circumference
    HIP               = auto()  # circumference
    WRIST             = auto()  # circumference
    KNEE              = auto()  # circumference
    HEAD              = auto()  # circumference
    WRIST_to_WRIST    = auto()  # distance
    HEAD_to_WAIST     = auto()  # distance
    KNEE_to_WAIST     = auto()  # distance
    WRIST_to_WAIST    = auto()  # distance
    WRIST_to_KNEE     = auto()  # distance
    WRIST_to_HEAD     = auto()  # distance
    HEAD_to_KNEE      = auto()  # distance


class BodyMeasurementUnit:
    HEIGHT            = "m"
    WEIGHT            = "kg"
    CHEST             = "m"
    WAIST             = "m"
    HIP               = "m"
    WRIST             = "m"
    KNEE              = "m"
    HEAD              = "m"
    WRIST_to_WRIST    = "m"
    HEAD_to_WAIST     = "m"
    KNEE_to_WAIST     = "m"
    WRIST_to_WAIST    = "m"
    WRIST_to_KNEE     = "m"
    WRIST_to_HEAD     = "m"
    HEAD_to_KNEE      = "m"


class paths:
    # change this path first, all other paths are relative to this path
    base_dir                   = "/home/kksix/Workspaces/umotion"

    data_dir_raw               = osp.join(base_dir, "datasets", "raw")
    data_dir_processed         = osp.join(base_dir, "datasets", "processed")
    data_dir_body_measurements = osp.join(base_dir, "datasets", "body_measurements")

    mean_meas_file             = "mean_meas.pt"

    amass_dir                  = "AMASS"
    dipimu_dir                 = "DIP_IMU_and_Others"
    uip_dir                    = "uip"
    totalcapture_dip_dir       = "TotalCapture_Real_60FPS"  # TotalCapture from DIP-IMU
    totalcapture_raw_dir       = "TotalCapture"  # TotalCapture from TotalCapture
    totalcapture_vicon_dir     = "TotalCapture_Vicon"  # Vicon gt from TotalCapture

    # SMPL_FILE                 = osp.join(base_dir, "models", "smpl", "models", "basicModel_%s_lbs_10_207_0_v1.0.0.pkl")
    SMPL_FILE                 = osp.join(base_dir, "models", "smpl", "basicmodel_%s_lbs_10_207_0_v1.1.0.pkl")
    smpl_model_file_male      = SMPL_FILE % "m"
    smpl_model_file_female    = SMPL_FILE % "f"
    smpl_model_file_neutral   = SMPL_FILE % "neutral"
    # smpl_model_file_neutral   = osp.join(base_dir, "models", "smpl", "SMPL_NEUTRAL.pkl")


class dim:
    # SMPL model
    pose   = 72
    beta   = 10
    shape  = 10
    vertex = 6890
    mesh   = 13776


class masks:

    amass_virtual_sensors     = [
        "left wrist",  # 0
        "right wrist",  # 1
        "left knee",  # 2
        "right knee",  # 3
        "head top",  # 4
        "back root",  # 5
    ]
    dipimu_vertices           = [1961, 5424, 1176, 4662, 411, 3021]  # related work (dipimu)
    amass_vertices            = [1961, 5424, 1176, 4662, 385, 3021]  # note: 335, 3163, 385 (back)
    D_idxs                    = [0, 6, 7, 8, 9, 13, 14]

    """
    0 : 'root (pelvis)',
    7 : 'left ankle',
    8 : 'right ankle',
    10: 'left foot',
    11: 'right foot',
    20: 'left wrist',
    21: 'right wrist',
    22: 'left_hand',
    23: 'right_hand',
    """
    smpl_skeleton             = [
        "pelvis",
        "left_hip",
        "right_hip",
        "spine1",
        "left_knee",
        "right_knee",
        "spine2",
        "left_ankle",
        "right_ankle",
        "spine3",
        "left_foot",
        "right_foot",
        "neck",
        "left_collar",
        "right_collar",
        "head",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hand",
        "right_hand",
    ]

    sip_joints                = [1, 2, 16, 17]

    smpl_root_joint           = [0]
    smpl_ankle_joints         = [7, 8]
    smpl_foot_joints          = [10, 11]
    smpl_hand_joints          = [22, 23]
    smpl_wrist_joints         = [20, 21]

    # ignore the ankle, foot, hand joints
    target_joints             = [0, 1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]
    ignored_joints            = [7, 8, 10, 11, 20, 21, 22, 23]

    amass_joints              = [18, 19, 4, 5, 15, 0]

    # The order of IMUs in DIPIMU is: [head, spine2, belly, lchest, rchest, lshoulder, rshoulder, lelbow, relbow, lhip, rhip, lknee, rknee, lwrist, lwrist, lankle, rankle].
    dip_imus                  = [7, 8, 11, 12, 0, 2]  # [lelbow, relbow, lknee, rknee, head, belly]

    head_top_vertex           = 411
    head_top_mask             = [0.0, 1.0, 0.0]
    head_top_face             = 435

    heel_left_vertex          = 3466
    heel_left_face            = 5975
    heel_left_mask            = [0.0, 0.0, 1.0]

    finger_tip_left_vertex    = 2445
    finger_tip_left_face      = 3259
    finger_tip_left_mask      = [0.0, 0.0, 1.0]

    finger_tip_right_vertex   = 5905
    finger_tip_right_face     = 10147
    finger_tip_right_mask     = [0.0, 0.0, 1.0]

    # elbow_left_vertex       = 1658 # SHAPY
    elbow_left_vertex         = 1655
    elbow_left_face           = 1867
    elbow_left_mask           = [1.0, 0.0, 0.0]

    elbow_right_vertex        = 5129
    elbow_right_face          = 8756
    elbow_right_mask          = [0.0, 0.0, 1.0]

    wrist_left_vertex_back    = 2099  # SHAPY
    wrist_left_vertex         = 1961
    wrist_left_face           = 2603
    wrist_left_mask           = [0.0, 0.0, 1.0]

    wrist_left_front_vertex   = 1923
    wrist_left_back_vertex    = 1941
    wrist_left_down_vertex    = 1934

    # wrist_right_vertex      = 5559 # SHAPY
    wrist_right_vertex        = 5424
    wrist_right_face          = 9491
    wrist_right_mask          = [0.0, 1.0, 0.0]

    belly_button_vertex       = 3501
    belly_button_face         = 6833
    belly_button_mask         = [0.0, 0.0, 1.0]

    belly_button_back_vertex  = 3022
    belly_button_left_vertex  = 677
    belly_button_right_vertex = 4165

    nipple_left_vertex        = 3042
    nipple_left_face          = 4997
    nipple_left_mask          = [0.0, 0.0, 1.0]

    nipple_right_vertex       = 6489
    nipple_right_face         = 11885
    nipple_right_mask         = [0.0, 0.0, 1.0]

    crotch_vertex             = 1210
    crotch_face               = 1341
    crotch_mask               = [0.0, 1.0, 0.0]

    knee_left_vertex          = 1175
    knee_left_back_vertex     = 1182
    knee_left_left_vertex     = 1085
    knee_left_right_vertex    = 1078

    knee_right_vertex         = 4662

    head_middle_vertex        = 335
    head_back_vertex          = 385
    head_left_vertex          = 169
    head_right_vertex         = 3679

    shoulder_left_vertex      = 2893
    shoulder_left_face        = 4572
    shoulder_left_mask        = [0.0, 0.0, 1.0]

    shoulder_right_vertex     = 5291
    shoulder_right_face       = 9117
    shoulder_right_mask       = [1.0, 0.0, 0.0]

    # pxy                     = p[y] - p[x]
    x_idxs                    = [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4]
    y_idxs                    = [1, 2, 3, 4, 5, 2, 3, 4, 5, 3, 4, 5, 4, 5, 5]

    target_joints             = [0, 1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]
    ignored_joints            = [7, 8, 10, 11, 20, 21, 22, 23]

class Colors:
    r = (255, 0, 0)
    g = (47, 109, 28)
    b = (0, 0, 255)


if __name__ == "__main__":
    print(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
