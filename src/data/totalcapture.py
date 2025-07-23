# process TotalCapture dataset
# https://dip.is.tuebingen.mpg.de/index.html
# -> ORIGINAL TotalCapture DATA W/ CORRESPONDING REFERENCE SMPL Poses (wo/ normalization, approx. 250MB)
# put it in datasets/raw/


import os, sys, torch, glob, pickle
import os.path as osp
import numpy as np
from tqdm import tqdm

current = osp.realpath(__file__)
base_dir = osp.dirname(osp.dirname(osp.dirname(current)))
os.chdir(base_dir)
sys.path.append(osp.join(base_dir, "src"))

from loguru import logger
from utils.smpl import get_smpl_model
from utils.rotation_conversions import matrix_to_axis_angle
from data.synthesis import (
    synthesize_acceleration_orientation_distance,
    synthesize_body_measurements,
    synthesize_line_of_sight_proportion,
)
import config
from utils.utils import interpolation


def process_totalcapture(
    in_fname,
    overwrite=False,
):
    """process totalcapture dataset

    read real_acc, real_ori, real_pose from TotalCapture_Real_60FPS
    read betas, trans from AMASS/TotalCapture
    """
    out_fname = in_fname.replace(
        config.paths.data_dir_raw, config.paths.data_dir_processed
    ).replace("pkl", config.EXT)

    if osp.isfile(out_fname) and overwrite is False:
        logger.warning(f"{out_fname} already exists. Set overwrite=True to overwrite.")
        return torch.load(out_fname)

    subject, motion = out_fname.split("/")[-1].split(".")[0].split("_")

    # load real_acc, real_ori, real_pose from DIP TotalCapture_Real_60FPS
    tc_dip = pickle.load(open(in_fname, "rb"), encoding="latin1")
    accs = torch.tensor(tc_dip["acc"].astype(np.float32)).cuda()
    oris = torch.tensor(tc_dip["ori"].astype(np.float32)).cuda()
    poses = torch.tensor(tc_dip["gt"].astype(np.float32)).cuda()
    framerate = 60
    if poses.shape[0] / framerate < config.MIN_DURATION:
        logger.warning(
            f"Skip {in_fname} due to short duration: {poses.shape[0] / framerate}"
        )
        return None

    length = min(accs.shape[0], oris.shape[0], poses.shape[0])
    accs, oris, poses = accs[:length], oris[:length], poses[:length]

    # load betas and genders from AMASS TotalCapture
    tc_amass = np.load(
        glob.glob(
            os.path.join(
                config.paths.data_dir_raw,
                config.paths.amass_dir,
                "TotalCapture",
                f"{subject}/*.npz",
            )
        )[0]
    )
    gender = tc_amass["gender"].item()
    betas = torch.tensor(tc_amass["betas"][:10].astype(np.float32)).cuda()

    tc_amass_specific_fname = os.path.join(
        config.paths.data_dir_raw,
        config.paths.amass_dir,
        "TotalCapture",
        subject,
        motion + "_poses.npz",
    )
    if os.path.exists(tc_amass_specific_fname):
        tc_amass_item = np.load(tc_amass_specific_fname)
        framerate = int(tc_amass_item["mocap_framerate"])
        trans = torch.tensor(
            interpolation(tc_amass_item["trans"], framerate, config.TARGET_FPS).astype(
                np.float32
            )
        ).cuda()
        amass_rot_to_dipimu = torch.tensor([[[1, 0, 0], [0, 0, 1], [0, -1, 0.0]]]).cuda()
        trans = amass_rot_to_dipimu.matmul(trans.unsqueeze(-1)).squeeze(-1)[
            : poses.shape[0]
        ]
        include_trans = True
    else:
        trans = torch.zeros((poses.shape[0], 3)).cuda()
        include_trans = False

    smpl = get_smpl_model(gender).cuda()

    # calculate pose_joint_vertex, dists, meas
    pose_joint_vertex = smpl(poses, betas)
    _, _, dists = synthesize_acceleration_orientation_distance(
        pose_joint_vertex["poses"],
        smpl.get_vertices_with_trans(pose_joint_vertex["vertices"], trans),
    )
    los_proportions = synthesize_line_of_sight_proportion(
        pose_joint_vertex["vertices"], smpl.faces
    )
    meas = synthesize_body_measurements(smpl(shape=betas)["vertices"].cpu(), smpl.faces.cpu())

    os.makedirs(osp.dirname(out_fname), exist_ok=True)
    saved_data = {
        "poses": poses,
        "trans": trans,
        "betas": betas,
        "gender": gender,
        "joints": pose_joint_vertex["joints"],
        "poses_global": matrix_to_axis_angle(pose_joint_vertex["poses"]),
        "accs": accs,
        "oris": oris,
        "dists": dists,
        "los_proportions": los_proportions,
        "meas": meas,
        "include_trans": include_trans,
    }
    torch.save(saved_data, out_fname)
    return saved_data


def totalcapture_fnames(processed=False):
    """return the list of all totalcapture file names
    :param processed: if True, return processed data file names
    :return: a list of file names
    """
    prefix = config.paths.data_dir_processed if processed else config.paths.data_dir_raw
    ext = config.EXT if processed else "pkl"
    return glob.glob(osp.join(prefix, config.paths.totalcapture_dip_dir, f"*.{ext}"))


if __name__ == "__main__":
    logger.info("Processing TotalCapture dataset...")

    for fname in tqdm(totalcapture_fnames()):
        process_totalcapture(fname, overwrite=True)
