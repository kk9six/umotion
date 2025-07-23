import os, sys, pickle, torch, glob
import os.path as osp
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


current = osp.realpath(__file__)
base_dir = osp.dirname(osp.dirname(osp.dirname(current)))
os.chdir(base_dir)
sys.path.append(osp.join(base_dir, "src"))

from utils.smpl import get_smpl_model
from data.synthesis import (
    synthesize_acceleration_orientation_distance,
    synthesize_body_measurements,
    synthesize_line_of_sight_proportion,
)
from utils.rotation_conversions import matrix_to_axis_angle
import config

from loguru import logger


def split_list_by_continuity(lst):
    """Split a list of integers into sublists where each sublist contains continuous integers.

    :param lst: list of integers
    :return: list of sublists
    """
    if len(lst) == 0:
        return []

    result = []
    sublist = [lst[0]]

    for i in range(1, len(lst)):
        if lst[i] == lst[i - 1] + 1:
            sublist.append(lst[i])
        else:
            result.append(sublist)
            sublist = [lst[i]]

    result.append(sublist)
    return result


def process_dipimu(
    in_fname,
    overwrite=False,
):
    # load data
    out_fname = in_fname.replace(
        config.paths.data_dir_raw, config.paths.data_dir_processed
    ).replace("pkl", config.EXT)

    if osp.isfile(out_fname) and overwrite is False:
        logger.warning(f"{out_fname} already exists. Set overwrite=True to overwrite.")
        return torch.load(out_fname)

    dip = pickle.load(open(in_fname, "rb"), encoding="latin1")
    subject_id = in_fname.split('/')[-2]
    metadata = torch.load(osp.join(config.paths.data_dir_raw, "metadata.pt"))[subject_id]
    gender = metadata["gender"]
    betas = metadata["betas"].cuda()

    framerate = 60
    poses = dip["gt"][1:-1]
    if poses.shape[0] / framerate < config.MIN_DURATION:
        logger.warning(
            f"Skip {in_fname} due to short duration: {poses.shape[0] / framerate}"
        )
        return None
    accs = dip["imu_acc"][1:-1, config.masks.dip_imus]  # (n, 6)
    oris = dip["imu_ori"][1:-1, config.masks.dip_imus]  # (n, 3, 3)

    # interpolate nans
    for i in range(6):
        # find nan indices, time step
        nan_indices = np.unique(np.argwhere(np.isnan(accs[:, i]))[:, 0])

        offset = 0
        for sublst in split_list_by_continuity(nan_indices):
            idx_start = sublst[0] - offset - 1  # the last valid index
            idx_end = sublst[-1] - offset + 1  # the next valid index
            if idx_start < 0:  # if nan is at the beginning, discard
                accs = accs[idx_end:]
                oris = oris[idx_end:]
                poses = poses[idx_end:]
                offset = idx_end
                continue
            if idx_end >= len(accs):  # if nan is at the end, discard
                accs = accs[:idx_start]
                oris = oris[:idx_start]
                poses = poses[:idx_start]
                continue

            # linear interpolate the nan values for acceleration
            accs[idx_start+1:idx_end, i] = np.linspace(
                accs[idx_start, i], accs[idx_end, i], len(sublst) + 2
            )[1:-1]
            # slerp interpolate the nan values for orientation
            oris[idx_start+1:idx_end, i] = Slerp(
                [0, 1], R.from_matrix([oris[idx_start, i], oris[idx_end, i]])
            )(np.linspace(0, 1, len(sublst) + 2)[1:-1]).as_matrix()

    accs = torch.tensor(accs.astype(np.float32)).cuda()
    oris = torch.tensor(oris.astype(np.float32)).cuda()
    poses = torch.tensor(poses.astype(np.float32)).cuda()
    trans = torch.zeros(poses.shape[0], 3).cuda()
    include_trans = False

    smpl = get_smpl_model(gender).cuda()
    meas = synthesize_body_measurements(smpl(shape=betas)["vertices"].cpu(), smpl.faces.cpu())

    pose_joint_vertex = smpl(poses, betas)
    _, _, dists = synthesize_acceleration_orientation_distance(
        pose_joint_vertex["poses"],
        smpl.get_vertices_with_trans(pose_joint_vertex["vertices"], trans),
    )
    # los_proportions = synthesize_line_of_sight_proportion(
    #     pose_joint_vertex["vertices"], smpl.faces
    # )

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
        # "los_proportions": los_proportions,
        "meas": meas,
        "include_trans": include_trans,
    }
    torch.save(saved_data, out_fname)
    return saved_data


def dipimu_fnames(processed=False, subject_ids: list = None):
    subject_ids = list(range(1, 11)) if subject_ids is None else subject_ids
    paths = []
    prefix = config.paths.data_dir_processed if processed else config.paths.data_dir_raw
    ext = config.EXT if processed else "pkl"
    for id in subject_ids:
        paths.extend(
            glob.glob(
                osp.join(
                    prefix,
                    config.paths.dipimu_dir,
                    f"*/s_{id:02d}/*.{ext}",
                )
            )
        )
    # s_10/05.pkl has too much nan (third IMU, 13 consecutive nan), discard, see also https://github.com/Xinyu-Yi/PIP/blob/0e8df58d3b67ac5922626d72e9d6b74a068df108/preprocess.py#L123
    """
    # print the maximum number of nan in each file
    import pickle
    from data.dipimu import split_list_by_continuity
    from data.dipimu import dipimu_fnames

    nan_max_number = []
    for fname in dipimu_fnames():
        data = pickle.load(open(fname, "rb"), encoding="latin1")
        accs = data["imu_acc"][1:-1, config.masks.dip_imus]
        oris = data["imu_ori"][1:-1, config.masks.dip_imus]
        for i in range(6):
            nan_indices = np.unique(np.argwhere(np.isnan(accs[:, i]))[:, 0])
            count = [len(_) for _ in split_list_by_continuity(nan_indices)]
        nan_max_number.append((fname, i, 0 if count == [] else max(count)))
    """
    paths = list(filter(lambda x: "s_10/05.pkl" not in x, paths))
    return paths


if __name__ == "__main__":
    logger.info("Processing DIP_IMU dataset...")

    for fname in tqdm(dipimu_fnames()):
        process_dipimu(fname, overwrite=True)
