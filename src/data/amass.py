# process AMASS dataset
# https://amass.is.tue.mpg.de/
# Download the dataset (SMPL-H) and put it in data/raw/AMASS/

import os, sys, torch, glob
import os.path as osp
import numpy as np
from tqdm import tqdm

current = osp.realpath(__file__)
base_dir = osp.dirname(osp.dirname(osp.dirname(current)))
os.chdir(base_dir)
sys.path.append(osp.join(base_dir, "src"))

import config
from utils.smpl import get_smpl_model
from utils.rotation_conversions import matrix_to_axis_angle, axis_angle_to_matrix
from utils.utils import interpolation, interpolation_axis_angles
from loguru import logger
from data.synthesis import (
    synthesize_acceleration_orientation_distance,
    synthesize_body_measurements,
)


def process_amass(
    in_fname,
    overwrite=False,
):
    out_fname = in_fname.replace(
        config.paths.data_dir_raw, config.paths.data_dir_processed
    ).replace("npz", config.EXT)

    if osp.isfile(out_fname) and overwrite is False:
        logger.warning(
            f"{out_fname} already exists. Skipping. Set overwrite=True to overwrite."
        )
        return torch.load(out_fname)

    data = np.load(in_fname)
    framerate = int(np.round(data["mocap_framerate"]))
    duration = data["poses"].shape[0] / framerate
    if duration < config.MIN_DURATION:  # skip short sequences
        logger.warning(f"Skip {in_fname} due to short duration: {duration}")
        return None
    gender = data["gender"].item()
    poses = interpolation_axis_angles(data["poses"], framerate, config.TARGET_FPS).view(
        -1, 52, 3
    ).cuda()
    poses[:, 23] = poses[:, 37]
    poses = poses[:, :24]
    trans = torch.tensor(
        interpolation(data["trans"], framerate, config.TARGET_FPS).astype(np.float32)
    ).cuda()
    betas = torch.tensor(data["betas"][:10].astype(np.float32)).cuda()

    smpl = get_smpl_model(gender).cuda()
    # align global frame with DIP (rotation root poses and translation)
    # rotation 90deg around x-axis, z-axis to y-axis, y-axis to negative z-axis
    amass_rot_to_dipimu = torch.tensor([[[1, 0, 0], [0, 0, 1], [0, -1, 0.0]]]).cuda()
    trans = amass_rot_to_dipimu.matmul(trans.unsqueeze(-1)).squeeze(-1)
    poses[:, 0] = matrix_to_axis_angle(
        amass_rot_to_dipimu.matmul(axis_angle_to_matrix(poses[:, 0]))
    )

    pose_joint_vertex = smpl(poses, betas)
    accs, oris, dists = synthesize_acceleration_orientation_distance(
        pose_joint_vertex["poses"],
        smpl.get_vertices_with_trans(pose_joint_vertex["vertices"], trans),
    )
    meas = synthesize_body_measurements(smpl(shape=betas)["vertices"].cpu(), smpl.faces.cpu())

    os.makedirs(osp.dirname(out_fname), exist_ok=True)
    saved_data = {
        "poses": poses,
        "trans": trans,
        "betas": betas,
        "gender": gender,
        "joints": pose_joint_vertex["joints"],
        "mocap_framerate": framerate,
        "poses_global": matrix_to_axis_angle(pose_joint_vertex["poses"]),
        "accs": accs,
        "oris": oris,
        "dists": dists,
        "meas": meas,
    }
    torch.save(saved_data, out_fname)
    return saved_data


def amass_fnames(includes: list[str] = [], excludes: list[str] = [], processed=False):
    """
    Get the file names of the motions in the AMASS dataset.
    :param includes: include motions (only load motions with these names)
    :param excludes: exclude motions (ignore motions with these names)
    :param processed: if True, load processed motions (in datasets/processed/AMASS); otherwise, load raw motions (in datasets/raw/AMASS)
    :return: a list of file names
    """
    prefix = osp.join(
        config.paths.data_dir_processed if processed else config.paths.data_dir_raw,
        config.paths.amass_dir,
    )
    pattern = "*_poses.pt" if processed else "*_poses.npz"

    # if includes is empty, load all motions
    if includes == []:
        fnames = glob.glob(osp.join(prefix, f"*/*/{pattern}"))
    else:  # otherwise, load motions with includes
        fnames = []
        for include in includes:
            fnames += glob.glob(osp.join(prefix, f"{include}/*/{pattern}"))

    # exclude motions
    for exclude in excludes:
        exclude_glob = glob.glob(osp.join(prefix, f"{exclude}/*/{pattern}"))
        fnames = list(set(fnames) - set(exclude_glob))
    return fnames


def process_amass_meas(fname, excludes: list[str] = [], includes: list[str] = []):
    model_male = get_smpl_model("male")
    model_female = get_smpl_model("female")

    pbar = tqdm(amass_fnames(includes=includes, excludes=excludes))

    Betas = []
    Meas = []
    Genders = []
    for in_npz_fname in pbar:
        data = np.load(in_npz_fname)
        gender = data["gender"].item()
        model = model_male if gender == "male" else model_female
        betas = torch.tensor(data["betas"][:10].astype(np.float32))
        vertices = model(shape=betas)["vertices"]
        meas = synthesize_body_measurements(vertices, model.faces)
        Betas.append(betas)
        Meas.append(torch.tensor(list(meas.values())))
        Genders.append(1 if gender == "male" else 0)

    Betas = torch.stack(Betas)
    Meas = torch.stack(Meas)
    Genders = torch.tensor(Genders)
    _, unique_indices = np.unique(Betas, return_index=True, axis=0)
    Betas = Betas[unique_indices]
    Meas = Meas[unique_indices]
    Genders = Genders[unique_indices]
    os.makedirs(config.paths.data_dir_body_measurements, exist_ok=True)
    out_fname = osp.join(config.paths.data_dir_body_measurements, fname)
    torch.save(
        {"Betas": Betas, "Meas": Meas, "Genders": Genders},
        out_fname,
    )

    mean_meas_male = torch.tensor(
        list(
            synthesize_body_measurements(
                model_male()["vertices"], model_male.faces
            ).values()
        )
    )
    mean_meas_female = torch.tensor(
        list(
            synthesize_body_measurements(
                model_female()["vertices"], model_female.faces
            ).values()
        )
    )
    torch.save(
        {0: mean_meas_female, 1: mean_meas_male},
        osp.join(
            config.paths.data_dir_body_measurements, config.paths.mean_meas_file
        ),
    )


if __name__ == "__main__":
    logger.info("Processing AMASS dataset...")

    for fname in tqdm(amass_fnames()):
        process_amass(fname, overwrite=True)
    process_amass_meas("amass_exclude_totalcapture.pt", excludes=["TotalCapture"])
    process_amass_meas("amass_include_totalcapture.pt", includes=["TotalCapture"])
