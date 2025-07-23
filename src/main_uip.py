import os, sys

current = os.path.realpath(__file__)
base_dir = os.path.dirname(os.path.dirname(current))
os.chdir(base_dir)
sys.path.append(os.path.join(base_dir, "src"))
import numpy as np
from utils.smpl import get_smpl_model

from data.dataset import Motion
from tracker import UKF, Tracker, NNPropagator, AverageFilter
from network.network import load_pose_estimator
import torch
import config
from utils.rotation_conversions import matrix_to_axis_angle, rotation_6d_to_matrix

from utils.utils import normalize_to_root
from network.eval import PoseEvaluator
from tqdm import tqdm
from loguru import logger

def do_eval(seq_idx, data):
    R3_scale = 10
    gender = "neutral"
    smpl_model = get_smpl_model(gender)
    nn, params = load_pose_estimator("pose_estimator_noise", True)
    nn.eval()
    nn.cuda()
    accs = data["accs"]
    oris = data["oris"]
    dists = data["dists"]
    dists_gt = data["dists_gt"]
    betas = data["betas"]
    poses = data["poses"]
    motion_gt = Motion(smpl_model, poses.cpu(), "axis_angle", betas.cpu(), trans=None)
    smpl_model.cuda()
    dist_0 = dists_gt[0]
    pxy_0 = (
        motion_gt.vertices[:, config.masks.amass_vertices][:, config.masks.y_idxs]
        - motion_gt.vertices[:, config.masks.amass_vertices][:, config.masks.x_idxs]
    )[0].cuda()

    X_h_0 = matrix_to_axis_angle(motion_gt.poses_global[0]).flatten().cuda()
    X_h_1 = motion_gt.joints[0].flatten().cuda()
    X_h = torch.cat([X_h_0, X_h_1])

    dt = 1 / 60
    ukf = UKF(alpha=0.2, beta=1.0, kappa=-105, device="cuda")  # 3 - 108
    P = torch.diag(
        torch.cat(
            [
                torch.ones(45) * (0.01**2),  # position
                torch.ones(45) * (0.02**2),  # velocity
                torch.ones(18) * (0.02**2),  # bias
            ]
        )
    ).cuda()
    x = torch.cat([pxy_0.flatten(), torch.zeros(45).cuda(), torch.zeros(18).cuda()])
    ukf.initialize(x, P)
    tracker = Tracker()
    nn_propagator = NNPropagator(
        smpl_model=smpl_model,
        shape=betas,
        alpha=0.09,
        beta=1.0,
        kappa=-93,  # 3 - 96
    )
    uwb_velocity_filter = AverageFilter(window_size=5)
    nn_velocity_filter = AverageFilter(window_size=2)
    R_dists = (dists - dists_gt).abs()
    # z1: relative distance (relative position norm)
    # z2: relative velocity norm
    # z3: relative position
    # z4: relative velocity
    for i, (acc, ori, dist) in tqdm(
        enumerate(zip(accs, oris, dists)), total=len(accs)
    ):
        u = acc.flatten()
        ukf.predict(u)
        tracker(prior=ukf.x)
        acc_norm, ori_norm = normalize_to_root(
            acc.flatten() - ukf.x[90:], ori.flatten()
        )
        input_nn = torch.cat(
            [acc_norm, ori_norm, ukf.x[:45].reshape(15, 3).norm(dim=-1)]
        ).cuda()
        with torch.no_grad():
            y_pred, y_pred_logstd = (
                nn(input_nn.unsqueeze(0), x_h=X_h.cuda())
                if i == 0
                else nn(input_nn.unsqueeze(0))
            )
        tracker(y_pred=y_pred, y_pred_logstd=y_pred_logstd)
        z3, R3 = nn_propagator(y_pred, y_pred_logstd)
        z1 = dist.clone()
        R1 = torch.diag((R_dists[i]).pow(2))
        R3 = R3 * R3_scale
        if i == 0:
            z2 = torch.zeros(15).cuda()
            R2 = torch.diag(torch.ones(15) * 0.01**2).cuda()
            z4 = torch.zeros(45).cuda()
            R4 = torch.diag(torch.ones(45) * 0.01**2).cuda()
        else:
            z2 = uwb_velocity_filter((z1 - tracker.z1[-1]) / dt)
            R2 = torch.diag((R1.diag() + tracker.R1[-1].diag()) / (dt**2))
            z4 = nn_velocity_filter((z3 - tracker.z3[-1]) / dt)
            R4 = torch.diag((R3.diag() + tracker.R3[-1].diag()) / (dt**2))
        z = torch.cat([z1, z2, z3, z4], dim=0)
        R = torch.block_diag(R1, R2, R3, R4)
        ukf.update(z, R)
        tracker(
            posterior=ukf.x,
            z1=z1,
            R1=R1,
            z3=z3,
            R3=R3,
        )
    tracker.serialization()
    os.makedirs("results/uip/", exist_ok=True)
    tracker.save(f"results/uip/{seq_idx}.pt")

    y_preds = tracker.y_pred.detach()
    target_joints = config.masks.target_joints
    ignored_joints = config.masks.ignored_joints
    poses_glb = torch.zeros((y_preds.shape[0], 24, 3, 3)).cuda()
    poses_glb[:, target_joints] = rotation_6d_to_matrix(
        y_preds.view(-1, len(target_joints), 6).cuda()
    ).view(-1, len(target_joints), 3, 3)
    U, _, Vh = torch.linalg.svd(poses_glb)
    poses_glb = torch.einsum("bnij, bnjk -> bnik", U, Vh)

    poses_local = smpl_model.get_local_pose_from_global_pose(poses_glb)
    poses_local = matrix_to_axis_angle(poses_local)
    poses_local[:, ignored_joints] = poses[:, ignored_joints]

    motion_pred = Motion(
        smpl_model.cpu(), poses_local.cpu(), "axis_angle", shape=betas.cpu(), trans=None
    )
    err = PoseEvaluator.eval(motion_pred, motion_gt)
    logger.info(
        f"{seq_idx}_{gender}: {err['sip'][0]:.4f}, {err['joint_positional_error'][0]:.4f}"
    )
    return err

if __name__ == "__main__":
    uip_dataset = torch.load(
        "datasets/processed/uip/test.pt", weights_only=False, map_location="cuda"
    )
    sip_errs = []
    jpe_errs = []
    evals = {}
    for seq_idx, data in enumerate(uip_dataset):
        logger.info(f"Evaluating sequence {seq_idx}...")
        err = do_eval(seq_idx, data)
        sip_errs.append(err["sip"][0])
        jpe_errs.append(err["joint_positional_error"][0])
        evals[seq_idx] = err
        torch.cuda.empty_cache()

    logger.info(f"SIP: {np.mean(sip_errs):.4f}")
    logger.info(f"JPE: {np.mean(jpe_errs):.4f}")
    torch.save(evals, "results/uip/evals_neutral.pt")
