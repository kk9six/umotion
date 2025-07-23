# This script runs the ukf and neural network-based pose estimation pipeline for TotalCapture.

import os, sys

from data.synthesis import synthesize_distance_noise_std
from utils.smpl import get_smpl_model

current = os.path.realpath(__file__)
base_dir = os.path.dirname(current)
os.chdir(base_dir)
sys.path.append(os.path.join(base_dir, "src"))

from data.dataset import Motion, Dataset
from tracker import UKF, Tracker, NNPropagator, AverageFilter
from network.network import load_pose_estimator
import torch
import config
from utils.utils import normalize_to_root
from tqdm import tqdm

if __name__ == "__main__":
    import random
    import numpy as np

    R3_scale = 1
    nn, cfg = load_pose_estimator("pose_estimator", return_cfg=True, device="cuda")
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    os.environ["PYTHONHASHSEED"] = str(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    tc = Dataset()
    tc.load("totalcapture_test")

    n_seq = 45
    alpha_ukf, beta_ukf, kappa_ukf = 0.2, 1.0, -105
    alpha_nn, beta_nn, kappa_nn = 0.09, 1.0, -93
    threshold_upper, threshold_lower = 0.9, 0.3
    sigma_min, sigma_imu, sigma_max = 0.03, 0.15, 0.25

    os.makedirs("results/totalcapture", exist_ok=True)

    for seq_id in range(n_seq):
        (
            accs,
            oris,
            dists,
            los_proportions,
            poses,
            shape,
            gender,
            poses_global,
            joints,
        ) = (
            tc.accs[seq_id].cuda(),
            tc.oris[seq_id].cuda(),
            tc.dists[seq_id].cuda(),
            tc.los_proportions[seq_id].cuda(),
            tc.poses[seq_id],
            tc.betas[seq_id],
            tc.gender[seq_id],
            tc.poses_global[seq_id],
            tc.joints[seq_id],
        )

        # synthesize distance noise
        dist_noise_std = synthesize_distance_noise_std(
            los_proportions,
            threshold_upper=threshold_upper,
            threshold_lower=threshold_lower,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            sigma_imu=sigma_imu,
        ).cuda()
        dist_noise_var = dist_noise_std.pow(2).cuda()

        nframes = accs.shape[0]
        # control input
        us = accs.clone().cuda()
        # uwb measurement
        z1s = dists.clone().cuda() + torch.normal(0, dist_noise_std).cuda()

        smpl_model = get_smpl_model(gender)
        motion = Motion(
            smpl_model, poses, "axis_angle", shape, trans=None
        )
        smpl_model.cuda()

        # initial state
        virtual_vertices_0 = motion.vertices[0, config.masks.amass_vertices]
        p_xy = ((
            virtual_vertices_0[config.masks.y_idxs]
            - virtual_vertices_0[config.masks.x_idxs]
        ).flatten() + torch.normal(0, torch.ones(45) * 0.03)).cuda()
        v_xy = torch.zeros_like(p_xy).cuda()
        bx = torch.zeros(18).cuda()

        x = torch.cat([p_xy, v_xy, bx], dim=0).cuda()
        P = torch.diag(
            torch.cat(
                [
                    torch.ones(45) * 0.03**2,
                    torch.ones(45) * 0.03**2,
                    torch.ones(18) * 0.4**2,
                ]
            )
        ).cuda()

        x_h_init = torch.cat([poses_global[0], joints[0]]).cuda()

        dt = 1 / 60

        ukf = UKF(alpha=alpha_ukf, beta=beta_ukf, kappa=kappa_ukf, device="cuda")
        sigma_acc = np.sqrt(0.004)
        sigma_ba = np.sqrt(1e-4)
        ukf.initialize(x, P, sigma_acc=sigma_acc, sigma_ba=sigma_ba)

        tracker = Tracker()
        nn_propagator = NNPropagator(
            smpl_model=smpl_model,
            shape=shape.cuda(),
            alpha=alpha_nn,
            beta=beta_nn,
            kappa=kappa_nn,
        )

        uwb_velocity_filter = AverageFilter(window_size=5)
        nn_velocity_filter = AverageFilter(window_size=2)

        with torch.no_grad():
            for i in tqdm(
                range(1, nframes)
            ):  # from 1 as the first frame is used for initialization
                # predict
                u = us[i]
                ukf.predict(u)
                tracker(prior=ukf.x)

                # NN predict
                acc, ori = normalize_to_root(accs[i] - ukf.x[90:], oris[i])
                input_nn = torch.cat(
                    [acc, ori, ukf.x[:45].reshape(15, 3).norm(dim=-1)]
                ).cuda()
                y_pred, y_pred_logstd = (
                    nn(input_nn.unsqueeze(0), x_h=x_h_init)
                    if i == 1
                    else nn(input_nn.unsqueeze(0))
                )
                # update using NN predictions and UWB measurements
                z3, R3 = nn_propagator(y_pred, y_pred_logstd)
                R3 = R3 * R3_scale
                z1, R1 = z1s[i], torch.diag(dist_noise_var[i]).cuda()
                if i == 1:
                    z2, R2 = torch.zeros(15).cuda(), torch.diag(torch.ones(15) * 0.01**2).cuda()
                    z4, R4 = torch.zeros(45).cuda(), torch.diag(torch.ones(45) * 0.01**2).cuda()
                else:
                    z2 = uwb_velocity_filter((z1 - tracker.z1[-1]) / dt).cuda()
                    R2 = torch.diag((R1.diag() + tracker.R1[-1]) / (dt**2)).cuda()
                    z4 = nn_velocity_filter((z3 - tracker.z3[-1]) / dt).cuda()
                    R4 = torch.diag((R3.diag() + tracker.R3[-1]) / (dt**2)).cuda()
                z = torch.cat([z1, z2, z3, z4], dim=0)
                R = torch.block_diag(R1, R2, R3, R4)
                ukf.update(z, R)

                tracker(
                    y_pred=y_pred,
                    y_pred_logstd=y_pred_logstd,
                    posterior=ukf.x,
                    z1=z1,
                    z3=z3,
                    R1=R1.diag(),
                    R3=R3.diag(),
                )
        tracker.serialization()
        tracker.save(os.path.join("results/totalcapture", f"{seq_id}.pt"))
