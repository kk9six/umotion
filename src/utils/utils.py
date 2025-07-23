import argparse
from collections import defaultdict
from collections.abc import Iterable
import os, pickle, torch, yaml
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import transform


def load_pickle(fname):
    with open(fname, "rb") as f:
        data = pickle.load(f)
    return data


def save_pickle(data, fname):
    with open(fname, "wb") as f:
        pickle.dump(data, f)

def get_intersection_points(p1, p1_vi, p2, p2_vi, vertices, faces):
    """
    Get the intersection points between the line segment p1 and p2 and the mesh

    params:
    - p1: torch.tensor, (3, 1), the starting point of the line segment
    - p1_vi: torch.tensor, (1,), the vertice index of p1
    - p2: torch.tensor, (3, 1), the ending point of the line segment
    - p2_vi: torch.tensor, (1,), the vertice index of p2
    - vertices: torch.tensor, (n, 3), the vertices of the mesh
    - faces: torch.tensor, (n, 3), the faces of the mesh

    (we use p1_vi and p2_vi to filter out the faces that the line segment is on)

    return:
    - intersection_triangles: torch.tensor, (n, 3), the triangles that the intersection points are on
    - intersection_points: torch.tensor, (n, 3), the intersection points
    - entry_exit: torch.tensor, (n, 1), 0: exit, 1: entry
    """
    # get the line from p1 and p2
    k = p2 - p1

    triangles = vertices[faces]
    V1, V2, V3 = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    E1 = V2 - V1
    E2 = V3 - V1
    H = torch.linalg.cross(k.unsqueeze(dim=0), E2)  # (k x e_2)
    A = (E1 * H).sum(dim=1)  # (k x e_2) \cdot e_1
    F = 1.0 / A
    S = p1 - V1
    U = F * (S * H).sum(dim=1)
    Q = torch.linalg.cross(S, E1)  # (s x e1)
    V = F * (Q * k).sum(dim=1)  # (s x e1) \cdot k
    T = F * (E2 * Q).sum(dim=1)  # (s x e1) \cdot e2
    filtered_indices = (
        (U >= 0)
        & (U <= 1)
        & (V >= 0)
        & (V <= 1)
        & (U + V <= 1)
        & (T < 1)
        & (T > 0)
        & torch.logical_not((p1_vi == faces).sum(dim=1))
        & torch.logical_not((p2_vi == faces).sum(dim=1))
    )
    intersection_triangles = triangles[filtered_indices][
        torch.argsort(T[filtered_indices])
    ]
    intersection_points = p1 + k * T[filtered_indices][
        torch.argsort(T[filtered_indices])
    ].reshape(-1, 1)
    normals = torch.linalg.cross(
        intersection_triangles[:, 1] - intersection_triangles[:, 0],
        intersection_triangles[:, 2] - intersection_triangles[:, 0],
        dim=1,
    )
    entry_exit = ((k * normals).sum(dim=1) < 0).int()  # 0: exit, 1: entry

    return intersection_triangles, intersection_points, entry_exit


def normalize_to_root(accs, oris):
    shape = accs.shape[:-1]
    accs = accs.reshape(-1, 6, 3).clone()
    oris = oris.reshape(-1, 6, 3, 3).clone()

    accs[:, :5] = accs[:, :5] - accs[:, 5].unsqueeze(1)
    accs = torch.einsum("nij, nmi->nmj", oris[:, 5], accs).reshape(shape + (18,))

    oris[:, :5] = torch.einsum("nij,nmik->nmjk", oris[:, 5], oris[:, :5])
    oris = oris.reshape(shape + (54,))

    return accs, oris


def plot_losses(losses, val_losses, val_step, num_epochs, fname):
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.8
    plt.rcParams["grid.linestyle"] = "dotted"
    # plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['figure.figsize'] = (4.845, 3.135)
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["mathtext.default"] = "regular"
    plt.plot(np.arange(1, num_epochs + 1), losses, label="Training loss")
    plt.plot(
        np.arange(1, num_epochs + 1, val_step),
        val_losses,
        label="Validation loss",
    )
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(
        fname,
        format="pdf",
        dpi=300,
    )
    plt.close()


def flatten(lis):
    def _flatten():
        for item in lis:
            if isinstance(item, Iterable) and not isinstance(item, str):
                for x in flatten(item):
                    yield x
            else:
                yield item

    return list(_flatten())


def interpolation_axis_angles(X, X_framerate, target_framerate):
    """interpolate axis angle data to target framerate

    :param np.ndarray X: input data in shape [n_samples, n_joints * 3]
    :param int X_framerate: framerate of the input data
    :param int target_framerate: target framerate

    :return np.ndarray: data in shape [n_samples, n_joints * 3] with target framerate

    Notes:
    if the target timestamp exceeds the maximum timestamp of the input data, the last pose is repeated.
    if the target timestamp is less than the minimum timestamp of the input data, the first pose is repeated.
    """
    if (X_framerate % target_framerate) == 0:
        step = int(X_framerate // target_framerate)
        return torch.tensor(X[::step].astype(np.float32))
    n_samples_orig, n_joints = X.shape[0], X.shape[1] // 3
    temp_poses = X.copy().reshape(n_samples_orig, n_joints, 3)
    duration = n_samples_orig / X_framerate  # in second
    timestamp_orig = np.arange(0, duration, 1.0 / X_framerate)[:n_samples_orig]
    timestamp_targ = np.arange(0, duration, 1.0 / target_framerate)

    count_exceed_max_timestamp = (timestamp_targ > timestamp_orig[-1]).sum()
    count_less_min_timestamp = (timestamp_targ < timestamp_orig[0]).sum()
    if count_exceed_max_timestamp:
        timestamp_targ = timestamp_targ[:-count_exceed_max_timestamp]
    timestamp_targ = timestamp_targ[count_less_min_timestamp:]

    interp_poses = torch.cat(
        [
            torch.tensor(
                transform.Slerp(
                    timestamp_orig,
                    transform.Rotation.from_rotvec(temp_poses[:, i]),
                )(timestamp_targ).as_rotvec()
            ).float()
            for i in range(n_joints)
        ],
        dim=1,
    )
    if count_exceed_max_timestamp:
        interp_poses = torch.cat(
            [interp_poses, interp_poses[-1:].repeat(count_exceed_max_timestamp, 1)]
        )
    if count_less_min_timestamp:
        interp_poses = torch.cat(
            [interp_poses[0].repeat(count_less_min_timestamp, 1), interp_poses]
        )
    return interp_poses


def interpolation(X, X_framerate, target_framerate):
    """interpolate input data to target framerate

    :param np.ndarray X: input data in shape [n_samples, n_features]
    :param int X_framerate: framerate of the input data
    :param int target_framerate: target framerate

    :return np.ndarray: data in shape [n_samples, n_features] with target framerate

    (2024: use `np.interp` instead of `scipy.interpolate.interp1d`)
    """
    if (X_framerate % target_framerate) == 0:
        step = int(X_framerate // target_framerate)
        return X[::step]
    else:
        duration = len(X) / X_framerate  # in second
        timestamp_ori = np.arange(0, duration, 1.0 / X_framerate)[: X.shape[0]]
        timestamp_targ = np.arange(0, duration, 1.0 / target_framerate)
        n_features = X.shape[1]
        return np.array(
            [
                np.interp(timestamp_targ, timestamp_ori, X[:, i])
                for i in range(n_features)
            ]
        ).T


def nearestPD(A):
    """Find the nearest positive-definite matrix to input
    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].
    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = torch.linalg.svd(B)

    H = torch.matmul(V.T, torch.matmul(torch.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(torch.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = torch.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = torch.min(torch.linalg.eigvals(A3).real)
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3

def isPD(B):
    try:
        _ = torch.linalg.cholesky(B)
        return True
    except torch.linalg.LinAlgError:
        return False

