import torch
import numpy as np
from torch.autograd.functional import jacobian
from functools import partial
from utils.rotation_conversions import rotation_6d_to_axis_angle, axis_angle_to_matrix

import os, sys

# from utils.utils import cholupdate, isPD, nearestPD

current = os.path.realpath(__file__)
base_dir = os.path.dirname(os.path.dirname(current))
os.chdir(base_dir)
sys.path.append(os.path.join(base_dir, "src"))
import config


class UT(object):

    @staticmethod
    def generate_sigma_points(x, P, alpha, beta, kappa):
        """
        Generate sigma points for the unscented Kalman filter given the mean and covariance matrix

        params:
        x: torch.Tensor, shape=(n_variables, ), mean
        P: torch.Tensor, shape=(n_variables, n_variables), covariance matrix
        alpha: float, scaling parameter, 0 <= alpha <= 1
        beta: float, prior knowledge about the distribution, beta = 2 is optimal for Gaussian distribution
        kappa: float, secondary scaling parameter, kappa = 3 - n

        return:
        sigma_points: torch.Tensor, shape=(2 * n + 1, n), sigma points
        Wm: torch.Tensor, shape=(2 * n + 1, ), weights for the mean
        Wc: torch.Tensor, shape=(2 * n + 1, ), weights for the covariance
        """
        n = x.shape[0]
        lambda_ = alpha**2 * (n + kappa) - n
        U = torch.linalg.cholesky((n + lambda_) * P).T  # upper triangular matrix
        sigma_points = torch.concatenate([x.unsqueeze(0), x + U, x - U], dim=0)
        c = 1 / (2 * (n + lambda_))
        Wm = torch.full((2 * n + 1,), c).to(x.device)
        Wc = torch.full((2 * n + 1,), c).to(x.device)

        Wm[0] = lambda_ / (n + lambda_)
        Wc[0] = lambda_ / (n + lambda_) + (1 - alpha**2 + beta)
        return sigma_points, Wm, Wc

    @staticmethod
    def unscented_transform(sigma_points_y, Wm, Wc, cov=None):
        """
        Unscented transform, calculate the transformed mean and covariance

        params:
        sigma_points_y: torch.Tensor, shape=(n_sigma_points, n_vars), sigma points after the propogation
        Wm: torch.Tensor, shape=(n_sigma_points, ), weights for the mean
        Wc: torch.Tensor, shape=(n_sigma_points, ), weights for the covariance
        cov: torch.Tensor, shape=(n_vars, n_vars), noise covariance

        return:
        x: torch.Tensor, shape=(n_vars, ), transformed mean
        P: torch.Tensor, shape=(n_vars, n_vars), transformed covariance
        """
        _, n_vars = sigma_points_y.shape
        x = torch.einsum("i, ij -> j", Wm, sigma_points_y)
        residual = sigma_points_y - x
        if cov is None:
            cov = torch.zeros(n_vars, n_vars).to(sigma_points_y.device)
        P = cov + torch.einsum("i, ij, ik -> jk", Wc, residual, residual)
        return x, P


class NNPropagator(object):
    """
    Propagate the sigma points of poses to the relative positions between selected vertices
    """
    def __init__(
        self,
        smpl_model,
        shape,
        alpha,
        beta,
        kappa,
        return_vertices=False,
        return_distance=False,
    ):
        self.device = smpl_model.shapedirs.device
        self.smpl_model = smpl_model
        self.shape = shape
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.x_idxs = torch.combinations(torch.arange(6), 2)[:, 0].to(self.device)
        self.y_idxs = torch.combinations(torch.arange(6), 2)[:, 1].to(self.device)
        self.zero_pose_glb = torch.eye(3).tile(193, 24, 1, 1).to(self.device)
        self.return_vertices = return_vertices
        self.return_distance = return_distance

    def __call__(self, y_pred, y_pred_logstd):
        y_pred_logstd = torch.clamp(y_pred_logstd, min=np.log(1e-4))
        self.target_joints = config.masks.target_joints
        self.ignored_joints = config.masks.ignored_joints
        sigma_points, Wm, Wc = UT.generate_sigma_points(
            y_pred,
            torch.diag(torch.exp(2 * y_pred_logstd)),
            self.alpha,
            self.beta,
            self.kappa,
        )
        if self.return_vertices:
            mean_vertices, Z = self._hx_pose_to_pxy(
                sigma_points
            )  # relative position
        else:
            Z = self._hx_pose_to_pxy(
                sigma_points
            )  # relative position
        if self.return_distance:
            Z = Z.reshape(-1, 15, 3).norm(dim=-1)  # relative distance
        z, R = UT.unscented_transform(Z, Wm, Wc)
        if self.return_vertices:
            return mean_vertices, z, torch.diag(((R + R.T) / 2).diag()).to(self.device)
        else:
            return z, torch.diag(((R + R.T) / 2).diag()).to(self.device)

    def _propagation(self, sigma_points, smpl_model):
        n = sigma_points.shape[0]
        pose_glb = torch.zeros((n, 24, 3), device=self.device)
        pose_glb[:, self.target_joints] = rotation_6d_to_axis_angle(sigma_points.reshape(n, -1, 6)).reshape(-1, len(self.target_joints), 3)
        pose_glb = axis_angle_to_matrix(pose_glb).reshape(n, -1, 3, 3)
        U, S, Vh = torch.linalg.svd(pose_glb)
        pose_glb = torch.einsum("bnij, bnjk -> bnik", U, Vh)
        pose_local = torch.cat(
            [
                pose_glb[:, 0].unsqueeze(1),
                torch.einsum(
                    "bnij, bnik -> bnjk",
                    pose_glb[:, smpl_model.parents[1:], :],
                    pose_glb[:, 1:],
                ),
            ],
            dim=1,
        )
        pose_local[:, self.ignored_joints] = torch.eye(3, device=self.device)

        pose_blend = torch.tensordot(
            (pose_local[:, 1:] - torch.eye(3).to(self.device)).flatten(1),
            smpl_model.posedirs,
            dims=([1], [2]),
        ).to(self.device)
        J, V_zero_pose = smpl_model.get_zero_pose_joint_and_vertex(self.shape)
        V = V_zero_pose + pose_blend
        Js, Vs = [_.expand(n, -1, -1).to(self.device) for _ in [J, V]]
        Js_local = Js - Js[:, smpl_model.parents]
        T_local = torch.zeros(n, 24, 4, 4, device=self.device)
        T_local[:, :24, :3, :3] = pose_local
        T_local[:, :24, :3, 3] = Js_local
        T_local[:, :24, 3, 3] = 1

        T_global = [T_local[:, 0]]
        for i in range(1, len(smpl_model.parents)):
            T_global.append(torch.bmm(T_global[smpl_model.parents[i]], T_local[:, i]))
        T_global = torch.stack(T_global, dim=1).to(self.device)
        T_global[..., -1:] -= torch.matmul(
            T_global, smpl_model.append_zero(Js).unsqueeze(-1)
        )

        if self.return_vertices:
            T_vertex_mean = torch.tensordot(
                T_global[0].unsqueeze(0),
                smpl_model.skinning_weights,
                dims=([1], [1]),
            ).permute(0, 3, 1, 2).to(self.device)
            vertices_mean = (
                torch.matmul(
                    T_vertex_mean,
                    smpl_model.append_one(Vs[0].unsqueeze(0)).unsqueeze(-1),
                )
                .squeeze(-1)[..., :3]
                .squeeze()
                .to(self.device)
            )

        T_vertex = torch.tensordot(
            T_global,
            smpl_model.skinning_weights[config.masks.amass_vertices],
            dims=([1], [1]),
        ).permute(0, 3, 1, 2).to(self.device)
        vertices = torch.matmul(
            T_vertex,
            smpl_model.append_one(Vs[:, config.masks.amass_vertices]).unsqueeze(
                -1
            ),
        ).squeeze(-1)[..., :3].to(self.device)

        if self.return_vertices:
            return vertices_mean.to(self.device), vertices.to(self.device)
        else:
            return vertices.to(self.device)

    def _hx_pose_to_pxy(self, sigma_points, trans=None):
        """
        Propagate the sigma points
            Estimated global poses to relative positions between selected vertices

        params:
        sigma_points: np.array, shape (2 * n + 1, 16 * 3)
            The sigma points to propagate, where n is the dimension of the state
            The sigma points are 3D axis-angles

        return:
        propagated_sigma_points: np.array, shape (2 * n + 1, m)
            The propagated sigma points
            THe propagated sigma points are 3D vertices
        """
        if self.return_vertices:
            mean_vertices, vertices = self._propagation(sigma_points, self.smpl_model)
            d_xy_2 = vertices[:, self.y_idxs] - vertices[:, self.x_idxs]
            return mean_vertices, d_xy_2.reshape(-1, 3 * 15)
        else:
            vertices = self._propagation(sigma_points, self.smpl_model)
            d_xy_2 = vertices[:, self.y_idxs] - vertices[:, self.x_idxs]
            return d_xy_2.reshape(-1, 3 * 15)


class AverageFilter(object):
    def __init__(self, window_size):
        self.window_size = window_size
        self.buffer = []

    def __call__(self, value):
        self.buffer.append(value)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        try:
            return torch.mean(torch.tensor(self.buffer), dim=0)
        except:
            return torch.mean(torch.vstack(self.buffer), dim=0)


class Tracker(object):
    def __init__(self):
        self.attrs = []

    def __call__(self, **kargs):
        for k, v in kargs.items():
            try:
                getattr(self, k).append(v)
            except:
                self.attrs.append(k)
                setattr(self, k, [v])

    def serialization(self):
        for attr in self.attrs:
            try:
                setattr(self, attr, torch.stack(getattr(self, attr), dim=0).cpu())
            except Exception as e:
                print(e)
                pass

    def save(self, path):
        torch.save({attr: getattr(self, attr) for attr in self.attrs}, path)

class UKF(object):
    def __init__(self, alpha=0.01, beta=2, kappa=0, device="cpu"):
        self.dim_x = 108
        self.dim_u = 18
        self.dim_z = 120
        self.dt = 1 / 60
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self._x_idxs, self._y_idxs = (
            torch.combinations(torch.arange(6), 2)[:, 0],
            torch.combinations(torch.arange(6), 2)[:, 1],
        )
        self.device = device

    def fx(self, x, dt, u):
        u = torch.where(
            (u < self.zero_acc_threshold) & (u > -self.zero_acc_threshold),
            torch.ones_like(u) * 1e-6,
            u,
        )
        p = x.view(-1, self.dim_x)[:, :45].view(-1, 15, 3)
        v = x.view(-1, self.dim_x)[:, 45 : 45 + 45].view(-1, 15, 3)
        b = x.view(-1, self.dim_x)[:, 90:]
        a = torch.sub(u, b).view(-1, 6, 3)
        a_xy = a[:, self._y_idxs] - a[:, self._x_idxs]
        return torch.cat(
            [
                (p + v * dt + 0.5 * a_xy * dt**2).flatten(1),
                (v + a_xy * dt).flatten(1),
                b,
            ],
            dim=1,
        ).squeeze(0)

    def hx(self, Y):
        return torch.hstack(
            [
                Y[:, :45].reshape(-1, 15, 3).norm(dim=-1),
                Y[:, 45:90].reshape(-1, 15, 3).norm(dim=-1),
                Y[:, :45],
                Y[:, 45:90],
            ]
        )

    def Q_u_to_Q_x(self, u, Q_u):
        def wrapper_fx(u, x, dt):
            return self.fx(x.cuda(), dt, u.cuda())

        F = jacobian(partial(wrapper_fx, x=self.x, dt=self.dt), u)
        return F.matmul(Q_u).matmul(F.T)

    def initialize(self, x, P, sigma_acc=None, sigma_ba=None, zero_acc_threshold=None):
        self.x = x
        self.P = P
        if zero_acc_threshold is not None:
            self.zero_acc_threshold = zero_acc_threshold
        else:
            self.zero_acc_threshold = 0.3
        if sigma_acc is None or sigma_ba is None:
            self.Q = torch.zeros(self.dim_x, self.dim_x).to(self.device)
            self.Q[:90, :90] = torch.eye(90).to(self.device) * 0.036641
            self.Q[90:, 90:] = torch.eye(18).to(self.device) * 0.036641
        else:
            self.sigma_acc = sigma_acc
            self.sigma_ba = sigma_ba
            self.Q_u = torch.diag(torch.ones(18) * self.sigma_acc**2).to(self.device)
            self.Q_ba = torch.diag(torch.ones(18) * self.sigma_ba**2).to(self.device)
            self.Q = self.Q_u_to_Q_x(torch.zeros(18).to(self.device), self.Q_u).to(self.device)
            self.Q[90 : 90 + 18, 90 : 90 + 18] += self.dt * self.Q_ba

    def predict(self, u):
        self.sigma_points, self.Wm, self.Wc = UT.generate_sigma_points(
            self.x, self.P, self.alpha, self.beta, self.kappa
        )
        self.Y = self.fx(self.sigma_points, self.dt, u)
        self.x, self.P = UT.unscented_transform(self.Y, self.Wm, self.Wc, self.Q)

    def update(self, z, R):
        Z = self.hx(self.Y)
        mu_z, Pz = UT.unscented_transform(Z, self.Wm, self.Wc, R)
        residual = z - mu_z
        Pxz = torch.einsum("i, ij, ik -> jk", self.Wc, self.Y - self.x, Z - mu_z)
        K = Pxz.matmul(torch.linalg.inv(Pz))
        self.x = self.x + K.matmul(residual)
        self.P = self.P - K.matmul(Pz).matmul(K.T)
        self.P = 0.5 * (self.P + self.P.T)
