from typing import Literal
import torch
import trimesh
import numpy as np

from utils.rotation_conversions import axis_angle_to_matrix

# -------------- fix chumpy error
np.bool = np.bool_
np.int = np.int_
np.float = np.float64
np.complex = np.complex128
np.object = np.object_
np.unicode = np.str_
np.str = np.str_
import inspect
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec
# --------------
import pickle
import config


"""
name: J_regressor_prior     type: <class 'scipy.sparse.csc.csc_matrix'>     size: (24, 6890)
name: pose                  type: <class 'chumpy.ch.Ch'>                    size: (72,)
name: f                     type: <type 'numpy.ndarray'>                    size: (13776, 3)
name: J_regressor           type: <class 'scipy.sparse.csc.csc_matrix'>     size: (24, 6890)
name: betas                 type: <class 'chumpy.ch.Ch'>                    size: (10,)
name: kintree_table         type: <type 'numpy.ndarray'>                    size: (2, 24)
name: J                     type: <class 'chumpy.reordering.transpose'>     size: (24, 3)
name: v_shaped              type: <class 'chumpy.ch_ops.add'>               size: (6890, 3)
name: weights_prior         type: <type 'numpy.ndarray'>                    size: (6890, 24)
name: trans                 type: <class 'chumpy.ch.Ch'>                    size: (3,)
name: v_posed               type: <class 'chumpy.ch_ops.add'>               size: (6890, 3)
name: weights               type: <class 'chumpy.ch.Ch'>                    size: (6890, 24)
name: vert_sym_idxs         type: <type 'numpy.ndarray'>                    size: (6890,)
name: posedirs              type: <class 'chumpy.ch.Ch'>                    size: (6890, 3, 207)
name: pose_training_info    type: <type 'dict'>                             size: 6
name: bs_style              type: <type 'str'>                              size: 3
name: v_template            type: <class 'chumpy.ch.Ch'>                    size: (6890, 3)
name: shapedirs             type: <class 'chumpy.ch.Ch'>                    size: (6890, 3, 10)
name: bs_type               type: <type 'str'>                              size: 7
name: r                     type: <type 'numpy.ndarray'>                    size: (6890, 3)
"""


class SMPLModel(torch.nn.Module):
    def __init__(self, model_file):
        super().__init__()
        with open(model_file, "rb") as f:
            r"""
            keys: 'trans', 'gender', 'mocap_framerate', 'betas', 'dmpls', 'poses'
            """
            model = pickle.load(f, encoding="latin1")
        # shapedirs: effect of shape on vertices
        self.register_buffer(
            "shapedirs",
            torch.from_numpy(np.array(model["shapedirs"]).astype(np.float32))[:, :, :10],
        )
        self.register_buffer(
            "J_regressor",
            torch.from_numpy(model["J_regressor"].toarray().astype(np.float32)),
        )
        self.register_buffer(
            "v_template", torch.tensor(model["v_template"].astype(np.float32))
        )
        self.register_buffer("faces", torch.tensor(model["f"].astype(np.int32)))
        self.register_buffer(
            "parents", torch.tensor(model["kintree_table"][0].tolist())
        )
        self.parents[0] = 0
        self.register_buffer(
            "posedirs", torch.from_numpy(model["posedirs"].astype(np.float32))
        )
        self.register_buffer(
            "skinning_weights", torch.from_numpy(model["weights"].astype(np.float32))
        )
        self.register_buffer("J_template", torch.tensor(model["J"].astype(np.float32)))

        self._vertex_faces = None
        self.append_zero = lambda x: torch.cat(
            (
                x,
                torch.zeros_like(
                    x.index_select(-1, torch.tensor([0], device=x.device))
                ).float(),
            ),
            dim=-1,
        )
        self.append_one = lambda x: torch.cat(
            (
                x,
                torch.ones_like(x.index_select(-1, torch.tensor([0], device=x.device))).float(),
            ),
            dim=-1,
        )

    def __call__(self, pose: torch.Tensor = None, shape: torch.Tensor = None, pose2rot: bool = True):
        """Get the global poses, joint positions, and vertex positions from the local pose and shape parameters.
        :param torch.Tensor pose: (n_frames, 24, 3) | (n_frames, 72), local axis-angle representation
        :param torch.Tensor shape: (10), shape parameters
        """
        device = self.shapedirs.device
        if shape is None:
            J_tpose, V_tpose = self.J_template - self.J_template[:1], self.v_template - self.J_template[:1]
        else:
            shape_blend = torch.tensordot(
                shape.view(-1, 10), self.shapedirs, dims=([1], [2])
            )
            V_tpose = shape_blend + self.v_template
            J_tpose = torch.matmul(self.J_regressor, V_tpose)
            J_tpose, V_tpose = (J_tpose - J_tpose[:, :1]).squeeze(), (V_tpose - J_tpose[:, :1]).squeeze()
        if pose is None:
            return {
                "poses": torch.zeros(24, 3, device=device),
                "vertices": V_tpose,
                "joints": J_tpose,
            }
        if pose2rot:
            poses_rotation_matrix = axis_angle_to_matrix(pose.view(-1, 24, 3))
            n_frames = pose.shape[0]
        else:
            poses_rotation_matrix = pose.view([-1, 24, 3, 3])
            n_frames = poses_rotation_matrix.shape[0]
        # 1. zero pose mesh vertices and joint positions
        I = torch.eye(3, device=device)
        pose_blend = torch.tensordot(
            (poses_rotation_matrix[:, 1:] - I).flatten(1),
            self.posedirs,
            dims=([1], [2]),
        )
        V = V_tpose + pose_blend

        # 2. extend to all frames
        Js, Vs = [_.expand(n_frames, -1, -1) for _ in [J_tpose, V]]
        joints_local = torch.add(torch.neg(Js[:, self.parents]), Js)
        T_local = torch.zeros(n_frames, 24, 4, 4, device=device)
        T_local[..., :3, :3] = poses_rotation_matrix
        T_local[..., :3, 3] = joints_local
        T_local[..., -1, -1] = 1
        T_global = torch.zeros_like(T_local, device=device)
        T_global[:, 0] = T_local[:, 0]
        for i in range(1, len(self.parents)):
            T_global[:, i] = torch.bmm(T_global[:, self.parents[i]], T_local[:, i])
        poses_global = T_global[..., :3, :3].clone()
        joints = T_global[..., :3, 3].clone()
        T_global[..., -1:] -= torch.matmul(
            T_global, self.append_zero(Js).unsqueeze(-1)
        )  # guarantees that the contribution of the pose blend shapes is zero in the rest pose
        T_vertex = torch.tensordot(
            T_global, self.skinning_weights, dims=([1], [1])
        ).permute(0, 3, 1, 2)
        vertices = torch.matmul(
            T_vertex, self.append_one(Vs).unsqueeze(-1)
        ).squeeze(-1)[..., :3]
        return {
            "poses": poses_global,
            "vertices": vertices,
            "joints": joints,
        }

    @property
    def vertex_faces(self):
        """For each vertex of the mesh defined by SMPL return a list of faces this vertex is a part of."""
        if self._vertex_faces is None:
            mesh = trimesh.Trimesh(
                np.zeros((self.faces.max() + 1, 3)), self.faces, process=False
            )
            self._vertex_faces = torch.tensor(mesh.vertex_faces.copy().astype(np.int32))
        return self._vertex_faces

    def get_zero_pose_joint_and_vertex(self, shape: torch.Tensor = None):
        """
        Get the joint and vertex positions in zero pose. Root joint is aligned at zero.

        :param torch.Tensor shape: model shapes that can reshape to [batch_size, 10]. Use None for the mean(zero) shape.
        :return Joint tensor in shape [batch_size, num_joint, 3] and vertex tensor in shape [batch_size, num_vertex, 3]
                 if shape is not None. Otherwise [num_joint, 3] and [num_vertex, 3] assuming the mean(zero) shape.
        """
        if shape is None:
            J, V = (
                self.J_template - self.J_template[:1],
                self.v_template - self.J_template[:1],
            )
            return J, V
        else:
            shape = shape.view(-1, 10)
            shape_blend = torch.tensordot(
                shape.view(-1, 10), self.shapedirs, dims=([1], [2])
            )
            V = shape_blend + self.v_template
            J = torch.matmul(self.J_regressor, V)
            J, V = J - J[:, :1], V - J[:, :1]  # root joint as (0, 0, 0)

            return J.squeeze(), V.squeeze()

    # def get_pose_joint_and_vertex(self, pose, shape: torch.Tensor = None):
    #     """Get the global joint and vertex positions from the local pose and shape parameters.
    #     :param torch.Tensor pose: (n_frames, 24, 3), local axis-angle representation
    #     :param torch.Tensor shape: (n_frames, 10), shape parameters

    #     :return: {poses_global, vertices, joints} dict
    #     """
    #     poses_rotation_matrix = transforms.axis_angle_to_rotation_matrix(
    #         pose.reshape(-1, 3)
    #     ).reshape(-1, 24, 3, 3)

    #     # 1. zero pose mesh vertices and joint positions
    #     J, V_zero_pose = self.get_zero_pose_joint_and_vertex(shape)
    #     pose_blend = torch.tensordot(
    #         (poses_rotation_matrix[:, 1:] - torch.eye(3)).flatten(1),
    #         self.posedirs,
    #         dims=([1], [2]),
    #     )
    #     V = V_zero_pose + pose_blend

    #     # 2. extend to all frames
    #     Js, Vs = [_.expand(pose.shape[0], -1, -1) for _ in [J, V]]
    #     joints_local = torch.add(torch.neg(Js[:, self.parents]), Js)
    #     T_local = transforms.Rt_to_matrix4x4(poses_rotation_matrix, joints_local)
    #     T_global = [T_local[:, 0]]
    #     for i in range(1, len(self.parents)):
    #         T_global.append(torch.bmm(T_global[self.parents[i]], T_local[:, i]))
    #     T_global = torch.stack(T_global, dim=1)
    #     poses_global, joints = transforms.matrix4x4_to_Rt(T_global)
    #     T_global[..., -1:] -= torch.matmul(
    #         T_global, self.append_zero(Js, dim=-1).unsqueeze(-1)
    #     )  # guarantees that the contribution of the pose blend shapes is zero in the rest pose
    #     T_vertex = torch.tensordot(
    #         T_global, self.skinning_weights, dims=([1], [1])
    #     ).permute(0, 3, 1, 2)
    #     vertices = torch.matmul(
    #         T_vertex, self.append_one(Vs, dim=-1).unsqueeze(-1)
    #     ).squeeze(-1)[..., :3]
    #     return {
    #         "poses_global": poses_global,
    #         "vertices": vertices,
    #         "joints": joints,
    #     }

    def get_local_pose_from_global_pose(self, pose_global):
        pose_global = pose_global.view(pose_global.shape[0], -1, 3, 3)
        pose_local = [pose_global[:, 0]]
        for i in range(1, len(self.parents)):
            pose_local.append(
                torch.bmm(
                    pose_global[:, self.parents[i]].transpose(1, 2), pose_global[:, i]
                )
            )
        pose_local = torch.stack(pose_local, dim=1)
        return pose_local

    @staticmethod
    def get_vertices_with_trans(vertices, trans):
        assert (
            vertices.shape[0] == trans.shape[0]
        ), "The sequence length must be the same."
        return vertices + trans.view(-1, 1, 3)

    @staticmethod
    def get_joints_with_trans(joints, trans):
        assert (
            joints.shape[0] == trans.shape[0]
        ), "The sequence length must be the same."
        return joints + trans.view(-1, 1, 3)


def get_vertex_normals(
    vertices: torch.Tensor, vertex_ids, faces, normalize: bool = False
):
    """
    :param vertices: (n_frames, n_vertices, 3)
    :param vertex_ids: (n_selected_vertices)
    :param faces: triangle faces (n_faces, 3)

    :return vertex_normals: (n_frames, n_selected_vertices, 3)

    example:
    vertex_normals = get_vertex_normals(vertices, config.masks.amass_vertices, smpl_model.faces, normalize=True)
    """
    vertices = vertices.view(
        -1, vertices.shape[-2], vertices.shape[-1]
    )  # to (n_frames, n_vertices, 3)
    mesh = trimesh.Trimesh(
        np.zeros((vertices.shape[1], 3)), faces, process=False
    )  # construct mesh
    vertex_faces = mesh.vertex_faces[
        vertex_ids
    ]  # corresponding face ids of selected vertices (
    face_ids = np.unique(
        vertex_faces[vertex_faces != -1]
    )  # unique face ids that contain selected vertices
    selected_faces = faces[face_ids]
    selected_vertices = vertices[:, selected_faces]
    face_normals = torch.cross(
        selected_vertices[:, :, 1] - selected_vertices[:, :, 0],
        selected_vertices[:, :, 2] - selected_vertices[:, :, 0],
        dim=-1,
    )
    mesh_with_selected_faces = trimesh.Trimesh(
        np.zeros((vertices.shape[1], 3)), selected_faces, process=False
    )
    face_ids = mesh_with_selected_faces.vertex_faces[vertex_ids]

    vertex_degrees = mesh_with_selected_faces.vertex_degree[vertex_ids]
    vertex_normals = (
        face_normals[:, face_ids].sum(dim=-2) / vertex_degrees[None, :, None]
    )
    if normalize:
        vertex_normals = vertex_normals / torch.norm(
            vertex_normals, dim=-1, keepdim=True
        )

    return vertex_normals.squeeze()


def get_local_coordinate_frame(
    vertices, vertex_ids, faces, vertex_faces, vertex_normals=None
):
    """
    :param vertices: (n_frames, n_vertices, 3)
    :param vertex_ids: (n_selected_vertices)
    :param faces: triangle faces (n_faces, 3)
    :param vertex_faces: vertex_faces of the mesh
    :param vertex_normals: (n_frames, n_vertices, 3), if None, it will be calculated

    :return local_rot: (n_frames, n_selected_vertices, 3, 3)

    example:
    local_rot = get_local_coordinate_frame(vertices, config.masks.amass_vertices, smpl_model.faces, smpl_model.vertex_faces, vertex_normals)
    """
    vertices = vertices.view(-1, vertices.shape[-2], vertices.shape[-1])
    if vertex_normals is None:
        vertex_normals = get_vertex_normals(vertices, vertex_ids, faces, normalize=True)
    vertex_normals = vertex_normals.view(
        -1, vertex_normals.shape[-2], vertex_normals.shape[-1]
    )

    adjacent_vertices = []
    for v in vertex_ids:
        for v_candidate in faces[vertex_faces[v, 0]]:
            if v_candidate != v:
                adjacent_vertices.append(v_candidate.item())
                break

    on_surface = vertices[:, adjacent_vertices] - vertices[:, vertex_ids]
    on_surface = on_surface / torch.norm(on_surface, dim=-1, keepdim=True)
    third_axis = torch.cross(vertex_normals.float(), on_surface)
    third_axis = third_axis / torch.norm(third_axis, dim=-1, keepdim=True)
    on_surface = torch.cross(third_axis, vertex_normals.float())
    on_surface = on_surface / torch.norm(on_surface, dim=-1, keepdim=True)

    local_rot = torch.zeros([vertex_normals.shape[0], len(vertex_ids), 3, 3])
    local_rot[..., :, 0] = on_surface
    local_rot[..., :, 1] = third_axis
    local_rot[..., :, 2] = vertex_normals

    return local_rot.squeeze()


def get_smpl_model(gender: Literal[0, 1, "male", "female", "neutral"]):
    if gender == 1 or gender == "male":
        return SMPLModel(config.paths.smpl_model_file_male)
    elif gender == 0 or gender == "female":
        return SMPLModel(config.paths.smpl_model_file_female)
    else:
        return SMPLModel(config.paths.smpl_model_file_neutral)
