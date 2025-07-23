import open3d as o3d
import torch
import trimesh
import seaborn as sns
import numpy as np
from trimesh.exchange.obj import Image
import config
import prettytable as pt


""" Example

1. draw mesh
from utils.visualizer import Visualizer
from data.amass import amass_fnames
from utils.smpl import get_smpl_model

# amass example
fname = amass_fnames(processed=True)[3550]
data = torch.load(fname)
gender = data["gender"]
poses, betas = data["poses"], data["betas"]

# calculate vertices
smpl_model = get_smpl_model(gender)
timestep = 440
vertices = smpl_model.get_pose_joint_and_vertex(pose=poses, shape=betas)["vertices"][timestep]
scene_content = [
    Visualizer.view_mesh(vertices, smpl_model.faces, wireframe=False),
]
Visualizer.viewer(scene_content).show()

2. draw selected amass vertices on the mesh
scene_content = [
    Visualizer.view_mesh(vertices, smpl_model.faces, wireframe=False),
    Visualizer.view_points(vs[config.masks.amass_vertices], radius=0.015, highlight=True),
]
Visualizer.viewer(scene_content).show()

3. draw 15 inter distances
paths = torch.stack([vs[config.masks.amass_vertices][config.masks.x_idxs], vs[config.masks.amass_vertices][config.masks.y_idxs]], dim=1)
scene_content = [
    Visualizer.view_mesh(vertices, smpl_model.faces, wireframe=False),
    Visualizer.view_paths(paths, color=(0, 0, 255)),  # blue
]
Visualizer.viewer(scene_content).show()

4. draw vertex normal and local coordinate frames
vertex_normals = get_vertex_normals(vertices, config.masks.amass_vertices, smpl_model.faces, normalize=True)
local_coordinate_frames = get_local_coordinate_frame(vertices, config.masks.amass_vertices, smpl_model.faces, smpl_model.vertex_faces, vertex_normals)

import trimesh

colors = [(255, 0, 0), (47, 109, 28), (0, 0, 255)]

content = []
for i, mask in enumerate(config.masks.amass_vertices):
    for j in range(3):
        path = torch.vstack(
            [vs[mask], vs[mask] + local_coordinate_frames[i].T[j] * 0.1]
        )
        p = trimesh.load_path(path)
        p.colors = [colors[j]]
        content.append(p)

scene_content = [
    Visualizer.view_mesh(vs, smpl_model.faces, wireframe=False),
    Visualizer.view_points(vs[config.masks.amass_vertices], radius=0.015, highlight=True),
    content,
]
Visualizer.viewer(scene_content).show()


5. visualize motion
# load data
in_fname = "data/processed/AMASS/HumanEva/S3/Walking_3_poses.pt"
data = torch.load(in_fname)
smpl_model = get_smpl_model(data["gender"])
vertices = smpl_model.get_pose_joint_and_vertex(data["poses"])["vertices"]
vertices_with_trans = smpl_model.get_vertices_with_trans(vertices, data["trans"])  # 1 seconds
faces = smpl_model.faces
colors = np.array(sns.light_palette("gray", n_colors=vertices.shape[0]))
scene_content = [
    [
        Visualizer.view_mesh(vertices_with_trans[[i], ...], faces, colors),
    ]
    for i in [0, 30, 60, 90, 120, 150, 240]
]
Visualizer.viewer(scene_content).show()

6. save to obj.
Visualizer.view_mesh(vertices[timestep], smpl_model.faces).export(f"poses_{timestep}.obj")
"""

class Visualizer:
    def __init__(self) -> None:
        pass

    @staticmethod
    def view_as_tables(data, title="", n_col=15):
        t = pt.PrettyTable()
        t.header = False
        if title != "":
            t.title = title
        t.align = "c"
        i = 0
        while True:
            try:
                t.add_row([f"{_:.2f}" for _ in data[i * n_col : (i + 1) * n_col]])
                i += 1
            except:
                break
        if len(data) % n_col != 0:
            last_row = [f"{_:.2f}" for _ in data[i * n_col :]]
            last_row += ["-"] * (n_col - len(last_row))
            t.add_row(last_row)
        print(t)

    @staticmethod
    def view_points(points, color=(255, 0, 0), highlight=False, radius=0.02):
        """
        :param (n, 3) or (3) points: points to be visualized
        :param rgb, (b, 3) or (3,) color: color of the points
        :param bool highlight: if True, draw a sphere at each point
        :param float radius: radius of the marker

        :return trimesh.PointCloud content: content to be visualized
        """
        points = points.reshape(-1, 3)
        if highlight == False:  # pure point
            point_cloud = trimesh.PointCloud(
                vertices=points, colors=[color] * len(points)
            )
            return point_cloud
        # sphere with radius 0.02
        content = []
        for p in points:
            marker = trimesh.creation.icosphere(subdivisions=3, radius=radius)
            marker.apply_translation(p)
            marker.visual.vertex_colors = [color] * len(marker.vertices)
            content.append(marker)
        return content

    @staticmethod
    def view_mesh(vertices, faces, colors=None, wireframe=False, vertex_colors=None):
        """
        :param (b, n, 3) or (n, 3) vertices: vertices to be visualized
        :param (n_tris, 3) faces: faces of the mesh
        :param (b, 3) or (3, ) colors: color of the mesh
        :param bool wireframe: if True, draw the wireframe of the mesh

        :return trimesh.Trimesh content: content to be visualized
        """
        # reshape to (b, n ,3)
        vertices = vertices.reshape(-1, vertices.shape[-2], vertices.shape[-1])
        n_batch = vertices.shape[0]
        if colors is None:
            colors = [None] * n_batch
        else:
            colors = torch.tensor(colors)
            colors = colors.reshape(-1, colors.shape[-1])

        if vertex_colors is None:
            vertex_colors = [None] * n_batch
        else:
            vertex_colors = torch.tensor(vertex_colors)
            vertex_colors = vertex_colors.reshape(-1, vertex_colors.shape[-1])

        content = []
        for i in range(n_batch):
            mesh = trimesh.Trimesh(
                vertices=vertices[i],
                faces=faces,
                process=False,
                maintain_order=True,
                face_colors=colors[i],
                vertex_colors=vertex_colors[i],
            )
            if wireframe == False:
                content.append(mesh)
                continue
            # Create 3D paths from the edges
            edges = mesh.edges
            edge_paths = trimesh.load_path(mesh.vertices[edges])
            edge_paths.colors = np.repeat(
                [[214, 214, 214, 1]], len(edge_paths.entities), axis=0
            )
            content.append(edge_paths)
        return content

    @staticmethod
    def view_coordinate_frame():
        """ """
        axis_size = 0.4
        coordinate_frame = trimesh.creation.axis(
            origin_color=(125, 125, 125, 255), axis_radius=0.01, axis_length=axis_size
        )
        return coordinate_frame

    @staticmethod
    def view_paths(paths, color=(255, 0, 0)):
        """
        :param paths:
        :param color:
        """
        try:
            re_paths = paths.reshape(-1, paths.shape[-2], paths.shape[-1])
            p = trimesh.load_path(re_paths)
        except:
            p = trimesh.load_path(paths)
        # p.colors = sns.light_palette('red', n_colors=len(paths)-1)
        p.colors = [color] * len(p.entities)
        return p

    @staticmethod
    def viewer(scene_content):
        """
        viewer(scene_content).show()
        """
        # scene = trimesh.Scene(flatten(scene_content))
        scene = trimesh.Scene(scene_content)
        return scene

    @staticmethod
    def view_motion(vertice_sequence, faces, fname="output.mp4", fps=60):
        mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(vertice_sequence[0]),
            triangles=o3d.utility.Vector3iVector(faces),
        )
        mesh.compute_vertex_normals()
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.3, origin=[0, 0, 0]
        )
        vis.add_geometry(coordinate_frame)

        vis.add_geometry(mesh)

        images = []
        for vertices in vertice_sequence[1:]:
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.compute_vertex_normals()
            vis.update_geometry(mesh)
            vis.poll_events()
            vis.update_renderer()

            image = vis.capture_screen_float_buffer(do_render=True)
            images.append((np.asarray(image) * 255).astype(np.uint8))

        vis.destroy_window()

        from moviepy.editor import ImageSequenceClip

        clip = ImageSequenceClip(images, fps=fps)
        clip.write_videofile(fname)

    @staticmethod
    def export_mesh(vertices, faces, fname):
        mesh = trimesh.Trimesh(
            vertices=vertices, faces=faces, process=False, maintain_order=True
        )
        mesh.export(fname)

    @staticmethod
    def view_all_joints(model, betas=None, draw_coordinate_frame=True):
        joints, vertices = model.get_zero_pose_joint_and_vertex(betas)
        content = [
            Visualizer.view_mesh(vertices, model.faces, wireframe=True),
            Visualizer.view_points(
                joints,
                color=[0, 0, 255, 255],
                highlight=True,
            ),
        ]
        if draw_coordinate_frame:
            content.append(Visualizer.view_coordinate_frame())
        return content

    @staticmethod
    def view_selected_joints(model, betas=None, draw_coordinate_frame=True):
        joints, vertices = model.get_zero_pose_joint_and_vertex(betas)
        content = [
            Visualizer.view_mesh(vertices, model.faces, wireframe=True),
            Visualizer.view_points(
                joints[config.masks.amass_joints],
                color=[0, 0, 255, 255],
                highlight=True,
            ),
        ]
        if draw_coordinate_frame:
            content.append(Visualizer.view_coordinate_frame())
        return content

    @staticmethod
    def view_selected_vertices(model, betas=None, draw_coordinate_frame=True):
        _, vertices = model.get_zero_pose_joint_and_vertex(betas)
        content = [
            Visualizer.view_mesh(vertices, model.faces, wireframe=True),
            Visualizer.view_points(
                vertices[config.masks.amass_vertices],
                color=[0, 0, 255, 255],
                highlight=True,
            ),
        ]
        if draw_coordinate_frame:
            content.append(Visualizer.view_coordinate_frame())
        return content
