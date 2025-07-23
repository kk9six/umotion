import torch
import trimesh
from config import masks, BodyMeasurement, TARGET_FPS
from scipy.spatial import ConvexHull
from utils.utils import get_intersection_points

def synthesize_distance_noise_std(los_proportions, threshold_upper=0.9, threshold_lower=0.5, sigma_min=0.03, sigma_max=0.25, sigma_imu=0.05):
    """
    """
    result = torch.where(
        los_proportions >= threshold_upper,
        sigma_min,
        torch.where(
            los_proportions < threshold_lower,
            sigma_imu,
            (sigma_max - sigma_min) / (threshold_lower - threshold_upper) * (los_proportions - threshold_upper) + sigma_min
        )
    )
    return result


def synthesize_line_of_sight_proportion(vertices, faces):
    # frame, vertex, xyz
    vertices = vertices.view((-1,) + vertices.shape[-2:])
    x_idxs, y_idxs = (
        torch.combinations(torch.arange(6), 2)[:, 0],
        torch.combinations(torch.arange(6), 2)[:, 1],
    )
    los_proportions = []
    for vert in vertices:
        lines = torch.cat(
            [
                vert[masks.amass_vertices][x_idxs].unsqueeze(1),
                vert[masks.amass_vertices][y_idxs].unsqueeze(1),
            ],
            dim=1,
        )
        for i in range(15):
            p1, p2 = lines[i][0], lines[i][1]
            p1_vi, p2_vi = (
                masks.amass_vertices[x_idxs[i]],
                masks.amass_vertices[y_idxs[i]],
            )
            _, intersected_points, entry_exit_indicators = get_intersection_points(
                p1, p1_vi, p2, p2_vi, vert, faces
            )

            def get_los_proportion(p1, p2, intersection_points, entry_exit_indicators):
                """
                Get the proportion of the line of sight between p1 and p2

                params:
                - p1: torch.tensor, (3,), the starting point of the line segment
                - p2: torch.tensor, (3,), the ending point of the line segment
                - intersection_points: torch.tensor, (n, 3), the intersection points
                - entry_exit_indicators: torch.tensor, (n, 1), 0: exit, 1: entry

                return:
                - proportion: float, the proportion of the line of sight between p1 and p2
                """
                device = p1.device
                if len(intersection_points) == 0:
                    return 1.0
                points = torch.cat(
                    [p1.unsqueeze(0), intersection_points, p2.unsqueeze(0)], dim=0
                )  # start, intersection, end
                indicators = torch.cat(
                    [
                        torch.tensor([1 - entry_exit_indicators[0]], device=device),
                        entry_exit_indicators,
                        torch.tensor([1 - entry_exit_indicators[-1]], device=device),
                    ]
                )  # entry -> exit -> entry ..., or exit -> entry -> exit ...
                # if entry -> exit, then its body occluded, otherwise not
                distances = (points[1:] - points[:-1]).norm(dim=1)
                LOS = (indicators[1:] - indicators[:-1]) == 1
                return ((distances[LOS]).sum() / distances.sum()).item()

            los_proportions.append(
                get_los_proportion(p1, p2, intersected_points, entry_exit_indicators)
            )

    return torch.tensor(los_proportions).reshape(-1, 15)


def synthesize_acceleration_orientation_distance(
    poses: torch.Tensor, vertices: torch.Tensor
):
    """synthesize acceleration, orientation, and distance from poses and vertices
    :param poses: torch.Tensor of shape [n_frames, n_joints, 3], global poses
    :param vertices: torch.Tensor of shape [n_frames, n_vertices, 3]

    :return: accs, accs_smooth, oris, dists
    """
    # 1. acceleration
    accs = torch.stack(
        [
            (vertices[i] + vertices[i + 2] - 2 * vertices[i + 1])
            * (TARGET_FPS**2)
            for i in range(0, vertices.shape[0] - 2)
        ]
    )
    accs = torch.cat((torch.zeros_like(accs[:1]), accs, torch.zeros_like(accs[:1])))
    accs = accs[:, masks.amass_vertices]

    # 2. orientation
    oris = poses[:, masks.amass_joints]

    # 3. distance
    n_vertices = len(masks.amass_vertices)
    dists = torch.cdist(
        vertices[:, masks.amass_vertices],
        vertices[:, masks.amass_vertices],
        p=2,
    )[
        :,
        torch.triu_indices(n_vertices, n_vertices, 1)[0],
        torch.triu_indices(n_vertices, n_vertices, 1)[1],
    ]

    return (
        accs,
        oris,
        dists,
    )


def synthesize_body_measurements(vertices, faces, types=None):
    def get_circumference(isect_vertices, dims):
        convex_hull = ConvexHull(isect_vertices[:, dims])
        convex_hull_paths = torch.tensor(isect_vertices[convex_hull.simplices])
        return (
            torch.norm(convex_hull_paths[:, 1] - convex_hull_paths[:, 0], dim=-1)
            .sum()
            .item()
        )

    measurements = {}
    mesh = trimesh.Trimesh(
        vertices=vertices, faces=faces, process=False, maintain_order=True
    )
    # tris = vertices[faces]
    if types is None:
        types = BodyMeasurement
    elif isinstance(types, BodyMeasurement):
        types = [types]
    else:
        types = types

    for t in types:
        if t == BodyMeasurement.HEIGHT:
            v = abs(
                vertices[masks.head_top_vertex][1]
                - vertices[masks.heel_left_vertex][1]
            ).item()
        elif t == BodyMeasurement.WEIGHT:
            """
            weight = volume * density (density = 985 in SHAPY)
            volume = sum of all terahedron volumes
            terahedron volumes = 1/6 * det(x y z), x y z are three vertices (if apex is origin)
            | a b c |
            | d e f | = aei + bfg + cdh - ceg - bdi - afh
            | g h i |

            the volume of tetrahedra = 1/6 * v_1 \dot (v_2 \cross v_3)
            volumn = (vertices[faces[:, 0]] * vertices[faces[:, 1]] * vertices[faces[:, 2]]).sum(dim=-1) / 6
            """
            density = 985
            v = mesh.volume * density
        elif t == BodyMeasurement.CHEST:
            slice2d = mesh.section(
                plane_normal=(0, 1, 0),
                plane_origin=vertices[masks.nipple_right_vertex],
            )
            v = get_circumference(slice2d.vertices, [0, 2])
        elif t == BodyMeasurement.WAIST:
            slice2d = mesh.section(
                plane_normal=(0, 1, 0),
                plane_origin=vertices[masks.belly_button_vertex],
            )
            v = get_circumference(slice2d.vertices, [0, 2])
        elif t == BodyMeasurement.HIP:
            slice2d = mesh.section(
                plane_normal=(0, 1, 0),
                plane_origin=vertices[masks.crotch_vertex],
            )
            v = get_circumference(slice2d.vertices, [0, 2])
        elif t == BodyMeasurement.WRIST:
            slice2d = mesh.section(
                plane_normal=(1, 0, 0),
                plane_origin=vertices[masks.wrist_left_vertex],
            )
            v = get_circumference(slice2d.vertices, [1, 2])
        elif t == BodyMeasurement.HEAD:
            slice2d = mesh.section(
                plane_normal=(0, 1, 0),
                plane_origin=vertices[masks.head_middle_vertex],
            )
            v = get_circumference(slice2d.vertices, [0, 2])
        elif t == BodyMeasurement.KNEE:
            slice2d = mesh.section(
                plane_normal=(0, 1, 0),
                plane_origin=vertices[masks.knee_left_vertex],
            )
            v = get_circumference(slice2d.discrete[0], [0, 2])
        elif t == BodyMeasurement.WRIST_to_WRIST:
            wrist_left = vertices[masks.amass_vertices[0]]
            wrist_right = vertices[masks.amass_vertices[1]]
            v = torch.norm(wrist_left - wrist_right).item()
        elif t == BodyMeasurement.HEAD_to_WAIST:
            waist = vertices[masks.amass_vertices[5]]
            head = vertices[masks.amass_vertices[4]]
            v = torch.norm(waist - head).item()
        elif t == BodyMeasurement.WRIST_to_WAIST:
            waist = vertices[masks.amass_vertices[5]]
            wrist_left = vertices[masks.amass_vertices[0]]
            v = torch.norm(waist - wrist_left).item()
        elif t == BodyMeasurement.KNEE_to_WAIST:
            waist = vertices[masks.amass_vertices[5]]
            knee_left = vertices[masks.amass_vertices[2]]
            v = torch.norm(waist - knee_left).item()
        elif t == BodyMeasurement.WRIST_to_HEAD:
            head = vertices[masks.amass_vertices[4]]
            wrist_left = vertices[masks.amass_vertices[0]]
            v = torch.norm(head - wrist_left).item()
        elif t == BodyMeasurement.WRIST_to_KNEE:
            wrist_left = vertices[masks.amass_vertices[0]]
            knee_left = vertices[masks.amass_vertices[2]]
            v = torch.norm(wrist_left - knee_left).item()
        elif t == BodyMeasurement.HEAD_to_KNEE:
            head = vertices[masks.amass_vertices[4]]
            knee_left = vertices[masks.amass_vertices[2]]
            v = torch.norm(head - knee_left).item()
        elif t in BodyMeasurement:
            continue
        else:
            raise Exception(f"Invalid measurement type: {t}")
        measurements[t.name] = v

    return measurements
