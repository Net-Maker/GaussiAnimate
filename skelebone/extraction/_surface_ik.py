"""Constrained correction for residual bone/surface intersections."""

import numpy as np

from ..kinematics import forward_kinematics, quaternion_to_matrix, hierarchy_order
from ..types import PoseSequence


class SignedSurface:
    """Raycast distances with a winding-number sign for intersecting surfaces."""

    def __init__(self, vertices, faces):
        import open3d as o3d

        self.vertices = np.ascontiguousarray(vertices, dtype=np.float64)
        self.faces = np.ascontiguousarray(faces, dtype=np.int64)
        legacy = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(self.vertices), o3d.utility.Vector3iVector(self.faces),
        )
        self.intersects = legacy.is_self_intersecting()
        self.scene = o3d.t.geometry.RaycastingScene(nthreads=1)
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(legacy))
        self.winding = None
        if self.intersects:
            import igl

            self.winding = getattr(igl, "fast_winding_number", None) or igl.fast_winding_number_for_meshes

    def compute_signed_distance(self, query_points, *, nsamples=3):
        import open3d as o3d

        result = self.scene.compute_signed_distance(query_points, nsamples=nsamples)
        if self.winding is None:
            return result
        values = result.numpy().copy()
        flat = values.reshape(-1)
        positive = np.flatnonzero(flat > 0.)
        if len(positive):
            queries = query_points.numpy().astype(np.float64).reshape(-1, 3)
            winding = self.winding(self.vertices, self.faces, queries[positive])
            interior = positive[np.abs(np.asarray(winding).reshape(-1)) >= .5]
            flat[interior] *= -1.
        return o3d.core.Tensor(values)

    def compute_closest_points(self, query_points):
        return self.scene.compute_closest_points(query_points)


def _rotation_right_jacobian(vector):
    angle = np.linalg.norm(vector)
    x, y, z = vector
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    if angle < 1e-5:
        return np.eye(3) - .5*skew + (skew@skew)/6.
    return np.eye(3) - (1.-np.cos(angle))/angle**2*skew + (angle-np.sin(angle))/angle**3*(skew@skew)


def constrain_surface_frame(skeleton, target, initial, scene, origin, *, margin, extent):
    """Correct a single pose with SLSQP signed-distance inequalities."""
    import open3d as o3d
    from scipy.optimize import minimize
    from scipy.spatial.transform import Rotation

    children = np.flatnonzero(skeleton.parents >= 0)
    sample_children = []
    sample_parents = []
    sample_fractions = []
    for child in children:
        length = np.linalg.norm(skeleton.offsets[child])
        subdivisions = max(1, int(np.ceil(length / (extent*.002))))
        sample_children.extend([int(child)]*(subdivisions-1))
        sample_parents.extend([int(skeleton.parents[child])]*(subdivisions-1))
        sample_fractions.extend(np.arange(1, subdivisions)/subdivisions)
    children = np.asarray(sample_children, dtype=np.int64)
    parents = np.asarray(sample_parents, dtype=np.int64)
    fractions = np.asarray(sample_fractions)[:, None]

    def samples(points):
        interiors = (1.-fractions)*points[parents] + fractions*points[children]
        return np.concatenate([points, interiors])

    def distances(points):
        return scene.compute_signed_distance(
            o3d.core.Tensor((samples(points)-origin).astype(np.float32)), nsamples=3,
        ).numpy().astype(np.float64)

    initial_positions = forward_kinematics(skeleton, initial)[1][0]
    initial_distance = distances(initial_positions)
    if initial_distance.max() <= margin:
        return initial
    correction_limit = 2.*max(float(initial_distance.max()), 0.) + extent*.005
    # Other branches must compensate when a shared ancestor moves.
    ids = np.unique(skeleton.parents[skeleton.parents >= 0])
    descendants = {int(joint): [] for joint in ids}
    for child in hierarchy_order(skeleton.parents):
        parent = int(skeleton.parents[child])
        while parent >= 0:
            if parent in descendants:
                descendants[parent].append(child)
            parent = int(skeleton.parents[parent])
    base = quaternion_to_matrix(initial.local_rotations[0])
    cached = {}

    def evaluate(parameters):
        if "parameters" not in cached or not np.array_equal(parameters, cached["parameters"]):
            local = base.copy()
            if len(ids):
                local[ids] = local[ids] @ Rotation.from_rotvec(parameters[:-3].reshape(-1, 3)).as_matrix()
            quaternion = Rotation.from_matrix(local).as_quat()[:, [3, 0, 1, 2]][None]
            pose = PoseSequence(quaternion, initial.root_translation + parameters[-3:]*extent, fps=initial.fps)
            transforms, positions = forward_kinematics(skeleton, pose)
            points = positions[0]
            jacobian = np.zeros((skeleton.num_joints, 3, len(parameters)))
            jacobian[:, :, -3:] = np.eye(3)*extent
            for index, joint in enumerate(ids):
                axes = transforms[0, joint, :3, :3] @ _rotation_right_jacobian(parameters[index*3:index*3+3])
                nodes = descendants[int(joint)]
                displacement = points[nodes] - points[joint]
                jacobian[nodes, :, index*3:index*3+3] = np.cross(axes.T[None], displacement[:, None]).transpose(0, 2, 1)
            queries = o3d.core.Tensor((samples(points)-origin).astype(np.float32))
            signed = scene.compute_signed_distance(queries, nsamples=3).numpy().astype(np.float64)
            closest = scene.compute_closest_points(queries)
            surface = closest["points"].numpy().astype(np.float64) + origin
            normal = closest["primitive_normals"].numpy().astype(np.float64)
            stable = np.abs(signed) > extent*1e-7
            normal[stable] = (samples(points)[stable]-surface[stable]) / signed[stable, None]
            interior_jacobian = (1.-fractions[..., None])*jacobian[parents] + fractions[..., None]*jacobian[children]
            sample_jacobian = np.concatenate([jacobian, interior_jacobian])
            constraint_jacobian = -np.einsum("ni,nik->nk", normal, sample_jacobian) / extent
            cached.update(parameters=parameters.copy(), pose=pose, points=points, distance=signed, jacobian=jacobian, constraint_jacobian=constraint_jacobian)
        return cached

    def objective(parameters):
        values = evaluate(parameters)
        return np.sum((values["points"]-target)**2) / extent**2 + 1e-3*np.dot(parameters[:-3], parameters[:-3])

    def constraint(parameters):
        values = evaluate(parameters)
        movement = (correction_limit**2 - np.sum((values["points"]-initial_positions)**2, axis=1)) / extent**2
        return np.concatenate([(-values["distance"]-margin) / extent, movement])

    def objective_jacobian(parameters):
        values = evaluate(parameters)
        gradient = 2.*np.einsum("ni,nik->k", values["points"]-target, values["jacobian"]) / extent**2
        gradient[:-3] += 2e-3*parameters[:-3]
        return gradient

    def constraint_jacobian(parameters):
        values = evaluate(parameters)
        movement = -2.*np.einsum("ni,nik->nk", values["points"]-initial_positions, values["jacobian"]) / extent**2
        return np.concatenate([values["constraint_jacobian"], movement])

    initial_parameters = np.zeros(len(ids)*3 + 3)
    result = minimize(
        objective, initial_parameters, jac=objective_jacobian, method="SLSQP",
        constraints={"type": "ineq", "fun": constraint, "jac": constraint_jacobian},
        bounds=[(-np.pi/2, np.pi/2)]*(len(ids)*3) + [(-.25, .25)]*3,
        options={"maxiter": 100, "ftol": 1e-9, "eps": 1e-5},
    )
    candidate = evaluate(result.x)
    # Solver status alone is insufficient: accept only a measured improvement.
    displacement = np.linalg.norm(candidate["points"]-initial_positions, axis=1).max()
    if np.all(np.isfinite(candidate["distance"])) and candidate["distance"].max() < initial_distance.max() and displacement <= correction_limit*1.01:
        return candidate["pose"]
    return initial
