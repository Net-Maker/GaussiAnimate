"""Surface preparation for curve-skeleton extraction."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
import warnings

import numpy as np

from ..types import DynamicPointSequence
from ._geometry import as_points


@dataclass
class TriangleMesh:
    vertices: np.ndarray
    faces: np.ndarray

    def __post_init__(self) -> None:
        self.vertices = as_points(self.vertices, "vertices")
        self.faces = np.asarray(self.faces, dtype=np.int64)
        if self.faces.size == 0:
            self.faces = np.empty((0, 3), dtype=np.int64)
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError("faces must have shape [triangles, 3]")
        if self.faces.size and (self.faces.min() < 0 or self.faces.max() >= len(self.vertices)):
            raise ValueError("faces contains an out-of-range vertex index")


@dataclass(frozen=True)
class MeshConfig:
    frame: int = 0
    poisson_depth: int = 9
    normal_radius: Optional[float] = None
    normal_neighbors: int = 30
    normal_consistency_neighbors: int = 15
    density_quantile: float = 0.01
    minimum_component_triangles: int = 10
    target_triangles: int = 20000

    def __post_init__(self) -> None:
        if self.frame < 0 or self.poisson_depth < 1:
            raise ValueError("frame must be non-negative and poisson_depth positive")
        if self.normal_radius is not None and self.normal_radius <= 0:
            raise ValueError("normal_radius must be positive")
        if self.normal_neighbors < 3 or self.normal_consistency_neighbors < 3:
            raise ValueError("normal-neighborhood sizes must be at least three")
        if not 0 <= self.density_quantile < 1:
            raise ValueError("density_quantile must be in [0, 1)")
        if self.minimum_component_triangles < 1:
            raise ValueError("minimum_component_triangles must be positive")
        if self.target_triangles < 100:
            raise ValueError("target_triangles must be at least 100")


def write_off(path: Union[str, Path], mesh: TriangleMesh) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as stream:
        stream.write("OFF\n")
        stream.write(f"{len(mesh.vertices)} {len(mesh.faces)} 0\n")
        for vertex in mesh.vertices:
            stream.write("{:.17g} {:.17g} {:.17g}\n".format(*vertex))
        for face in mesh.faces:
            stream.write(f"3 {face[0]} {face[1]} {face[2]}\n")
    return path


def read_off(path: Union[str, Path]) -> TriangleMesh:
    path = Path(path)
    with path.open("r", encoding="ascii") as stream:
        lines = [line.partition("#")[0].strip() for line in stream]
    tokens = [line for line in lines if line]
    if not tokens or tokens[0] != "OFF":
        raise ValueError(f"{path} is not an ASCII OFF mesh")
    if len(tokens) < 2:
        raise ValueError(f"{path} has no OFF counts line")
    counts = tokens[1].split()
    if len(counts) < 2:
        raise ValueError(f"{path} has an invalid OFF counts line")
    num_vertices, num_faces = int(counts[0]), int(counts[1])
    vertex_lines = tokens[2:2 + num_vertices]
    face_lines = tokens[2 + num_vertices:2 + num_vertices + num_faces]
    if len(vertex_lines) != num_vertices or len(face_lines) != num_faces:
        raise ValueError(f"{path} is truncated")
    vertices = np.asarray([[float(v) for v in line.split()[:3]] for line in vertex_lines])
    faces = []
    for line in face_lines:
        values = [int(value) for value in line.split()]
        if not values or values[0] != 3 or len(values) < 4:
            raise ValueError("only triangular OFF faces are supported")
        faces.append(values[1:4])
    return TriangleMesh(vertices, np.asarray(faces, dtype=np.int64))


def load_mesh(path: Union[str, Path]) -> TriangleMesh:
    path = Path(path)
    if path.suffix.lower() == ".off":
        return read_off(path)
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError("loading non-OFF meshes requires the 'geometry' extra (Open3D)") from exc
    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh.is_empty():
        raise ValueError(f"could not load a triangle mesh from {path}")
    return TriangleMesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles))


def remove_small_components(mesh: TriangleMesh, minimum_triangles: int = 10) -> TriangleMesh:
    """Discard disconnected triangle components smaller than a fixed support."""

    if len(mesh.faces) == 0:
        return mesh
    vertex_faces = [[] for _ in range(len(mesh.vertices))]
    for face_index, face in enumerate(mesh.faces):
        for vertex in face:
            vertex_faces[int(vertex)].append(face_index)
    unvisited = set(range(len(mesh.faces)))
    keep = []
    while unvisited:
        seed = unvisited.pop()
        component = [seed]
        stack = [seed]
        while stack:
            face_index = stack.pop()
            for vertex in mesh.faces[face_index]:
                for neighbor in vertex_faces[int(vertex)]:
                    if neighbor in unvisited:
                        unvisited.remove(neighbor)
                        component.append(neighbor)
                        stack.append(neighbor)
        if len(component) >= minimum_triangles:
            keep.extend(component)
    if not keep:
        keep = list(range(len(mesh.faces)))
    faces = mesh.faces[np.asarray(sorted(keep), dtype=np.int64)]
    used = np.unique(faces)
    remap = np.full(len(mesh.vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return TriangleMesh(mesh.vertices[used], remap[faces])


def is_watertight(mesh: TriangleMesh) -> bool:
    """Return whether faces form a closed, consistently oriented triangle surface."""

    if len(mesh.faces) < 4:
        return False
    directed = np.concatenate(
        [mesh.faces[:, [0, 1]], mesh.faces[:, [1, 2]], mesh.faces[:, [2, 0]]],
        axis=0,
    )
    edges = np.sort(directed, axis=1)
    unique, inverse, counts = np.unique(
        edges, axis=0, return_inverse=True, return_counts=True
    )
    if not len(unique) or not np.all(counts == 2):
        return False
    direction = np.where(directed[:, 0] == edges[:, 0], 1, -1)
    balance = np.zeros(len(unique), dtype=np.int64)
    np.add.at(balance, inverse, direction)
    return bool(np.all(balance == 0))


def split_nonmanifold_vertices(mesh: TriangleMesh) -> TriangleMesh:
    """Duplicate vertices whose incident triangles form disjoint surface fans."""

    faces = mesh.faces.copy()
    vertices = mesh.vertices.tolist()
    for vertex in range(len(mesh.vertices)):
        incident = np.flatnonzero(np.any(faces == vertex, axis=1))
        if len(incident) < 2:
            continue
        incident_set = set(incident.tolist())
        adjacency = {int(face): set() for face in incident}
        other_vertices = np.unique(faces[incident][faces[incident] != vertex])
        for other in other_vertices:
            shared = [face for face in incident_set if other in faces[face]]
            if len(shared) == 2:
                adjacency[shared[0]].add(shared[1])
                adjacency[shared[1]].add(shared[0])
        components = []
        unseen_faces = set(incident_set)
        while unseen_faces:
            seed = unseen_faces.pop()
            component = {seed}
            stack = [seed]
            while stack:
                face = stack.pop()
                for neighbor in adjacency[face]:
                    if neighbor in unseen_faces:
                        unseen_faces.remove(neighbor)
                        component.add(neighbor)
                        stack.append(neighbor)
            components.append(component)
        for component in components[1:]:
            duplicate = len(vertices)
            vertices.append(mesh.vertices[vertex].tolist())
            for face in component:
                faces[face, faces[face] == vertex] = duplicate
    return TriangleMesh(np.asarray(vertices), faces)


def close_boundary_loops(mesh: TriangleMesh) -> TriangleMesh:
    """Cap manifold boundary loops on a temporary skeletonization proxy."""

    if len(mesh.faces) == 0:
        raise ValueError("cannot close a mesh without faces")
    directed = np.concatenate(
        [mesh.faces[:, [0, 1]], mesh.faces[:, [1, 2]], mesh.faces[:, [2, 0]]],
        axis=0,
    )
    undirected = np.sort(directed, axis=1)
    _, inverse, counts = np.unique(
        undirected, axis=0, return_inverse=True, return_counts=True
    )
    if np.any(counts > 2):
        raise ValueError("boundary closure requires a manifold mesh")
    boundary = directed[counts[inverse] == 1]
    if len(boundary) == 0:
        return mesh

    boundary_degree = np.bincount(
        boundary.ravel(), minlength=len(mesh.vertices)
    )
    pinched = np.flatnonzero(boundary_degree > 2)
    if len(pinched):
        return close_boundary_loops(
            keep_largest_component(split_nonmanifold_vertices(mesh))
        )

    outgoing = {}
    incoming_count = {}
    for edge_index, (first, second) in enumerate(boundary):
        outgoing.setdefault(int(first), []).append(edge_index)
        incoming_count[int(second)] = incoming_count.get(int(second), 0) + 1
    vertices_on_boundary = set(outgoing).union(incoming_count)
    if any(
        len(outgoing.get(vertex, ())) != incoming_count.get(vertex, 0)
        for vertex in vertices_on_boundary
    ):
        raise ValueError("mesh boundary orientation is inconsistent")

    unused = set(range(len(boundary)))
    cycles = []
    while unused:
        start_edge = next(iter(unused))
        start_vertex = int(boundary[start_edge, 0])
        path_vertices = [start_vertex]
        path_edges = []
        current = start_vertex
        while True:
            choices = [edge for edge in outgoing[current] if edge in unused]
            if not choices:
                raise ValueError("mesh boundary does not close into directed loops")
            edge = choices[0]
            unused.remove(edge)
            path_edges.append(edge)
            current = int(boundary[edge, 1])
            if current in path_vertices:
                cycle_start = path_vertices.index(current)
                unused.update(path_edges[:cycle_start])
                cycles.append(path_edges[cycle_start:])
                break
            path_vertices.append(current)

    vertices = mesh.vertices.tolist()
    faces = mesh.faces.tolist()
    for cycle in cycles:
        cycle_vertices = np.unique(boundary[np.asarray(cycle)].ravel())
        center = len(vertices)
        vertices.append(mesh.vertices[cycle_vertices].mean(axis=0).tolist())
        for edge in cycle:
            first, second = boundary[edge]
            faces.append([int(second), int(first), center])
    closed = TriangleMesh(np.asarray(vertices), np.asarray(faces))
    if not is_watertight(closed):
        raise RuntimeError("boundary-loop closure did not produce a watertight mesh")
    return closed


def keep_largest_component(mesh: TriangleMesh) -> TriangleMesh:
    """Keep the single surface component required by CGAL skeletonization."""

    if len(mesh.faces) == 0:
        return mesh
    vertex_faces = [[] for _ in range(len(mesh.vertices))]
    for face_index, face in enumerate(mesh.faces):
        for vertex in face:
            vertex_faces[int(vertex)].append(face_index)
    unseen = set(range(len(mesh.faces)))
    components = []
    while unseen:
        seed = unseen.pop()
        component = [seed]
        stack = [seed]
        while stack:
            face_index = stack.pop()
            for vertex in mesh.faces[face_index]:
                for neighbor in vertex_faces[int(vertex)]:
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        component.append(neighbor)
                        stack.append(neighbor)
        components.append(component)
    faces = mesh.faces[np.asarray(max(components, key=len), dtype=np.int64)]
    used = np.unique(faces)
    remap = np.full(len(mesh.vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return TriangleMesh(mesh.vertices[used], remap[faces])


def reconstruct_poisson(points, config: MeshConfig = MeshConfig()) -> TriangleMesh:
    """Reconstruct a watertight surface; Open3D stays optional until this call."""

    points = as_points(points)
    if len(points) < 4:
        raise ValueError("Poisson reconstruction needs at least four points")
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError("Poisson reconstruction requires the 'geometry' extra (Open3D)") from exc
    extent = float(np.ptp(points, axis=0).max())
    radius = config.normal_radius or max(extent * 0.02, 1e-6)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius, max_nn=config.normal_neighbors
        )
    )
    cloud.orient_normals_consistent_tangent_plane(config.normal_consistency_neighbors)
    def cleaned_candidate(open3d_mesh):
        if len(open3d_mesh.triangles) > config.target_triangles:
            open3d_mesh = open3d_mesh.simplify_quadric_decimation(
                target_number_of_triangles=config.target_triangles
            )
        open3d_mesh.remove_degenerate_triangles()
        open3d_mesh.remove_duplicated_triangles()
        open3d_mesh.remove_duplicated_vertices()
        open3d_mesh.remove_non_manifold_edges()
        open3d_mesh.remove_unreferenced_vertices()
        open3d_mesh.orient_triangles()
        candidate = keep_largest_component(
            TriangleMesh(
                np.asarray(open3d_mesh.vertices).copy(),
                np.asarray(open3d_mesh.triangles).copy(),
            )
        )
        candidate = keep_largest_component(
            split_nonmanifold_vertices(candidate)
        )
        if not is_watertight(candidate):
            candidate = close_boundary_loops(candidate)
        return candidate

    mesh = None
    minimum_depth = max(2, config.poisson_depth - 4)
    for depth in range(config.poisson_depth, minimum_depth - 1, -1):
        reconstructed, densities = (
            o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                cloud, depth=depth
            )
        )
        candidate = cleaned_candidate(o3d.geometry.TriangleMesh(reconstructed))
        densities = np.asarray(densities)
        if len(densities) and config.density_quantile > 0:
            trimmed_mesh = o3d.geometry.TriangleMesh(reconstructed)
            trimmed_mesh.remove_vertices_by_mask(
                densities < np.quantile(densities, config.density_quantile)
            )
            try:
                trimmed = cleaned_candidate(trimmed_mesh)
            except (ValueError, RuntimeError):
                trimmed = None
            if trimmed is not None and is_watertight(trimmed):
                candidate = trimmed
        if is_watertight(candidate):
            mesh = candidate
            if depth != config.poisson_depth:
                warnings.warn(
                    f"Poisson depth {config.poisson_depth} was not watertight; "
                    f"using depth {depth}",
                    RuntimeWarning,
                )
            break
    if mesh is None:
        raise RuntimeError(
            "Poisson reconstruction did not produce a watertight surface; "
            "provide a watertight canonical mesh"
        )
    return mesh


def prepare_mesh(
    sequence: DynamicPointSequence,
    config: MeshConfig = MeshConfig(),
    external_mesh: Optional[Union[str, Path, TriangleMesh]] = None,
) -> TriangleMesh:
    """Resolve an external, registered, or reconstructed rest-frame surface."""

    if not isinstance(sequence, DynamicPointSequence):
        sequence = DynamicPointSequence(sequence)
    if config.frame >= sequence.num_frames:
        raise ValueError("mesh frame is outside the point sequence")
    if external_mesh is not None:
        mesh = external_mesh if isinstance(external_mesh, TriangleMesh) else load_mesh(external_mesh)
        if not is_watertight(mesh):
            raise ValueError("the supplied canonical mesh must be watertight for curve skeletonization")
        return mesh
    if sequence.faces is not None:
        mesh = TriangleMesh(sequence.points[config.frame], sequence.faces)
        if is_watertight(mesh):
            return mesh
        warnings.warn(
            "input faces are not watertight; reconstructing a closed rest-frame surface",
            RuntimeWarning,
        )
    return reconstruct_poisson(sequence.points[config.frame], config)
