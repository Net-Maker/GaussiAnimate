"""Robust surface skin-weight transfer with biharmonic inpainting.

The formulation follows Robust Skin Weights Transfer via Weight Inpainting
(Abdrashitov et al., 2023). The reference implementation is MIT licensed;
see THIRD_PARTY_NOTICES.md.
"""

import numpy as np


def _normalize(weights):
    weights = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    return weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)


def transfer_skin_weights(
    source_vertices,
    source_faces,
    source_weights,
    target_vertices,
    target_faces,
    *,
    search_radius_scale=0.05,
    normal_threshold_degrees=30.0,
    smooth_steps=10,
    smooth_alpha=0.2,
    return_diagnostics=False,
):
    """Transfer weights between triangle meshes and fill unmatched regions."""

    import igl
    import scipy.sparse

    source_vertices = np.asarray(source_vertices, dtype=np.float64)
    source_faces = np.asarray(source_faces, dtype=np.int32)
    source_weights = np.asarray(source_weights, dtype=np.float64)
    target_vertices = np.asarray(target_vertices, dtype=np.float64)
    target_faces = np.asarray(target_faces, dtype=np.int32)

    source_normals = igl.per_vertex_normals(source_vertices, source_faces)
    target_normals = igl.per_vertex_normals(target_vertices, target_faces)
    squared_distance, face_indices, closest = igl.point_mesh_squared_distance(
        target_vertices, source_vertices, source_faces
    )
    triangles = source_faces[face_indices]
    tri0 = source_vertices[triangles[:, 0]]
    tri1 = source_vertices[triangles[:, 1]]
    tri2 = source_vertices[triangles[:, 2]]
    barycentric_fn = (
        igl.barycentric_coordinates_tri
        if hasattr(igl, "barycentric_coordinates_tri")
        else igl.barycentric_coordinates
    )
    barycentric = barycentric_fn(
        np.ascontiguousarray(closest),
        np.ascontiguousarray(tri0),
        np.ascontiguousarray(tri1),
        np.ascontiguousarray(tri2),
    )

    copied = np.einsum(
        "vi,vib->vb", barycentric, source_weights[triangles]
    )
    interpolated_normals = np.einsum(
        "vi,vic->vc", barycentric, source_normals[triangles]
    )
    interpolated_normals /= np.maximum(
        np.linalg.norm(interpolated_normals, axis=1, keepdims=True), 1e-12
    )
    normalized_target_normals = target_normals / np.maximum(
        np.linalg.norm(target_normals, axis=1, keepdims=True), 1e-12
    )
    angles = np.degrees(
        np.arccos(
            np.clip(
                np.sum(interpolated_normals * normalized_target_normals, axis=1),
                -1.0,
                1.0,
            )
        )
    )
    radius = float(search_radius_scale) * float(
        igl.bounding_box_diagonal(target_vertices)
    )
    matched = (squared_distance <= radius * radius) & (
        angles <= float(normal_threshold_degrees)
    )
    if not matched.any():
        raise RuntimeError("robust weight transfer found no valid surface matches")

    laplacian = 2.0 * igl.cotmatrix(target_vertices, target_faces)
    adjacency_matrix = igl.adjacency_matrix(target_faces)
    component_count, component_labels = scipy.sparse.csgraph.connected_components(
        adjacency_matrix,
        directed=False,
    )
    component_anchors = np.zeros(len(target_vertices), dtype=bool)
    for component in range(component_count):
        vertices = np.flatnonzero(component_labels == component)
        if not np.any(matched[vertices]):
            anchor = vertices[np.argmin(squared_distance[vertices])]
            matched[anchor] = True
            component_anchors[anchor] = True

    mass = igl.massmatrix(
        target_vertices, target_faces, igl.MASSMATRIX_TYPE_VORONOI
    )
    inverse_mass = scipy.sparse.diags(1.0 / mass.diagonal())
    system = -laplacian + laplacian @ inverse_mass @ laplacian
    rhs = np.zeros((len(target_vertices), source_weights.shape[1]), dtype=np.float64)
    known = np.flatnonzero(matched).astype(np.int64)
    constraints = copied[matched]
    empty_matrix = scipy.sparse.csc_matrix((0, 0), dtype=np.float64)
    empty_rhs = np.zeros((0, 0), dtype=np.float64)
    inpainted = igl.min_quad_with_fixed(
        system,
        rhs,
        known,
        constraints,
        empty_matrix,
        empty_rhs,
        True,
    )
    if isinstance(inpainted, tuple):
        success, inpainted = inpainted
        if not success:
            raise RuntimeError("robust weight inpainting failed")
    if not np.isfinite(inpainted).all():
        raise RuntimeError("robust weight inpainting produced non-finite weights")

    adjacency = igl.adjacency_list(target_faces)
    vertices_to_smooth = ~matched
    for seed in np.flatnonzero(~matched):
        queue = [int(seed)]
        while queue:
            vertex = queue.pop()
            for neighbor in adjacency[vertex]:
                if (
                    not vertices_to_smooth[neighbor]
                    and np.linalg.norm(
                        target_vertices[seed] - target_vertices[neighbor]
                    )
                    < radius
                ):
                    vertices_to_smooth[neighbor] = True
                    queue.append(neighbor)

    smoothed = np.array(inpainted, copy=True)
    for _ in range(int(smooth_steps)):
        for vertex in np.flatnonzero(vertices_to_smooth):
            neighbors = adjacency[vertex]
            if not neighbors:
                continue
            neighbor_mean = smoothed[np.asarray(neighbors, dtype=np.int64)].mean(axis=0)
            smoothed[vertex] = (
                (1.0 - smooth_alpha) * smoothed[vertex]
                + smooth_alpha * neighbor_mean
            )
    output = (
        _normalize(smoothed).astype(np.float32),
        matched,
        vertices_to_smooth,
    )
    if return_diagnostics:
        return output + (
            {
                "component_count": int(component_count),
                "component_anchors": component_anchors,
            },
        )
    return output
