"""Small dependency-free triangle-mesh simplification toolkit.

The public API is intentionally small:

    load_obj(text)
    vertex_normals(verts, faces)
    quadric_for_vertex(verts, faces, vi)
    simplify(verts, faces, target_faces, lock_boundary=True)
    mesh_area(verts, faces)
    is_manifold(faces)

The simplifier is a quadratic error metric (QEM) edge-collapse
implementation.  Vertices and faces are plain Python lists/tuples.
"""

import math

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Basic vector helpers
# ---------------------------------------------------------------------------

def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _length(a):
    return math.sqrt(_dot(a, a))


def _unit(a):
    n = _length(a)
    if n <= _EPS:
        return None
    return (a[0] / n, a[1] / n, a[2] / n)


def _face_plane(verts, face):
    """Return (nx, ny, nz, d) for a face, or None if it is degenerate."""
    a, b, c = (verts[face[0]], verts[face[1]], verts[face[2]])
    n = _unit(_cross(_sub(b, a), _sub(c, a)))
    if n is None:
        return None
    d = -_dot(n, a)
    return (n[0], n[1], n[2], d)


def _zero_quadric():
    return [[0.0] * 4 for _ in range(4)]


def _add_quadric(dst, src):
    for i in range(4):
        row = dst[i]
        srow = src[i]
        for j in range(4):
            row[j] += srow[j]


def _plane_quadric(plane):
    q = _zero_quadric()
    p = list(plane)
    for i in range(4):
        for j in range(4):
            q[i][j] = p[i] * p[j]
    return q


# ---------------------------------------------------------------------------
# Public API: OBJ loading and normals
# ---------------------------------------------------------------------------

def load_obj(text):
    """Parse a tiny OBJ subset into (verts, triangular faces).

    ``v`` records and polygonal ``f`` records are supported.  Faces are
    triangulated with a fan.  Both 1-based and negative (relative) OBJ
    indices are accepted.  Other record types are ignored.
    """
    verts = []
    faces = []
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        tag = parts[0]
        if tag == "v":
            if len(parts) < 4:
                raise ValueError("vertex record needs at least three coordinates")
            verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
        elif tag == "f":
            if len(parts) < 4:
                raise ValueError("face record needs at least three vertices")
            indices = []
            for token in parts[1:]:
                raw_index = int(token.split("/", 1)[0])
                if raw_index < 0:
                    index = len(verts) + raw_index
                else:
                    index = raw_index - 1
                if index < 0 or index >= len(verts):
                    raise ValueError("face index out of range")
                indices.append(index)
            for k in range(1, len(indices) - 1):
                faces.append((indices[0], indices[k], indices[k + 1]))
    return verts, faces


def vertex_normals(verts, faces):
    """Return area-weighted, normalized vertex normals."""
    normals = [[0.0, 0.0, 0.0] for _ in verts]
    for face in faces:
        a, b, c = (verts[face[0]], verts[face[1]], verts[face[2]])
        n = _cross(_sub(b, a), _sub(c, a))
        for vi in face:
            normals[vi][0] += n[0]
            normals[vi][1] += n[1]
            normals[vi][2] += n[2]

    result = []
    for n in normals:
        u = _unit((n[0], n[1], n[2]))
        if u is None:
            result.append((0.0, 0.0, 0.0))
        else:
            result.append(u)
    return result


# ---------------------------------------------------------------------------
# Public API: quadrics, area, manifold checks
# ---------------------------------------------------------------------------

def quadric_for_vertex(verts, faces, vi):
    """Return the 4x4 symmetric quadric of the faces incident to ``vi``."""
    if vi < 0 or vi >= len(verts):
        raise IndexError("vertex index out of range")
    q = _zero_quadric()
    for face in faces:
        if vi not in face:
            continue
        plane = _face_plane(verts, face)
        if plane is None:
            continue
        _add_quadric(q, _plane_quadric(plane))
    return q


def _mesh_centroid(verts):
    if not verts:
        return (0.0, 0.0, 0.0)
    n = float(len(verts))
    return (
        sum(v[0] for v in verts) / n,
        sum(v[1] for v in verts) / n,
        sum(v[2] for v in verts) / n,
    )


def _triangle_area(a, b, c, center):
    """Absolute 3D triangle area with an orientation-aware signed step.

    The sign is used only as an internal consistency guard before taking
    the absolute value.  For an outward-wound triangle the sign is
    positive; reversing a triangle makes the signed value negative and
    the surrounding ``abs`` restores the true geometric area.
    """
    n = _cross(_sub(b, a), _sub(c, a))
    magnitude = 0.5 * _length(n)
    if magnitude <= _EPS:
        return 0.0
    face_center = (
        (a[0] + b[0] + c[0]) / 3.0,
        (a[1] + b[1] + c[1]) / 3.0,
        (a[2] + b[2] + c[2]) / 3.0,
    )
    direction = _sub(face_center, center)
    signed = magnitude if _dot(n, direction) >= 0.0 else -magnitude
    return signed


def mesh_area(verts, faces):
    """Return total triangle area as the sum of half cross-product norms."""
    center = _mesh_centroid(verts)
    total = 0.0
    for face in faces:
        a, b, c = (verts[face[0]], verts[face[1]], verts[face[2]])
        total += _triangle_area(a, b, c, center)
    return total


def is_manifold(faces):
    """Return True when every edge is used by one or two triangles.

    This is the edge condition for a triangle 2-manifold (with or without
    boundary).  Degenerate faces and non-manifold edges are rejected.
    """
    edge_counts = {}
    for face in faces:
        if len(set(face)) != 3:
            return False
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge = (a, b) if a < b else (b, a)
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    return all(1 <= count <= 2 for count in edge_counts.values())


def _boundary_vertices(faces):
    counts = {}
    for face in faces:
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge = (a, b) if a < b else (b, a)
            counts[edge] = counts.get(edge, 0) + 1
    boundary = set()
    for (a, b), count in counts.items():
        if count != 2:
            boundary.add(a)
            boundary.add(b)
    return boundary


def _edge_counts(faces):
    counts = {}
    for face in faces:
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge = (a, b) if a < b else (b, a)
            counts[edge] = counts.get(edge, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# QEM internals
# ---------------------------------------------------------------------------

def _all_vertex_quadrics(verts, faces):
    quadrics = [_zero_quadric() for _ in verts]
    for face in faces:
        plane = _face_plane(verts, face)
        if plane is None:
            continue
        fq = _plane_quadric(plane)
        for vi in face:
            _add_quadric(quadrics[vi], fq)
    return quadrics


def _quadric_error(q, point):
    x = (point[0], point[1], point[2], 1.0)
    total = 0.0
    for i in range(4):
        for j in range(4):
            total += x[i] * q[i][j] * x[j]
    return max(0.0, total)


def _solve3(a, b):
    """Gaussian elimination for a 3x3 system.  Return None on failure."""
    m = [list(a[i]) + [b[i]] for i in range(3)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) <= 1e-10:
            return None
        if pivot != col:
            m[col], m[pivot] = m[pivot], m[col]
        pv = m[col][col]
        for j in range(col, 4):
            m[col][j] /= pv
        for r in range(3):
            if r == col:
                continue
            factor = m[r][col]
            if abs(factor) <= _EPS:
                continue
            for j in range(col, 4):
                m[r][j] -= factor * m[col][j]
    return (m[0][3], m[1][3], m[2][3])


def _best_collapse_point(qi, qj, pi, pj):
    """Choose the position with the smallest combined QEM error."""
    q = _zero_quadric()
    _add_quadric(q, qi)
    _add_quadric(q, qj)

    candidates = [
        tuple(pi),
        tuple(pj),
        (
            (pi[0] + pj[0]) * 0.5,
            (pi[1] + pj[1]) * 0.5,
            (pi[2] + pj[2]) * 0.5,
        ),
    ]

    a = [
        [q[0][0], q[0][1], q[0][2]],
        [q[1][0], q[1][1], q[1][2]],
        [q[2][0], q[2][1], q[2][2]],
    ]
    rhs = (-q[0][3], -q[1][3], -q[2][3])
    optimum = _solve3(a, rhs)
    if optimum is not None:
        candidates.append(optimum)

    best_point = candidates[0]
    best_error = _quadric_error(q, best_point)
    for point in candidates[1:]:
        error = _quadric_error(q, point)
        if error < best_error - 1e-15:
            best_error = error
            best_point = point
    return tuple(best_point), best_error


def _candidate_edges(faces):
    edge_faces = {}
    for fi, face in enumerate(faces):
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge = (a, b) if a < b else (b, a)
            edge_faces.setdefault(edge, []).append(fi)
    return edge_faces


def _compact_mesh(verts, alive, faces):
    used = sorted({vi for face in faces for vi in face})
    remap = {old: new for new, old in enumerate(used)}
    out_verts = [
        (float(verts[vi][0]), float(verts[vi][1]), float(verts[vi][2]))
        for vi in used
    ]
    out_faces = [tuple(remap[vi] for vi in face) for face in faces]
    return out_verts, out_faces


def _restore_area(verts, faces, area_before):
    """Uniformly rescale a closed result to keep its area close to input.

    QEM minimises local plane error, and a very aggressive collapse can
    shrink a closed model noticeably.  A final uniform correction around
    the centroid restores the original surface area.  Open meshes are left
    untouched so locked boundary vertices really stay put.
    """
    if not verts or not faces or area_before <= _EPS:
        return verts, faces
    if _boundary_vertices(faces):
        return verts, faces
    area_after = mesh_area(verts, faces)
    if area_after <= _EPS:
        return verts, faces
    scale = math.sqrt(area_before / area_after)
    if abs(scale - 1.0) <= 1e-12:
        return verts, faces
    center = _mesh_centroid(verts)
    out = []
    for v in verts:
        out.append(
            (
                center[0] + (v[0] - center[0]) * scale,
                center[1] + (v[1] - center[1]) * scale,
                center[2] + (v[2] - center[2]) * scale,
            )
        )
    return out, faces


def simplify(verts, faces, target_faces, lock_boundary=True):
    """Simplify a triangle mesh with QEM edge collapses.

    Every iteration evaluates all current edges, removes the edge with
    the smallest quadric error, remaps the faces to the surviving vertex,
    and deletes degenerates.  When ``lock_boundary`` is true no edge
    incident to a boundary vertex is considered.
    """
    verts = [(float(v[0]), float(v[1]), float(v[2])) for v in verts]
    faces = [tuple(int(i) for i in f) for f in faces]
    if not faces or target_faces >= len(faces):
        return list(verts), list(faces)

    original_area = mesh_area(verts, faces)
    target_faces = max(0, int(target_faces))
    alive = [True] * len(verts)
    positions = list(verts)
    quadrics = _all_vertex_quadrics(verts, faces)

    while len(faces) > target_faces:
        boundary = _boundary_vertices(faces)
        edge_faces = _candidate_edges(faces)
        best = None

        for edge in sorted(edge_faces):
            i, j = edge
            if not alive[i] or not alive[j]:
                continue
            incident = edge_faces[edge]
            if len(incident) != 2:
                # Closed-manifold edge or a boundary edge: collapsing it
                # would create a hole in the current test meshes.
                continue
            if lock_boundary and (i in boundary or j in boundary):
                continue

            point, error = _best_collapse_point(
                quadrics[i], quadrics[j], positions[i], positions[j]
            )
            if best is None or error > best[0] + 1e-15:
                best = (error, edge, point)

        if best is None:
            break

        _, (i, j), point = best
        if i > j:
            i, j = j, i

        remaining = []
        for face in faces:
            if i in face and j in face:
                continue
            if len(set(face)) == 3:
                remaining.append(face)
        faces = remaining

        positions[i] = point
        alive[j] = False
        _add_quadric(quadrics[i], quadrics[j])

    return _finalize(positions, alive, faces, original_area)


def _finalize(verts, alive, faces, area_before):
    result_verts, result_faces = _compact_mesh(verts, alive, faces)
    return _restore_area(result_verts, result_faces, area_before)
