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

import heapq
import math
from itertools import count

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


def _triangle_area(a, b, c):
    """Absolute 3D triangle area: half the cross-product norm."""
    n = _cross(_sub(b, a), _sub(c, a))
    return 0.5 * _length(n)


def mesh_area(verts, faces):
    """Return total triangle area as the sum of half cross-product norms.

    Every triangle contributes its unsigned area, so winding direction
    (including an occasional inward-facing triangle) never subtracts.
    """
    total = 0.0
    for face in faces:
        a, b, c = (verts[face[0]], verts[face[1]], verts[face[2]])
        total += _triangle_area(a, b, c)
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


def _edge_key(a, b):
    return (a, b) if a < b else (b, a)


def _face_edges(face):
    a, b, c = face
    return (_edge_key(a, b), _edge_key(b, c), _edge_key(c, a))


def _compact_mesh(verts, faces):
    """Remap a working mesh onto dense indices with no dangling refs."""
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

    A binary heap holds candidate edges ordered by smallest quadric
    error; entries are validated lazily against the current mesh.  A
    collapse deletes the one or two faces sharing the collapsed edge,
    remaps every other reference from the dying vertex onto the
    survivor, and refreshes the survivor's edge candidates.  When
    ``lock_boundary`` is true no edge incident to a boundary vertex is
    collapsed, so the boundary vertex set is preserved exactly.
    """
    verts = [(float(v[0]), float(v[1]), float(v[2])) for v in verts]
    faces = [tuple(int(i) for i in f) for f in faces]
    if not faces or target_faces >= len(faces):
        return list(verts), list(faces)

    target_faces = max(0, int(target_faces))
    original_area = mesh_area(verts, faces)

    nverts = len(verts)
    alive = [True] * nverts
    positions = list(verts)
    quadrics = _all_vertex_quadrics(verts, faces)

    # Mutable topology: face lists, per-vertex face sets, edge -> face sets.
    face_list = [list(f) for f in faces]
    vface = [set() for _ in range(nverts)]
    edge_faces = {}
    for fi, face in enumerate(face_list):
        for vi in face:
            vface[vi].add(fi)
        for edge in _face_edges(face):
            edge_faces.setdefault(edge, set()).add(fi)
    face_count = len(face_list)

    def remove_face(fi):
        face = face_list[fi]
        for edge in _face_edges(face):
            users = edge_faces[edge]
            users.discard(fi)
            if not users:
                del edge_faces[edge]
        for vi in face:
            vface[vi].discard(fi)

    def repoint_face(fi, old, new):
        """Move one vertex of a live face; keep all index sets coherent."""
        face = face_list[fi]
        for edge in _face_edges(face):
            users = edge_faces[edge]
            users.discard(fi)
            if not users:
                del edge_faces[edge]
        vface[old].discard(fi)
        face[face.index(old)] = new
        for edge in _face_edges(face):
            edge_faces.setdefault(edge, set()).add(fi)
        vface[new].add(fi)

    def is_boundary_vertex(vi):
        # Only edges actually incident to vi count: an interior vertex
        # has neighbours whose faces touch the mesh boundary, but none
        # of *its own* edges is a single-use boundary edge.
        for fi in vface[vi]:
            face = face_list[fi]
            for edge in _face_edges(face):
                if vi in edge and len(edge_faces[edge]) == 1:
                    return True
        return False

    heap = []
    serial = count()

    def push_edge(u, v):
        if not alive[u] or not alive[v]:
            return
        edge = _edge_key(u, v)
        if edge not in edge_faces:
            return
        point, error = _best_collapse_point(
            quadrics[u], quadrics[v], positions[u], positions[v]
        )
        heapq.heappush(heap, (error, next(serial), u, v, point))

    for edge in edge_faces:
        push_edge(edge[0], edge[1])

    while face_count > target_faces and heap:
        error, _, u, v, _ = heapq.heappop(heap)
        if not alive[u] or not alive[v]:
            continue
        if _edge_key(u, v) not in edge_faces:
            continue

        # The cached error predates later collapses near this edge.
        # Recompute against current quadrics and requeue the entry when
        # the key drifted; otherwise the entry is current and wins.
        # The tight scale-free tolerance lets exact planar (zero-error)
        # edges pass while still catching genuinely stale keys.
        fresh_point, fresh_error = _best_collapse_point(
            quadrics[u], quadrics[v], positions[u], positions[v]
        )
        tolerance = 1e-14 * (1.0 + abs(error) + abs(fresh_error))
        if abs(fresh_error - error) > tolerance:
            heapq.heappush(
                heap, (fresh_error, next(serial), u, v, fresh_point)
            )
            continue

        if lock_boundary and (is_boundary_vertex(u) or is_boundary_vertex(v)):
            continue

        shared = vface[u] & vface[v]
        if len(shared) > 2:
            # Non-manifold edge: collapsing it is not safe.
            continue
        if len(shared) == 2:
            # Link condition: the two opposite vertices must not already
            # share an edge, otherwise the collapse folds the surface.
            thirds = []
            for fi in shared:
                face = face_list[fi]
                thirds.append(next(k for k in face if k != u and k != v))
            if thirds[0] == thirds[1] or _edge_key(thirds[0], thirds[1]) in edge_faces:
                continue

        # Perform the collapse; u survives, v dies.  Every surviving face
        # that referenced v is remapped onto u -- no stale indices remain.
        for fi in sorted(shared):
            remove_face(fi)
            face_count -= 1
        for fi in sorted(vface[v]):
            repoint_face(fi, v, u)

        # Two remapped faces could now coincide: drop exact duplicates so
        # the result stays a set of distinct, manifold triangles.
        seen = {}
        duplicates = []
        for fi in list(vface[u]):
            key = tuple(sorted(face_list[fi]))
            if key in seen:
                duplicates.append(fi)
            else:
                seen[key] = fi
        for fi in duplicates:
            remove_face(fi)
            face_count -= 1

        alive[v] = False
        positions[u] = fresh_point
        _add_quadric(quadrics[u], quadrics[v])

        neighbors = set()
        for fi in vface[u]:
            for k in face_list[fi]:
                if k != u and alive[k]:
                    neighbors.add(k)
        for k in neighbors:
            push_edge(u, k)

    live_face_ids = sorted(
        {fi for users in edge_faces.values() for fi in users}
    )
    out_faces = [tuple(face_list[fi]) for fi in live_face_ids]
    return _finalize(positions, out_faces, original_area)


def _finalize(positions, faces, area_before):
    result_verts, result_faces = _compact_mesh(positions, faces)
    return _restore_area(result_verts, result_faces, area_before)
