"""Small self-contained test suite for src/simplify.py.

Run with either of:

    python3 tests/test_simplify.py
    make test
"""

import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.simplify import (
    is_manifold,
    load_obj,
    mesh_area,
    quadric_for_vertex,
    simplify,
    vertex_normals,
)

TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


CUBE_VERTS = [
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (1.0, 1.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 1.0),
    (1.0, 1.0, 1.0),
    (0.0, 1.0, 1.0),
]

CUBE_FACES = [
    (0, 2, 1),
    (0, 3, 2),
    (4, 5, 6),
    (4, 6, 7),
    (0, 1, 5),
    (0, 5, 4),
    (1, 2, 6),
    (1, 6, 5),
    (2, 3, 7),
    (2, 7, 6),
    (3, 0, 4),
    (3, 4, 7),
]


def _edge_counts(faces):
    counts = {}
    for face in faces:
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge = (a, b) if a < b else (b, a)
            counts[edge] = counts.get(edge, 0) + 1
    return counts


def _boundary_vertex_count(faces):
    counts = _edge_counts(faces)
    boundary = set()
    for (a, b), count in counts.items():
        if count != 2:
            boundary.add(a)
            boundary.add(b)
    return len(boundary)


def _planar_grid(n=4):
    verts = [(float(x), float(y), 0.0) for y in range(n) for x in range(n)]
    faces = []
    for y in range(n - 1):
        for x in range(n - 1):
            a = y * n + x
            b = a + 1
            c = a + n + 1
            d = a + n
            faces.append((a, b, c))
            faces.append((a, c, d))
    return verts, faces


@test
def test_load_obj_triangulates_quad():
    text = """
    # tiny pyramid with a quad base
    v 0 0 0
    v 1 0 0
    v 1 1 0
    v 0 1 0
    v 0.5 0.5 1
    f 1 2 3 4
    f 1 2 5
    f 2 3 5
    f 3 4 5
    f 4 1 5
    """
    verts, faces = load_obj(text)
    assert len(verts) == 5, "expected five vertices"
    assert len(faces) == 6, "quad base must be split into two triangles"
    assert verts[4] == (0.5, 0.5, 1.0), "vertex coordinates parsed incorrectly"


@test
def test_cube_vertex_normals_point_outward():
    normals = vertex_normals(CUBE_VERTS, CUBE_FACES)
    assert len(normals) == 8, "one normal per vertex expected"
    for normal in normals:
        assert abs(math.sqrt(sum(c * c for c in normal)) - 1.0) < 1e-9
    for axis in range(3):
        assert normals[0][axis] < 0.0, "corner normal should point away from cube"
        assert normals[6][axis] > 0.0, "opposite corner normal is wrong"


@test
def test_quadric_for_vertex_is_symmetric_and_zero_at_vertex():
    q = quadric_for_vertex(CUBE_VERTS, CUBE_FACES, 0)
    assert len(q) == 4 and all(len(row) == 4 for row in q)
    for i in range(4):
        for j in range(4):
            assert abs(q[i][j] - q[j][i]) < 1e-12, "quadric must be symmetric"
    x = (CUBE_VERTS[0][0], CUBE_VERTS[0][1], CUBE_VERTS[0][2], 1.0)
    error = sum(x[i] * q[i][j] * x[j] for i in range(4) for j in range(4))
    assert abs(error) < 1e-9, "incident planes contain the vertex"


@test
def test_mesh_area_cube_and_inward_triangle():
    area = mesh_area(CUBE_VERTS, CUBE_FACES)
    assert abs(area - 6.0) < 1e-9, "unit cube area must be 6"

    reversed_faces = list(CUBE_FACES)
    reversed_faces[0] = (CUBE_FACES[0][0], CUBE_FACES[0][2], CUBE_FACES[0][1])
    area_with_flip = mesh_area(CUBE_VERTS, reversed_faces)
    assert abs(area_with_flip - 6.0) < 1e-9, "one inward triangle must not subtract area"


@test
def test_cube_simplifies_to_six_faces():
    verts2, faces2 = simplify(CUBE_VERTS, CUBE_FACES, 6)
    assert len(faces2) == 6, "cube must reach the six-face target"


@test
def test_cube_simplified_result_is_closed():
    _, faces2 = simplify(CUBE_VERTS, CUBE_FACES, 6)
    counts = _edge_counts(faces2)
    assert counts, "simplified mesh must contain faces"
    assert all(count == 2 for count in counts.values()), "closed mesh needs two faces per edge"


@test
def test_cube_simplified_result_is_manifold():
    _, faces2 = simplify(CUBE_VERTS, CUBE_FACES, 6)
    assert is_manifold(faces2), "QEM collapse must preserve manifoldness"


@test
def test_cube_simplified_vertex_count_matches_euler():
    verts2, faces2 = simplify(CUBE_VERTS, CUBE_FACES, 6)
    assert len(verts2) == 5, "six closed triangles need five vertices (V-E+F=2)"


@test
def test_cube_simplified_faces_are_distinct():
    _, faces2 = simplify(CUBE_VERTS, CUBE_FACES, 6)
    seen = set()
    for face in faces2:
        key = tuple(sorted(face))
        assert key not in seen, "duplicate triangle in simplified mesh"
        seen.add(key)


@test
def test_cube_simplified_area_error_under_15_percent():
    original = mesh_area(CUBE_VERTS, CUBE_FACES)
    verts2, faces2 = simplify(CUBE_VERTS, CUBE_FACES, 6)
    simplified = mesh_area(verts2, faces2)
    relative_error = abs(simplified - original) / original
    assert relative_error < 0.15, "area error %.4f exceeds 15%%" % relative_error


@test
def test_planar_simplification_stays_flat():
    verts, faces = _planar_grid()
    verts2, faces2 = simplify(verts, faces, 4, lock_boundary=True)
    assert len(faces2) < len(faces), "planar mesh should be simplified"
    assert all(abs(v[2]) < 1e-12 for v in verts2), "all vertices must stay in z=0 plane"


@test
def test_boundary_lock_keeps_boundary_vertex_count():
    verts, faces = _planar_grid()
    before = _boundary_vertex_count(faces)
    _, faces2 = simplify(verts, faces, 4, lock_boundary=True)
    after = _boundary_vertex_count(faces2)
    assert before == 12
    assert after == before, "boundary vertices must not be collapsed"


@test
def test_target_larger_than_input_returns_original():
    verts, faces = _planar_grid()
    verts2, faces2 = simplify(verts, faces, len(faces) + 5)
    assert verts2 == verts, "vertices should be returned unchanged"
    assert faces2 == faces, "faces should be returned unchanged"


def main():
    failures = 0
    for fn in TESTS:
        name = fn.__name__
        try:
            fn()
        except AssertionError as exc:
            failures += 1
            print("FAIL %s: %s" % (name, exc))
        except Exception as exc:  # unexpected errors are still failures
            failures += 1
            print("ERROR %s: %s: %s" % (name, type(exc).__name__, exc))
        else:
            print("PASS %s" % name)

    print("checks=%d failures=%d" % (len(TESTS), failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
