# mini-mesh-simplify

A tiny dependency-free Python 3 triangle-mesh toolkit with a QEM
(quadric error metric) edge-collapse simplifier.  It is small enough to
read in one sitting and is covered by a self-contained assertion suite.

## Requirements

- Python 3.8 or newer
- No third-party packages

## Layout

```text
mini-mesh-simplify/
├── Makefile
├── README.md
├── run_tests.sh
├── src/
│   └── simplify.py
└── tests/
    └── test_simplify.py
```

## API

All coordinates are plain Python tuples/lists of three floats.  Faces are
triangles stored as three integer vertex indices.

### `load_obj(text) -> (verts, faces)`

Parses the common `v` and `f` records of a Wavefront OBJ document.
Polygonal faces are triangulated with a fan, comments are ignored, and
negative (relative) OBJ indices are supported.

### `vertex_normals(verts, faces) -> list[tuple[float, float, float]]`

Returns one normalized area-weighted normal per vertex.  Isolated or
degenerate vertices get the zero vector.

### `quadric_for_vertex(verts, faces, vi) -> 4x4 symmetric matrix`

Returns the quadric matrix accumulated from the unit plane equations of
all faces incident to vertex `vi`.  The matrix is a list of four rows,
each containing four floats.

### `simplify(verts, faces, target_faces, lock_boundary=True) -> (verts2, faces2)`

Runs QEM edge collapse:

1. Build a quadric for every vertex.
2. Evaluate every current edge with the summed endpoint quadrics.
3. Collapse the edge with the smallest QEM error.
4. Remap faces from the removed vertex to the survivor and drop
   degenerate faces.
5. Repeat until `target_faces` is reached or no eligible edge remains.

With `lock_boundary=True`, no edge incident to a boundary vertex is
collapsed.  If `target_faces` is greater than or equal to the input face
count, the input is returned unchanged.  A closed result is uniformly
rescaled about its centroid at the end so an aggressive low-poly result
keeps the original surface area; open meshes are not rescaled, so locked
boundary vertices stay in place.

### `mesh_area(verts, faces) -> float`

Sums the absolute area of every triangle, i.e. one half of the norm of
the triangle's cross product.  Inward/clockwise triangles therefore add
positive area too.

### `is_manifold(faces) -> bool`

Checks the edge condition of a triangle 2-manifold: every undirected edge
is used by one or two triangles and every face has three distinct
vertices.

## Test commands

The canonical command is:

```sh
make test
```

Equivalent commands (run from the project root) are:

```sh
python3 tests/test_simplify.py
```

```sh
./run_tests.sh
```

`run_tests.sh` changes to its own directory first, so it can also be
invoked from elsewhere.

## Test coverage

The suite contains 13 checks:

- OBJ parsing and fan triangulation
- outward, unit-length vertex normals
- quadric symmetry and zero error at the source vertex
- unit-cube area `6.0` and consistent area for an inward-wound triangle
- cube simplification to 6 faces that stays closed and manifold
- five-vertex result and no duplicate triangles for the closed 6-face result
- cube simplified area error below 15%
- planar simplification stays at `z = 0`
- boundary-locked simplification preserves the boundary vertex count
- target larger than input returns the original mesh
