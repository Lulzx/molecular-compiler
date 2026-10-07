import numpy as np
import pytest

from molecular_compiler import SkeletonTree, compile
from molecular_compiler.compiler import ResolutionPolicy

SWC = """# demo
1 1 0 0 0 2 -1
2 3 10 0 0 0.5 1
3 3 20 0 0 0.5 2
4 3 20 10 0 0.4 3
5 3 20 -10 0 0.4 3
"""


def cable(n=40, step=5.0, radius=0.5):
    x = np.arange(n) * step
    xyz = np.stack([x, 0 * x, 0 * x], axis=1)
    return SkeletonTree.from_arrays(
        np.arange(n),
        np.full(n, 3),
        xyz,
        np.full(n, radius),
        np.r_[-1, np.arange(n - 1)],
    )


def binary_tree(depth=5, step=8.0, radius=0.6):
    ids, xyz, rad, par = [0], [[0.0, 0, 0]], [radius], [-1]
    frontier = [(0, 0.0, 0.0, 0.0)]
    for d in range(depth):
        nxt = []
        for p, x, y, _ in frontier:
            for sign in (-1, 1):
                k = len(ids)
                ids.append(k)
                nx, ny = x + step, y + sign * step / (d + 1)
                xyz.append([nx, ny, 0.0])
                rad.append(radius * 0.8 ** (d + 1))
                par.append(p)
                nxt.append((k, nx, ny, 0.0))
        frontier = nxt
    return SkeletonTree.from_arrays(ids, np.full(len(ids), 3), xyz, rad, par)


def test_M6_swc_parse_and_branch_structure():
    tree = SkeletonTree.from_swc(SWC)
    assert tree.n_nodes == 5
    np.testing.assert_allclose(tree.path_um, [0, 10, 20, 30, 30])
    assert list(tree.ids[tree.branch_nodes]) == [3]
    assert sorted(tree.ids[tree.tip_nodes]) == [4, 5]


def test_M6_swc_file_roundtrip(tmp_path):
    f = tmp_path / "n.swc"
    f.write_text(SWC)
    assert SkeletonTree.from_swc(f).n_nodes == 5


@pytest.mark.parametrize(
    "text",
    [
        "",
        "1 1 0 0 0 1 -1\n2 3 1 0 0 1",  # field count
        "1 1 0 0 0 1 -1\n2 3 1 0 0 x 1",  # non numeric
        "1 1 0 0 0 1 -1\n2 3 1 0 0 1 9",  # missing parent
        "1 1 0 0 0 1 -1\n2 3 1 0 0 0 1",  # zero radius
        "1 1 0 0 0 1 -1\n2 3 1 0 0 1 -1",  # two roots
        "1 1 0 0 0 1 2\n2 3 1 0 0 1 1",  # cycle, no root
        "1 1 0 0 0 1 -1\n2 3 1 0 0 1 3\n3 3 2 0 0 1 2",  # cycle off a root
        "1 1 0 0 0 1 -1\n1 3 1 0 0 1 1",  # duplicate id
        "1 1 0 0 0 1 -1\n2 3 0 0 0 1 1",  # zero-length segment
    ],
)
def test_M6_swc_validation_errors(text):
    with pytest.raises(ValueError):
        SkeletonTree.from_swc(text + "\n")


def test_M6_unbranched_cable_matches_path_distance_bins():
    tree = cable()
    lam0 = 20.0
    red = tree.reduce(4, lam0)
    length = lam0 * np.sqrt(2 * 0.5)
    expected = np.minimum(np.floor(tree.path_um / length).astype(int), 3)
    np.testing.assert_array_equal(red.node_compartment, expected)


@pytest.mark.parametrize("make", [cable, binary_tree])
def test_M6_reduction_conserves_area_and_input_resistance(make):
    tree = make()
    red = make().reduce(4, 15.0)
    np.testing.assert_allclose(red.area_um2.sum(), tree.total_area_um2(), rtol=1e-12)
    assert len(red.occupied) > 1
    assert red.input_resistance_ohm == pytest.approx(
        red.tree_input_resistance_ohm, rel=1e-6
    )


def test_M6_tree_input_resistance_matches_dense_solve():
    tree = binary_tree(depth=3)
    ra, rm = 100.0, 20000.0
    child, p, area, g = tree._segments(ra)
    n = tree.n_nodes
    lap = np.zeros((n, n))
    for c, q, gg in zip(child, p, g):
        lap[c, c] += gg
        lap[q, q] += gg
        lap[c, q] -= gg
        lap[q, c] -= gg
    leak = np.zeros(n)
    np.add.at(leak, child, area / 2)
    np.add.at(leak, p, area / 2)
    leak[tree.order[0]] += 4 * np.pi * tree.radius[tree.order[0]] ** 2
    lap += np.diag(leak * 1e-8 / rm)
    rhs = np.zeros(n)
    rhs[tree.order[0]] = 1.0
    dense = np.linalg.solve(lap, rhs)[tree.order[0]]
    assert tree.input_resistance_ohm(ra, rm) == pytest.approx(dense, rel=1e-9)


def test_M6_synapse_mapping_by_id_and_xyz():
    tree = SkeletonTree.from_swc(SWC)
    np.testing.assert_array_equal(tree.map_synapses(nodes=[1, 4]), [0, 3])
    np.testing.assert_array_equal(
        tree.map_synapses(xyz=[[19, 1, 0], [21, -9, 0], [-5, 0, 0]]), [2, 4, 0]
    )
    with pytest.raises(KeyError):
        tree.map_synapses(nodes=[99])
    with pytest.raises(ValueError):
        tree.map_synapses()


def _straight_skeletons(graph):
    syn = graph.connectome.synapses.to_pylist()
    out = {}
    for r in syn:
        x = r["xyz"][0]
        # Synapse sits at path 5 um on a 1 um diameter cable, with a long tail.
        nodes = [(x - r["path_dist_post"], 0.0, 0.0), (x, 0.0, 0.0), (x + 50, 0.0, 0.0)]
        out[r["post_id"]] = SkeletonTree.from_arrays(
            [0, 1, 2], [1, 3, 3], nodes, [0.5] * 3, [-1, 0, 1]
        )
    return out


def test_M6_compile_with_straight_skeleton_matches_supplied_values(system):
    graph, rules, kinetics = system
    base = compile(graph, rules, kinetics)
    sk = compile(graph, rules, kinetics, skeletons=_straight_skeletons(graph))
    for key in ("attenuation", "compartment", "gate"):
        np.testing.assert_array_equal(sk.syn_params[key], base.syn_params[key])
    assert sk.syn_params["compartment"].shape == base.syn_params["compartment"].shape


def test_M6_compile_with_branched_skeleton_changes_geometry(system):
    graph, rules, kinetics = system
    policy = ResolutionPolicy.default("C. elegans")
    base = compile(graph, rules, kinetics, resolution=policy)
    skeletons = {}
    for r in graph.connectome.synapses.to_pylist():
        x = r["xyz"][0]
        # Thin far branch: synapse 200 um out on a 0.2 um radius process.
        skeletons[r["post_id"]] = SkeletonTree.from_arrays(
            [0, 1, 2],
            [1, 3, 3],
            [(x - 200, 0, 0), (x - 100, 0, 0), (x, 0, 0)],
            [1.0, 0.2, 0.2],
            [-1, 0, 1],
        )
    sk = compile(graph, rules, kinetics, resolution=policy, skeletons=skeletons)
    assert np.all(
        np.asarray(sk.syn_params["attenuation"])
        < np.asarray(base.syn_params["attenuation"])
    )
    assert np.all(
        np.asarray(sk.syn_params["compartment"])
        >= np.asarray(base.syn_params["compartment"])
    )
    assert np.any(
        np.asarray(sk.syn_params["compartment"])
        > np.asarray(base.syn_params["compartment"])
    )


def test_M6_compile_default_has_no_skeleton_effect(system):
    graph, rules, kinetics = system
    a = compile(graph, rules, kinetics)
    b = compile(graph, rules, kinetics, skeletons=None)
    c = compile(graph, rules, kinetics, skeletons={})
    for other in (b, c):
        for key in a.syn_params:
            np.testing.assert_array_equal(other.syn_params[key], a.syn_params[key])
