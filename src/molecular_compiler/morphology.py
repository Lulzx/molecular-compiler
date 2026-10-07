"""M6 morphology: SWC skeleton trees and compile-time compartment reduction (NG3).

Units: coordinates and radii in um, Ra in ohm cm, Rm in ohm cm^2, results in SI
(S, ohm) except areas (um^2).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

UM_PER_CM = 1e4
CM2_PER_UM2 = 1e-8


@dataclass(frozen=True)
class Reduction:
    """A tree reduced to at most n_comp electrotonic compartments.

    The chain runs over `occupied` compartments in order; `axial_S[k]` joins
    occupied[k] and occupied[k+1]. Areas are conserved exactly. Axial
    conductances share one scale (`axial_scale`) chosen so the chain's soma
    input resistance equals the tree's. That is exact when at least two
    compartments are occupied and the tree value lies between the isopotential
    and soma-only bounds; otherwise compare the two stored resistances.
    """

    n_comp: int
    node_compartment: np.ndarray
    node_electrotonic: np.ndarray
    area_um2: np.ndarray
    occupied: np.ndarray
    axial_S: np.ndarray
    axial_scale: float
    input_resistance_ohm: float
    tree_input_resistance_ohm: float


def _chain_input_resistance(leak, axial):
    """Soma input resistance of a leaky chain, root first."""
    y = leak[-1]
    for k in range(len(axial) - 1, -1, -1):
        y = leak[k] + axial[k] * y / (axial[k] + y)
    return 1.0 / y


@dataclass(frozen=True)
class SkeletonTree:
    ids: np.ndarray
    types: np.ndarray
    xyz: np.ndarray
    radius: np.ndarray
    parent: np.ndarray  # index of parent node, -1 for the root
    order: np.ndarray  # parents before children
    path_um: np.ndarray  # path distance from the root along the tree

    @staticmethod
    def from_arrays(ids, types, xyz, radius, parent_ids):
        ids = np.asarray(ids, dtype=np.int64)
        types = np.asarray(types, dtype=np.int64)
        xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
        radius = np.asarray(radius, dtype=float)
        parent_ids = np.asarray(parent_ids, dtype=np.int64)
        n = len(ids)
        if n == 0 or not (
            len(types) == len(xyz) == len(radius) == len(parent_ids) == n
        ):
            raise ValueError("skeleton arrays must be non-empty and equal length")
        if len(np.unique(ids)) != n:
            raise ValueError("skeleton node ids must be unique")
        if not (np.isfinite(xyz).all() and np.isfinite(radius).all()):
            raise ValueError("skeleton coordinates and radii must be finite")
        if np.any(radius <= 0):
            raise ValueError("skeleton radii must be positive")
        index = {int(i): k for k, i in enumerate(ids)}
        parent = np.full(n, -1, dtype=np.int64)
        for k, p in enumerate(parent_ids):
            if p == -1:
                continue
            if int(p) not in index:
                raise ValueError(f"node {ids[k]}: parent {p} does not exist")
            parent[k] = index[int(p)]
        roots = np.flatnonzero(parent < 0)
        if len(roots) != 1:
            raise ValueError("skeleton must have exactly one root")
        children = [[] for _ in range(n)]
        for k in range(n):
            if parent[k] >= 0:
                children[parent[k]].append(k)
        order, stack = [], [int(roots[0])]
        while stack:
            k = stack.pop()
            order.append(k)
            stack.extend(children[k])
        if len(order) != n:
            raise ValueError("skeleton contains a cycle")
        order = np.asarray(order)
        child = parent >= 0
        length = np.zeros(n)
        length[child] = np.linalg.norm(xyz[child] - xyz[parent[child]], axis=1)
        if np.any(length[child] <= 0):
            raise ValueError("skeleton segments must have positive length")
        path = np.zeros(n)
        for k in order[1:]:
            path[k] = path[parent[k]] + length[k]
        return SkeletonTree(ids, types, xyz, radius, parent, order, path)

    @staticmethod
    def from_swc(source):
        """Parse SWC (id type x y z radius parent) from a file path or text."""
        text = str(source)
        if isinstance(source, Path) or "\n" not in text:
            text = Path(source).read_text()
        rows = []
        for number, line in enumerate(text.splitlines(), 1):
            fields = line.split("#", 1)[0].split()
            if not fields:
                continue
            if len(fields) != 7:
                raise ValueError(f"SWC line {number}: expected 7 fields")
            try:
                values = [float(f) for f in fields]
            except ValueError:
                raise ValueError(f"SWC line {number}: non-numeric field") from None
            if any(values[i] != int(values[i]) for i in (0, 1, 6)):
                raise ValueError(
                    f"SWC line {number}: id, type, parent must be integers"
                )
            rows.append(values)
        if not rows:
            raise ValueError("SWC source has no nodes")
        a = np.asarray(rows)
        return SkeletonTree.from_arrays(a[:, 0], a[:, 1], a[:, 2:5], a[:, 5], a[:, 6])

    @property
    def n_nodes(self):
        return len(self.ids)

    def _child_counts(self):
        return np.bincount(self.parent[self.parent >= 0], minlength=self.n_nodes)

    @property
    def branch_nodes(self):
        """Indices of nodes with two or more children."""
        return np.flatnonzero(self._child_counts() >= 2)

    @property
    def tip_nodes(self):
        return np.flatnonzero(self._child_counts() == 0)

    def node_index(self, node_id):
        hit = np.flatnonzero(self.ids == node_id)
        if len(hit) == 0:
            raise KeyError(f"no node with id {node_id}")
        return int(hit[0])

    def nearest_node(self, xyz):
        """Index of the node closest to xyz (ties go to the lower index)."""
        return int(
            np.argmin(np.linalg.norm(self.xyz - np.asarray(xyz, dtype=float), axis=1))
        )

    def _segments(self, ra_ohm_cm):
        """Per non-root node: parent, segment area (um^2), axial conductance (S)."""
        child = np.flatnonzero(self.parent >= 0)
        p = self.parent[child]
        length = self.path_um[child] - self.path_um[p]
        r1, r2 = self.radius[p], self.radius[child]
        area = np.pi * (r1 + r2) * np.sqrt(length**2 + (r1 - r2) ** 2)
        # Frustum axial resistance Ra L / (pi r1 r2).
        return child, p, area, np.pi * r1 * r2 / (ra_ohm_cm * UM_PER_CM * length)

    def _sphere_um2(self):
        return 4 * np.pi * self.radius[self.order[0]] ** 2  # single-point soma

    def total_area_um2(self):
        return float(self._segments(1.0)[2].sum() + self._sphere_um2())

    def input_resistance_ohm(self, ra_ohm_cm=100.0, rm_ohm_cm2=20000.0):
        """Exact soma input resistance of the passive tree (leaf-to-root recursion)."""
        child, p, area, g = self._segments(ra_ohm_cm)
        node_area = np.zeros(self.n_nodes)
        np.add.at(node_area, child, area / 2)
        np.add.at(node_area, p, area / 2)
        node_area[self.order[0]] += self._sphere_um2()
        g_up = np.zeros(self.n_nodes)
        g_up[child] = g
        y = node_area * CM2_PER_UM2 / rm_ohm_cm2
        for k in self.order[:0:-1]:
            y[self.parent[k]] += g_up[k] * y[k] / (g_up[k] + y[k])
        return float(1.0 / y[self.order[0]])

    def electrotonic_distance(self, lambda0_um):
        """X(node) = integral of ds / lambda, with lambda = lambda0 sqrt(diameter)."""
        inverse = 1 / (lambda0_um * np.sqrt(np.maximum(2 * self.radius, 1e-6)))
        x = np.zeros(self.n_nodes)
        for k in self.order[1:]:
            p = self.parent[k]
            x[k] = (
                x[p]
                + (self.path_um[k] - self.path_um[p]) * (inverse[p] + inverse[k]) / 2
            )
        return x

    def reduce(self, n_comp, lambda0_um, ra_ohm_cm=100.0, rm_ohm_cm2=20000.0):
        """Bin by electrotonic distance: compartment = min(floor(X), n_comp - 1).

        For a constant-diameter cable this equals floor(path / lambda), the bin
        the compiler uses for supplied path distances. Each segment belongs to
        the compartment of its electrotonic midpoint; membrane area is summed
        per compartment, and axial links join area-weighted centroids of the
        cumulative axial resistance to the soma.
        """
        if min(n_comp, lambda0_um, ra_ohm_cm, rm_ohm_cm2) <= 0:
            raise ValueError(
                "reduction needs positive n_comp, length and resistivities"
            )
        x = self.electrotonic_distance(lambda0_um)
        node_comp = np.minimum(np.floor(x).astype(np.int64), n_comp - 1)
        child, p, seg_area, g = self._segments(ra_ohm_cm)
        seg_comp = np.minimum(
            np.floor((x[child] + x[p]) / 2).astype(np.int64), n_comp - 1
        )
        area = np.bincount(seg_comp, weights=seg_area, minlength=n_comp)
        area[0] += self._sphere_um2()
        g_up = np.zeros(self.n_nodes)
        g_up[child] = g
        r_node = np.zeros(self.n_nodes)
        for k in self.order[1:]:
            r_node[k] = r_node[self.parent[k]] + 1 / g_up[k]
        r_mid = (r_node[child] + r_node[p]) / 2
        occupied = np.flatnonzero(area > 0)
        centroid = np.bincount(seg_comp, weights=seg_area * r_mid, minlength=n_comp)
        centroid = centroid[occupied] / area[occupied]
        link = np.maximum(np.diff(centroid), 1e-12 * max(centroid.max(), 1e-30))
        leak = area[occupied] * CM2_PER_UM2 / rm_ohm_cm2
        target = self.input_resistance_ohm(ra_ohm_cm, rm_ohm_cm2)
        scale = 1.0
        if len(occupied) > 1:
            # Input resistance increases with axial resistance: bisect the scale.
            lo, hi = -30.0, 30.0
            for _ in range(200):
                mid = (lo + hi) / 2
                if _chain_input_resistance(leak, 1 / (link * 10**mid)) < target:
                    lo = mid
                else:
                    hi = mid
            scale = 10 ** ((lo + hi) / 2)
        axial = 1 / (link * scale)
        return Reduction(
            n_comp,
            node_comp,
            x,
            area,
            occupied,
            axial,
            float(scale),
            float(_chain_input_resistance(leak, axial)),
            float(target),
        )

    def map_synapses(self, nodes=None, xyz=None):
        """Node indices for synapses given by node id (`nodes`) or position (`xyz`)."""
        if (nodes is None) == (xyz is None):
            raise ValueError("give exactly one of node ids or xyz")
        if nodes is not None:
            return np.array([self.node_index(i) for i in nodes], dtype=np.int64)
        return np.array(
            [self.nearest_node(q) for q in np.asarray(xyz, float).reshape(-1, 3)]
        )
