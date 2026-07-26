import unittest

import numpy as np

from football_ai.calibration.global_frame_graph import (
    FrameGraphEdge,
    FrameGraphNode,
    GroundDirectionConstraint,
    select_maximum_quality_tree,
    solve_global_frame_graph,
)
from football_ai.calibration.ground_line_evidence import GroundLineFamily


class GlobalFrameGraphTests(unittest.TestCase):
    @staticmethod
    def _edge(source: str, target: str, dx: float) -> FrameGraphEdge:
        matrix = np.asarray(((1.0, 0.0, dx), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
        return FrameGraphEdge(source, target, matrix, 100, 90, 0.9, 0.3, 0.3, 1.0)

    def test_uses_redundant_edges_to_solve_every_connected_node(self) -> None:
        nodes = tuple(FrameGraphNode(str(i), i, float(i)) for i in range(3))
        edges = (self._edge("0", "1", 10.0), self._edge("1", "2", 10.0), self._edge("0", "2", 20.2))
        result = solve_global_frame_graph(nodes, edges, "0")
        self.assertEqual(result.connected_nodes, ("0", "1", "2"))
        point = np.asarray((0.0, 0.0, 1.0))
        mapped = result.node_to_reference["2"] @ point
        mapped /= mapped[2]
        self.assertAlmostEqual(mapped[0], -20.0, delta=0.3)

    def test_reports_disconnected_nodes(self) -> None:
        nodes = (FrameGraphNode("a", 1, 0.0), FrameGraphNode("b", 2, 1.0), FrameGraphNode("c", 3, 2.0))
        result = solve_global_frame_graph(nodes, (self._edge("a", "b", 5.0),), "a")
        self.assertEqual(result.rejected_nodes, ("c",))

    def test_quality_tree_keeps_all_nodes_without_cycle(self) -> None:
        nodes = tuple(FrameGraphNode(str(i), i, float(i)) for i in range(4))
        edges = (
            self._edge("0", "1", 5.0),
            self._edge("1", "2", 5.0),
            self._edge("2", "3", 5.0),
            self._edge("0", "2", 11.0),
        )
        tree = select_maximum_quality_tree(nodes, edges)
        self.assertEqual(len(tree), 3)
        result = solve_global_frame_graph(nodes, tree, "0", _pruning_rounds=0)
        self.assertEqual(len(result.connected_nodes), 4)

    def test_direction_constraint_moves_vanishing_point_toward_white_line(self) -> None:
        nodes = (FrameGraphNode("a", 1, 0.0), FrameGraphNode("b", 2, 1.0))
        edge = self._edge("a", "b", 0.0)
        ground_to_reference = np.asarray(
            ((100.0, 0.0, 500.0), (0.0, 100.0, 300.0), (0.01, 0.02, 1.0))
        )
        constraint = GroundDirectionConstraint(
            "b",
            GroundLineFamily.LONGITUDINAL,
            (0.0, 100.0),
            (1280.0, 100.0),
            10000.0,
        )
        result = solve_global_frame_graph(
            nodes,
            (edge,),
            "a",
            _pruning_rounds=0,
            direction_constraints=(constraint,),
            reference_ground_to_image=ground_to_reference,
        )
        vanishing = np.linalg.inv(result.node_to_reference["b"]) @ ground_to_reference[:, 0]
        vanishing /= vanishing[2]
        self.assertLess(abs(float(vanishing[1]) - 100.0), 50.0)


if __name__ == "__main__":
    unittest.main()
