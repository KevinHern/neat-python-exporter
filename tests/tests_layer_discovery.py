# Models
from neat_python_utility.neat_utility.models.genome_analyzer import GenomeAnalyzer
from neat_python_utility.tests.utils.tests_constants import TestCases

# Testing
import unittest


class LayerDiscoveryTestCase(unittest.TestCase):
    def run_test(self, test_case, expected_layers, expected_inputs_per_layer):
        # Extracting values from test_case
        id_inputs = test_case.id_inputs
        id_outputs = test_case.id_outputs
        connections = test_case.connections

        # Instantiation a ConnectionAnalysis object
        connection_analysis = GenomeAnalyzer(
            id_inputs=id_inputs, id_outputs=id_outputs,
            connections=connections, nodes=None,
        )

        # Filter connections
        connection_analysis.filter_useful_connections()

        # Get all paths
        connection_analysis.discover_all_connection_paths()

        # Test
        connection_analysis.construct_layers()

        # Assertions
        self.assertEqual(expected_layers, connection_analysis.layers)
        self.assertEqual(expected_inputs_per_layer, connection_analysis.inputs_per_layer)

    def test_simple_nn(self):
        expected_layers = [
            {1, 2, 3},
            {4}
        ]

        expected_inputs_per_layer = [
            [-1, -2, -3],
            [1, 2, 3]
        ]

        # Test
        self.run_test(
            test_case=TestCases.SIMPLE_NN,
            expected_layers=expected_layers,
            expected_inputs_per_layer=expected_inputs_per_layer,
        )

    def test_multiple_outputs_nn(self):
        expected_layers = [
            {1, 2, 3},
            {4, 5, 6}
        ]

        expected_inputs_per_layer = [
            [-1, -2, -3],
            [1, 2, 3]
        ]

        # Test
        self.run_test(
            test_case=TestCases.MULTIPLE_OUTPUTS_NN,
            expected_layers=expected_layers,
            expected_inputs_per_layer=expected_inputs_per_layer,
        )

    def test_two_hidden_layer_nn(self):
        expected_layers = [
            {1, 2, 3, 4},
            {6, 7, 8},
            {9, 10},
            {5}
        ]

        expected_inputs_per_layer = [
            [-1, -2, -3, -4],
            [1, 2, 3, 4],
            [6, 7, 8],
            [9, 10]
        ]

        # Test
        self.run_test(
            test_case=TestCases.TWO_HIDDEN_LAYERS_NN,
            expected_layers=expected_layers,
            expected_inputs_per_layer=expected_inputs_per_layer,
        )

    def test_skipped_input_to_output_nn(self):
        expected_layers = [
            {15},
            {0},
        ]

        expected_inputs_per_layer = [
            [-1, -2],
            [-1, -2, 15],
        ]

        # Test
        self.run_test(
            test_case=TestCases.SKIPPED_INPUT_TO_OUTPUT_NN,
            expected_layers=expected_layers,
            expected_inputs_per_layer=expected_inputs_per_layer,
        )

    def test_abandoned_nodes_one_nn(self):
        expected_layers = [
            {41},
            {0},
        ]

        expected_inputs_per_layer = [
            [-1, -2],
            [-1, 41],
        ]

        # Test
        self.run_test(
            test_case=TestCases.ABANDONED_NODES_ONE_NN,
            expected_layers=expected_layers,
            expected_inputs_per_layer=expected_inputs_per_layer,
        )

    def test_abandoned_nodes_two_nn(self):
        expected_layers = [
            {0},
            {421},
            {107}
        ]

        expected_inputs_per_layer = [
            [-1, -2],
            [0],
            [-1, 0, 421],
        ]

        # Test
        self.run_test(
            test_case=TestCases.ABANDONED_NODES_TWO_NN,
            expected_layers=expected_layers,
            expected_inputs_per_layer=expected_inputs_per_layer,
        )

    def test_abandoned_nodes_three_nn(self):
        expected_layers = [
            {291},
            {0}
        ]

        expected_inputs_per_layer = [
            [-1, -2],
            [291]
        ]

        # Test
        self.run_test(
            test_case=TestCases.ABANDONED_NODES_THREE_NN,
            expected_layers=expected_layers,
            expected_inputs_per_layer=expected_inputs_per_layer,
        )

    def test_weird_topology_one_nn(self):
        expected_layers = [
            {2, 3},
            {1, 5},
            {6},
            {4}
        ]

        expected_inputs_per_layer = [
            [-2, -3],
            [-1, 2, 3],
            [1, 5],
            [1, 3, 5, 6],
        ]

        # Test
        self.run_test(
            test_case=TestCases.WEIRD_TOPOLOGY_ONE_NN,
            expected_layers=expected_layers,
            expected_inputs_per_layer=expected_inputs_per_layer,
        )

    def test_weird_topology_two_nn(self):
        expected_layers = [
            {1, 2, 3},
            {6},
            {7},
            {4, 5}
        ]

        expected_inputs_per_layer = [
            [-1, -2, -3],
            [1, 2, 3],
            [1, 6],
            [1, 2, 3, 6, 7],
        ]

        # Test
        self.run_test(
            test_case=TestCases.WEIRD_TOPOLOGY_TWO_NN,
            expected_layers=expected_layers,
            expected_inputs_per_layer=expected_inputs_per_layer,
        )

    def test_weird_topology_three_nn(self):
        expected_layers = [
            {101},
            {0}
        ]

        expected_inputs_per_layer = [
            [-1, -2],
            [101],
        ]

        # Test
        self.run_test(
            test_case=TestCases.WEIRD_TOPOLOGY_THREE_NN,
            expected_layers=expected_layers,
            expected_inputs_per_layer=expected_inputs_per_layer,
        )

    def test_weird_topology_four_nn(self):
        expected_layers = [
            {451},
            {0}
        ]

        expected_inputs_per_layer = [
            [-1, -2],
            [-1, -2, 451],
        ]

        # Test
        self.run_test(
            test_case=TestCases.WEIRD_TOPOLOGY_FOUR_NN,
            expected_layers=expected_layers,
            expected_inputs_per_layer=expected_inputs_per_layer,
        )

    def test_weird_topology_five_nn(self):
        expected_layers = [
            {1024},
            {1666},
            {51},
            {0}
        ]

        expected_inputs_per_layer = [
            [-2],
            [1024],
            [-1, -2, 1024, 1666],
            [51]
        ]

        # Test
        self.run_test(
            test_case=TestCases.WEIRD_TOPOLOGY_FIVE_NN,
            expected_layers=expected_layers,
            expected_inputs_per_layer=expected_inputs_per_layer,
        )


if __name__ == '__main__':
    unittest.main()
