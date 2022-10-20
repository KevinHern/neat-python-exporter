# Models
from neat_python_utility.neat_utility.models.genome_analyzer import GenomeAnalyzer
from .utils.tests_constants import TestCases

# Testing
import unittest


class ConnectionAnalysisTestCase(unittest.TestCase):
    def run_test(self, test_case, expected_connection_keys):
        # Extracting values from test_case
        id_inputs = test_case.id_inputs
        id_outputs = test_case.id_outputs
        connections = test_case.connections

        # Initializing connections
        expected_useful_connections = list(
            filter(
                lambda x: x.identification_number in expected_connection_keys,
                connections
            )
        )

        # Test
        filtered_connections = GenomeAnalyzer(
            id_inputs=id_inputs, id_outputs=id_outputs,
            connections=connections, nodes=None,
        )

        filtered_connections.filter_useful_connections()

        # Assertions
        self.assertEqual(expected_useful_connections, filtered_connections.connections)

    def test_simple_nn(self):
        # Setting up connections
        expected_connections_keys = TestCases.SIMPLE_NN.connections_keys.copy()

        self.run_test(
            test_case=TestCases.SIMPLE_NN,
            expected_connection_keys=expected_connections_keys,
        )

    def test_multiple_outputs_nn(self):
        # Setting up connections
        expected_connections_keys = TestCases.MULTIPLE_OUTPUTS_NN.connections_keys.copy()

        self.run_test(
            test_case=TestCases.MULTIPLE_OUTPUTS_NN,
            expected_connection_keys=expected_connections_keys,
        )

    def test_two_hidden_layer_nn(self):
        # Setting up connections
        expected_connections_keys = TestCases.TWO_HIDDEN_LAYERS_NN.connections_keys.copy()

        self.run_test(
            test_case=TestCases.TWO_HIDDEN_LAYERS_NN,
            expected_connection_keys=expected_connections_keys,
        )

    def test_skipped_input_to_output_nn(self):
        # Setting up connections
        expected_connections_keys = TestCases.SKIPPED_INPUT_TO_OUTPUT_NN.connections_keys.copy()

        self.run_test(
            test_case=TestCases.SKIPPED_INPUT_TO_OUTPUT_NN,
            expected_connection_keys=expected_connections_keys,
        )

    def test_abandoned_nodes_one_nn(self):
        # Setting up connections
        expected_connections_keys = [
            (-1, 41), (-2, 41),
            (-1, 0), (41, 0)
        ]

        self.run_test(
            test_case=TestCases.ABANDONED_NODES_ONE_NN,
            expected_connection_keys=expected_connections_keys,
        )

    def test_abandoned_nodes_two_nn(self):
        # Setting up connections
        expected_connections_keys = [
            (-1, 0), (-2, 0),
            (0, 421),
            (421, 107), (0, 107), (-1, 107)
        ]

        self.run_test(
            test_case=TestCases.ABANDONED_NODES_TWO_NN,
            expected_connection_keys=expected_connections_keys,
        )

    def test_abandoned_nodes_three_nn(self):
        # Setting up connections
        expected_connections_keys = [
            (-1, 291), (-2, 291),
            (291, 0),
        ]

        self.run_test(
            test_case=TestCases.ABANDONED_NODES_THREE_NN,
            expected_connection_keys=expected_connections_keys,
        )

    def test_weird_topology_one_nn(self):
        # Setting up connections
        expected_connections_keys = TestCases.WEIRD_TOPOLOGY_ONE_NN.connections_keys.copy()

        self.run_test(
            test_case=TestCases.WEIRD_TOPOLOGY_ONE_NN,
            expected_connection_keys=expected_connections_keys,
        )

    def test_weird_topology_two_nn(self):
        # Setting up connections
        expected_connections_keys = TestCases.WEIRD_TOPOLOGY_TWO_NN.connections_keys.copy()

        self.run_test(
            test_case=TestCases.WEIRD_TOPOLOGY_TWO_NN,
            expected_connection_keys=expected_connections_keys,
        )

    def test_weird_topology_three_nn(self):
        # Setting up connections
        expected_connections_keys = [
            (-1, 101), (-2, 101), (101, 0)
        ]

        self.run_test(
            test_case=TestCases.WEIRD_TOPOLOGY_THREE_NN,
            expected_connection_keys=expected_connections_keys,
        )

    def test_weird_topology_four_nn(self):
        # Setting up connections
        expected_connections_keys = [
            (-1, 451), (-2, 451),
            (-1, 0), (-2, 0), (451, 0)
        ]

        self.run_test(
            test_case=TestCases.WEIRD_TOPOLOGY_FOUR_NN,
            expected_connection_keys=expected_connections_keys,
        )

    def test_asymmetric_topology_one_nn(self):
        # Setting up connections
        expected_connections_keys = [
            (-2, 1024),
            (1024, 1666),
            (-1, 51), (-2, 51), (1024, 51), (1666, 51),
            (51, 0),
        ]

        self.run_test(
            test_case=TestCases.ASYMMETRIC_TOPOLOGY_ONE_NN,
            expected_connection_keys=expected_connections_keys,
        )

    def test_asymmetric_topology_two_nn(self):
        # Setting up connections
        expected_connections_keys = TestCases.ASYMMETRIC_TOPOLOGY_TWO_NN.connections_keys.copy()

        self.run_test(
            test_case=TestCases.ASYMMETRIC_TOPOLOGY_TWO_NN,
            expected_connection_keys=expected_connections_keys,
        )

    def test_asymmetric_topology_three_nn(self):
        # Setting up connections
        expected_connections_keys = TestCases.ASYMMETRIC_TOPOLOGY_THREE_NN.connections_keys.copy()

        self.run_test(
            test_case=TestCases.ASYMMETRIC_TOPOLOGY_THREE_NN,
            expected_connection_keys=expected_connections_keys,
        )

    def test_asymmetric_topology_four_nn(self):
        # Setting up connections
        expected_connections_keys = TestCases.ASYMMETRIC_TOPOLOGY_FOUR_NN.connections_keys.copy()

        self.run_test(
            test_case=TestCases.ASYMMETRIC_TOPOLOGY_FOUR_NN,
            expected_connection_keys=expected_connections_keys,
        )


if __name__ == '__main__':
    unittest.main()
