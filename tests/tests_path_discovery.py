# Models
from neat_python_utility.neat_utility.models.genome_analyzer import GenomeAnalyzer
from utils.tests_constants import TestCases

# Testing
import unittest


class PathDiscoveryTestCase(unittest.TestCase):
    def run_test(self, test_case, expected_paths):
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

        # Test
        connection_analysis.discover_all_connection_paths()

        result = list(
            map(
                lambda x: list(
                    map(
                        lambda y: y.identification_number,
                        x
                    )
                ),
                connection_analysis.all_paths
            )
        )

        # Assertions
        self.assertTrue(
            # Check if EVERY path in expected_paths is in result
            all(item_path in result for item_path in expected_paths)
            and
            # Make sure there are NO residual paths
            all(item_path in expected_paths for item_path in result)
        )

    def test_simple_nn(self):
        # Initializing dummy paths
        expected_paths = [
            [(-1, 1), (1, 4)],
            [(-2, 2), (2, 4)],
            [(-3, 3), (3, 4)]
        ]

        # Test
        self.run_test(
            test_case=TestCases.SIMPLE_NN,
            expected_paths=expected_paths,
        )

    def test_multiple_outputs_nn(self):
        # Initializing dummy paths
        expected_paths = [
            [(-1, 1), (1, 4)],
            [(-1, 1), (1, 5)],
            [(-2, 2), (2, 4)],
            [(-2, 2), (2, 6)],
            [(-3, 3), (3, 4)],
            [(-3, 3), (3, 5)]
        ]

        # Test
        self.run_test(
            test_case=TestCases.MULTIPLE_OUTPUTS_NN,
            expected_paths=expected_paths,
        )

    def test_two_hidden_layer_nn(self):
        # Initializing dummy paths
        expected_paths = [
            [(-1, 1), (1, 6), (6, 9), (9, 5)],
            [(-1, 1), (1, 8), (8, 10), (10, 5)],
            [(-2, 2), (2, 6), (6, 9), (9, 5)],
            [(-3, 3), (3, 7), (7, 9), (9, 5)],
            [(-3, 3), (3, 7), (7, 10), (10, 5)],
            [(-4, 4), (4, 7), (7, 9), (9, 5)],
            [(-4, 4), (4, 7), (7, 10), (10, 5)],
            [(-4, 4), (4, 8), (8, 10), (10, 5)],
        ]

        # Test
        self.run_test(
            test_case=TestCases.TWO_HIDDEN_LAYERS_NN,
            expected_paths=expected_paths,
        )

    def test_skipped_input_to_output_nn(self):
        # Initializing dummy paths
        expected_paths = [
            [(-1, 0)],
            [(-2, 0)],
            [(-1, 15), (15, 0)],
            [(-2, 15), (15, 0)],
        ]

        # Test
        self.run_test(
            test_case=TestCases.SKIPPED_INPUT_TO_OUTPUT_NN,
            expected_paths=expected_paths,
        )

    def test_abandoned_nodes_one_nn(self):
        # Initializing dummy paths
        expected_paths = [
            [(-1, 0)],
            [(-1, 41), (41, 0)],
            [(-2, 41), (41, 0)]
        ]

        # Test
        self.run_test(
            test_case=TestCases.ABANDONED_NODES_ONE_NN,
            expected_paths=expected_paths,
        )

    def test_abandoned_nodes_two_nn(self):
        # Initializing dummy paths
        expected_paths = [
            [(-1, 107)],
            [(-1, 0), (0, 107)],
            [(-1, 0), (0, 421), (421, 107)],
            [(-2, 0), (0, 107)],
            [(-2, 0), (0, 421), (421, 107)],
        ]

        # Test
        self.run_test(
            test_case=TestCases.ABANDONED_NODES_TWO_NN,
            expected_paths=expected_paths,
        )

    def test_abandoned_nodes_three_nn(self):
        # Initializing dummy paths
        expected_paths = [
            [(-1, 291), (291, 0)],
            [(-2, 291), (291, 0)],
        ]

        # Test
        self.run_test(
            test_case=TestCases.ABANDONED_NODES_THREE_NN,
            expected_paths=expected_paths,
        )

    def test_weird_topology_one_nn(self):
        # Initializing dummy paths
        expected_paths = [
            [(-1, 1), (1, 4)],
            [(-1, 1), (1, 6), (6, 4)],
            [(-2, 2), (2, 5), (5, 6), (6, 4)],
            [(-2, 2), (2, 5), (5, 4)],
            [(-3, 3), (3, 4)],
            [(-3, 3), (3, 5), (5, 6), (6, 4)],
            [(-3, 3), (3, 5), (5, 4)],
        ]

        # Test
        self.run_test(
            test_case=TestCases.WEIRD_TOPOLOGY_ONE_NN,
            expected_paths=expected_paths,
        )

    def test_weird_topology_two_nn(self):
        # Initializing dummy paths
        expected_paths = [
            [(-1, 1), (1, 4)],
            [(-1, 1), (1, 6), (6, 5)],
            [(-1, 1), (1, 6), (6, 7), (7, 4)],
            [(-1, 1), (1, 7), (7, 4)],

            [(-2, 2), (2, 5)],
            [(-2, 2), (2, 6), (6, 5)],
            [(-2, 2), (2, 6), (6, 7), (7, 4)],

            [(-3, 3), (3, 4)],
            [(-3, 3), (3, 6), (6, 7), (7, 4)],
            [(-3, 3), (3, 6), (6, 5)],
        ]

        # Test
        self.run_test(
            test_case=TestCases.WEIRD_TOPOLOGY_TWO_NN,
            expected_paths=expected_paths,
        )

    def test_weird_topology_three_nn(self):
        # Initializing dummy paths
        expected_paths = [
            [(-1, 101), (101, 0)],
            [(-2, 101), (101, 0)],
        ]

        # Test
        self.run_test(
            test_case=TestCases.WEIRD_TOPOLOGY_THREE_NN,
            expected_paths=expected_paths,
        )

    def test_weird_topology_four_nn(self):
        # Initializing dummy paths
        expected_paths = [
            [(-1, 0)],
            [(-2, 0)],
            [(-1, 451), (451, 0)],
            [(-2, 451), (451, 0)],
        ]

        # Test
        self.run_test(
            test_case=TestCases.WEIRD_TOPOLOGY_FOUR_NN,
            expected_paths=expected_paths,
        )

    def test_asymmetric_topology_one_nn(self):
        # Initializing dummy paths
        expected_paths = [
            [(-1, 51), (51, 0)],
            [(-2, 51), (51, 0)],
            [(-2, 1024), (1024, 51), (51, 0)],
            [(-2, 1024), (1024, 1666), (1666, 51), (51, 0)],
        ]

        # Test
        self.run_test(
            test_case=TestCases.ASYMMETRIC_TOPOLOGY_ONE_NN,
            expected_paths=expected_paths,
        )

    def test_asymmetric_topology_two_nn(self):
        # Initializing dummy paths
        expected_paths = [
            [(-1, 320), (320, 0)],
            [(-1, 398), (398, 513), (513, 0)],
            [(-1, 320), (320, 398), (398, 513), (513, 0)],
            [(-2, 513), (513, 0)],
            [(-2, 398), (398, 513), (513, 0)],
        ]

        # Test
        self.run_test(
            test_case=TestCases.ASYMMETRIC_TOPOLOGY_TWO_NN,
            expected_paths=expected_paths,
        )

    def test_asymmetric_topology_three_nn(self):
        # Initializing dummy paths
        expected_paths = [
            [(-1, 168), (168, 0)],
            [(-1, 1850), (1850, 168), (168, 0)],
            [(-2, 168), (168, 0)],
            [(-2, 865), (865, 168), (168, 0)],
            [(-2, 865), (865, 1850), (1850, 168), (168, 0)],
        ]

        # Test
        self.run_test(
            test_case=TestCases.ASYMMETRIC_TOPOLOGY_THREE_NN,
            expected_paths=expected_paths,
        )

    def test_asymmetric_topology_four_nn(self):
        # Initializing dummy paths
        expected_paths = [
            [(-1, 897), (897, 49), (49, 0)],
            [(-1, 967), (967, 49), (49, 0)],
            [(-2, 49), (49, 0)],
        ]

        # Test
        self.run_test(
            test_case=TestCases.ASYMMETRIC_TOPOLOGY_FOUR_NN,
            expected_paths=expected_paths,
        )


if __name__ == '__main__':
    unittest.main()
