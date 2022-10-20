# Models
from neat_python_utility.neat_utility.models.genome_analyzer import GenomeAnalyzer
from neat_python_utility.tests.utils.tests_constants import TestCases

# Testing
import unittest


class MatricesDefinitionTestCase(unittest.TestCase):
    def run_test(self, test_case,
                 expected_weights_matrix, expected_biases_matrix, expected_afs_matrix):
        # Extracting values from test_case
        id_inputs = test_case.id_inputs
        id_outputs = test_case.id_outputs
        connections = test_case.connections
        nodes = test_case.nodes

        # Instantiation a ConnectionAnalysis object
        connection_analysis = GenomeAnalyzer(
            id_inputs=id_inputs, id_outputs=id_outputs,
            connections=connections, nodes=nodes,
        )

        # Filter connections
        connection_analysis.filter_useful_connections()

        # Get all paths
        connection_analysis.discover_all_connection_paths()

        # Construct layers and deduce inputs per layer
        connection_analysis.construct_layers()

        # Test
        connection_analysis.construct_matrices()

        # Assertions
        self.assertEqual(expected_weights_matrix, connection_analysis.weights_matrix)
        self.assertEqual(expected_biases_matrix, connection_analysis.biases_matrix)
        self.assertEqual(expected_afs_matrix, connection_analysis.activation_functions_matrix)

    def test_simple_nn(self):
        # Initializing variables
        test_case = TestCases.SIMPLE_NN
        connections = test_case.connections
        nodes = test_case.nodes

        # Initializing expected matrices
        expected_weights_matrix = [
            [
                [connections[0].weight, 0.0, 0.0],
                [0.0, connections[1].weight, 0.0],
                [0.0, 0.0, connections[2].weight],

            ],
            [
                [connections[3].weight, connections[4].weight, connections[5].weight]
            ]
        ]

        expected_biases_matrix = [
            [nodes[0].bias, nodes[1].bias, nodes[2].bias],
            [nodes[3].bias],
        ]

        expected_afs_matrix = [
            [nodes[0].activation_function, nodes[1].activation_function, nodes[2].activation_function],
            [nodes[3].activation_function],
        ]

        # Test
        self.run_test(
            test_case=test_case,
            expected_weights_matrix=expected_weights_matrix,
            expected_biases_matrix=expected_biases_matrix,
            expected_afs_matrix=expected_afs_matrix
        )

    def test_multiple_outputs_nn(self):
        # Initializing variables
        test_case = TestCases.MULTIPLE_OUTPUTS_NN
        connections = test_case.connections
        nodes = test_case.nodes

        # Initializing expected matrices
        expected_weights_matrix = [
            [
                [connections[0].weight, 0.0, 0.0],
                [0.0, connections[1].weight, 0.0],
                [0.0, 0.0, connections[2].weight],

            ],
            [
                [connections[3].weight, connections[4].weight, connections[5].weight],
                [connections[6].weight, 0.0, connections[7].weight],
                [0.0, connections[8].weight, 0.0]
            ]
        ]

        expected_biases_matrix = [
            [nodes[0].bias, nodes[1].bias, nodes[2].bias],
            [nodes[3].bias, nodes[4].bias, nodes[5].bias],
        ]

        expected_afs_matrix = [
            [nodes[0].activation_function, nodes[1].activation_function, nodes[2].activation_function],
            [nodes[3].activation_function, nodes[4].activation_function, nodes[5].activation_function],
        ]

        # Test
        self.run_test(
            test_case=test_case,
            expected_weights_matrix=expected_weights_matrix,
            expected_biases_matrix=expected_biases_matrix,
            expected_afs_matrix=expected_afs_matrix
        )

    def test_two_hidden_layers_nn(self):
        # Initializing variables
        test_case = TestCases.TWO_HIDDEN_LAYERS_NN
        connections = test_case.connections
        nodes = test_case.nodes

        # Initializing expected matrices
        expected_weights_matrix = [
            [
                [connections[0].weight, 0.0, 0.0, 0.0],
                [0.0, connections[1].weight, 0.0, 0.0],
                [0.0, 0.0, connections[2].weight, 0.0],
                [0.0, 0.0, 0.0, connections[3].weight],
            ],
            [
                [connections[4].weight, connections[5].weight, 0.0, 0.0],
                [0.0, 0.0, connections[6].weight, connections[7].weight],
                [connections[8].weight, 0.0, 0.0, connections[9].weight],
            ],
            [
                [connections[10].weight, connections[11].weight, 0.0],
                [0.0, connections[12].weight, connections[13].weight],
            ],
            [
                [connections[14].weight, connections[15].weight],
            ]
        ]

        expected_biases_matrix = [
            [nodes[0].bias, nodes[1].bias, nodes[2].bias, nodes[3].bias],
            [nodes[5].bias, nodes[6].bias, nodes[7].bias],
            [nodes[8].bias, nodes[9].bias],
            [nodes[4].bias],
        ]

        expected_afs_matrix = [
            [nodes[0].activation_function, nodes[1].activation_function, nodes[2].activation_function,
             nodes[3].activation_function],
            [nodes[5].activation_function, nodes[6].activation_function, nodes[7].activation_function],
            [nodes[8].activation_function, nodes[9].activation_function],
            [nodes[4].activation_function],
        ]

        # Test
        self.run_test(
            test_case=test_case,
            expected_weights_matrix=expected_weights_matrix,
            expected_biases_matrix=expected_biases_matrix,
            expected_afs_matrix=expected_afs_matrix
        )

    def test_skipped_input_to_output_nn(self):
        # Initializing variables
        test_case = TestCases.SKIPPED_INPUT_TO_OUTPUT_NN
        connections = test_case.connections
        nodes = test_case.nodes

        # Initializing expected matrices
        expected_weights_matrix = [
            [
                [connections[0].weight, connections[1].weight],
            ],
            [
                [connections[3].weight, connections[4].weight, connections[2].weight],
            ],
        ]

        expected_biases_matrix = [
            [nodes[0].bias],
            [nodes[1].bias],
        ]

        expected_afs_matrix = [
            [nodes[0].activation_function],
            [nodes[1].activation_function],
        ]

        # Test
        self.run_test(
            test_case=test_case,
            expected_weights_matrix=expected_weights_matrix,
            expected_biases_matrix=expected_biases_matrix,
            expected_afs_matrix=expected_afs_matrix
        )

    def test_abandoned_nodes_one_nn(self):
        # Initializing variables
        test_case = TestCases.ABANDONED_NODES_ONE_NN
        connections = test_case.connections
        nodes = test_case.nodes

        # Initializing expected matrices
        expected_weights_matrix = [
            [
                [connections[0].weight, connections[1].weight],
            ],
            [
                [connections[3].weight, connections[4].weight],
            ],
        ]

        expected_biases_matrix = [
            [nodes[0].bias],
            [nodes[2].bias],
        ]

        expected_afs_matrix = [
            [nodes[0].activation_function],
            [nodes[2].activation_function],
        ]

        # Test
        self.run_test(
            test_case=test_case,
            expected_weights_matrix=expected_weights_matrix,
            expected_biases_matrix=expected_biases_matrix,
            expected_afs_matrix=expected_afs_matrix
        )

    def test_abandoned_nodes_two_nn(self):
        # Initializing variables
        test_case = TestCases.ABANDONED_NODES_TWO_NN
        connections = test_case.connections
        nodes = test_case.nodes

        # Initializing expected matrices
        expected_weights_matrix = [
            [
                [connections[0].weight, connections[1].weight],
            ],
            [
                [connections[3].weight],
            ],
            [
                [connections[6].weight, connections[5].weight, connections[4].weight],
            ],
        ]

        expected_biases_matrix = [
            [nodes[0].bias],
            [nodes[2].bias],
            [nodes[4].bias],
        ]

        expected_afs_matrix = [
            [nodes[0].activation_function],
            [nodes[2].activation_function],
            [nodes[4].activation_function],
        ]

        # Test
        self.run_test(
            test_case=test_case,
            expected_weights_matrix=expected_weights_matrix,
            expected_biases_matrix=expected_biases_matrix,
            expected_afs_matrix=expected_afs_matrix
        )

    def test_abandoned_nodes_three_nn(self):
        # Initializing variables
        test_case = TestCases.ABANDONED_NODES_THREE_NN
        connections = test_case.connections
        nodes = test_case.nodes

        # Initializing expected matrices
        expected_weights_matrix = [
            [
                [connections[0].weight, connections[1].weight],
            ],
            [
                [connections[2].weight],
            ],
        ]

        expected_biases_matrix = [
            [nodes[0].bias],
            [nodes[1].bias],
        ]

        expected_afs_matrix = [
            [nodes[0].activation_function],
            [nodes[1].activation_function],
        ]

        # Test
        self.run_test(
            test_case=test_case,
            expected_weights_matrix=expected_weights_matrix,
            expected_biases_matrix=expected_biases_matrix,
            expected_afs_matrix=expected_afs_matrix
        )

    def test_weird_topology_one_nn(self):
        # Initializing variables
        test_case = TestCases.WEIRD_TOPOLOGY_ONE_NN
        connections = test_case.connections
        nodes = test_case.nodes

        # Initializing expected matrices
        expected_weights_matrix = [
            [
                [connections[1].weight, 0.0],
                [0.0, connections[2].weight],
            ],
            [
                [connections[0].weight, 0.0, 0.0],
                [0.0, connections[3].weight, connections[4].weight],
            ],
            [
                [connections[5].weight, connections[6].weight],
            ],
            [
                [connections[7].weight, connections[10].weight, connections[9].weight, connections[8].weight],
            ]
        ]

        expected_biases_matrix = [
            [nodes[1].bias, nodes[2].bias],
            [nodes[0].bias, nodes[4].bias],
            [nodes[5].bias],
            [nodes[3].bias],
        ]

        expected_afs_matrix = [
            [nodes[1].activation_function, nodes[2].activation_function],
            [nodes[0].activation_function, nodes[4].activation_function],
            [nodes[5].activation_function],
            [nodes[3].activation_function],
        ]

        # Test
        self.run_test(
            test_case=test_case,
            expected_weights_matrix=expected_weights_matrix,
            expected_biases_matrix=expected_biases_matrix,
            expected_afs_matrix=expected_afs_matrix
        )

    def test_weird_topology_two_nn(self):
        # Initializing variables
        test_case = TestCases.WEIRD_TOPOLOGY_TWO_NN
        connections = test_case.connections
        nodes = test_case.nodes

        # Initializing expected matrices
        expected_weights_matrix = [
            [
                [connections[0].weight, 0.0, 0.0],
                [0.0, connections[1].weight, 0.0],
                [0.0, 0.0, connections[2].weight],
            ],
            [
                [connections[3].weight, connections[4].weight, connections[5].weight],
            ],
            [
                [connections[6].weight, connections[7].weight],
            ],
            [
                [connections[8].weight, 0.0, connections[9].weight, 0.0, connections[10].weight],
                [0.0, connections[11].weight, 0.0, connections[12].weight, 0.0],
            ]
        ]

        expected_biases_matrix = [
            [nodes[0].bias, nodes[1].bias, nodes[2].bias],
            [nodes[5].bias],
            [nodes[6].bias],
            [nodes[3].bias, nodes[4].bias],
        ]

        expected_afs_matrix = [
            [nodes[0].activation_function, nodes[1].activation_function, nodes[2].activation_function],
            [nodes[5].activation_function],
            [nodes[6].activation_function],
            [nodes[3].activation_function, nodes[4].activation_function],
        ]

        # Test
        self.run_test(
            test_case=test_case,
            expected_weights_matrix=expected_weights_matrix,
            expected_biases_matrix=expected_biases_matrix,
            expected_afs_matrix=expected_afs_matrix
        )

    def test_weird_topology_three_nn(self):
        # Initializing variables
        test_case = TestCases.WEIRD_TOPOLOGY_THREE_NN
        connections = test_case.connections
        nodes = test_case.nodes

        # Initializing expected matrices
        expected_weights_matrix = [
            [
                [connections[0].weight, connections[1].weight],
            ],
            [
                [connections[2].weight],
            ],
        ]

        expected_biases_matrix = [
            [nodes[0].bias],
            [nodes[1].bias],
        ]

        expected_afs_matrix = [
            [nodes[0].activation_function],
            [nodes[1].activation_function],
        ]

        # Test
        self.run_test(
            test_case=test_case,
            expected_weights_matrix=expected_weights_matrix,
            expected_biases_matrix=expected_biases_matrix,
            expected_afs_matrix=expected_afs_matrix
        )

    def test_weird_topology_four_nn(self):
        # Initializing variables
        test_case = TestCases.WEIRD_TOPOLOGY_FOUR_NN
        connections = test_case.connections
        nodes = test_case.nodes

        # Initializing expected matrices
        expected_weights_matrix = [
            [
                [connections[0].weight, connections[1].weight],
            ],
            [
                [connections[2].weight, connections[3].weight, connections[4].weight],
            ],
        ]

        expected_biases_matrix = [
            [nodes[0].bias],
            [nodes[1].bias],
        ]

        expected_afs_matrix = [
            [nodes[0].activation_function],
            [nodes[1].activation_function],
        ]

        # Test
        self.run_test(
            test_case=test_case,
            expected_weights_matrix=expected_weights_matrix,
            expected_biases_matrix=expected_biases_matrix,
            expected_afs_matrix=expected_afs_matrix
        )

    def test_asymmetric_topology_one_nn(self):
        # Initializing variables
        test_case = TestCases.ASYMMETRIC_TOPOLOGY_ONE_NN
        connections = test_case.connections
        nodes = test_case.nodes

        # Initializing expected matrices
        expected_weights_matrix = [
            [
                [connections[0].weight],
            ],
            [
                [connections[1].weight],
            ],
            [
                [connections[2].weight, connections[3].weight, connections[4].weight, connections[5].weight],
            ],
            [
                [connections[6].weight],
            ],
        ]

        expected_biases_matrix = [
            [nodes[0].bias],
            [nodes[1].bias],
            [nodes[2].bias],
            [nodes[3].bias],
        ]

        expected_afs_matrix = [
            [nodes[0].activation_function],
            [nodes[1].activation_function],
            [nodes[2].activation_function],
            [nodes[3].activation_function],
        ]

        # Test
        self.run_test(
            test_case=test_case,
            expected_weights_matrix=expected_weights_matrix,
            expected_biases_matrix=expected_biases_matrix,
            expected_afs_matrix=expected_afs_matrix
        )

    def test_asymmetric_topology_two_nn(self):
        # Initializing variables
        test_case = TestCases.ASYMMETRIC_TOPOLOGY_TWO_NN
        connections = test_case.connections
        nodes = test_case.nodes

        # Initializing expected matrices
        expected_weights_matrix = [
            [
                [connections[0].weight],
            ],
            [
                [connections[1].weight, connections[2].weight, connections[3].weight],
            ],
            [
                [connections[4].weight, connections[5].weight],
            ],
            [
                [connections[6].weight, connections[7].weight],
            ],
        ]

        expected_biases_matrix = [
            [nodes[0].bias],
            [nodes[1].bias],
            [nodes[2].bias],
            [nodes[3].bias],
        ]

        expected_afs_matrix = [
            [nodes[0].activation_function],
            [nodes[1].activation_function],
            [nodes[2].activation_function],
            [nodes[3].activation_function],
        ]

        # Test
        self.run_test(
            test_case=test_case,
            expected_weights_matrix=expected_weights_matrix,
            expected_biases_matrix=expected_biases_matrix,
            expected_afs_matrix=expected_afs_matrix
        )

    def test_asymmetric_topology_three_nn(self):
        # Initializing variables
        test_case = TestCases.ASYMMETRIC_TOPOLOGY_THREE_NN
        connections = test_case.connections
        nodes = test_case.nodes

        # Initializing expected matrices
        expected_weights_matrix = [
            [
                [connections[0].weight],
            ],
            [
                [connections[1].weight, connections[2].weight],
            ],
            [
                [connections[3].weight, connections[4].weight, connections[5].weight, connections[6].weight],
            ],
            [
                [connections[7].weight],
            ],
        ]

        expected_biases_matrix = [
            [nodes[0].bias],
            [nodes[1].bias],
            [nodes[2].bias],
            [nodes[3].bias],
        ]

        expected_afs_matrix = [
            [nodes[0].activation_function],
            [nodes[1].activation_function],
            [nodes[2].activation_function],
            [nodes[3].activation_function],
        ]

        # Test
        self.run_test(
            test_case=test_case,
            expected_weights_matrix=expected_weights_matrix,
            expected_biases_matrix=expected_biases_matrix,
            expected_afs_matrix=expected_afs_matrix
        )

    def test_asymmetric_topology_four_nn(self):
        # Initializing variables
        test_case = TestCases.ASYMMETRIC_TOPOLOGY_FOUR_NN
        connections = test_case.connections
        nodes = test_case.nodes

        # Initializing expected matrices
        expected_weights_matrix = [
            [
                [connections[0].weight],
                [connections[1].weight],
            ],
            [
                [connections[2].weight, connections[3].weight, connections[4].weight],
            ],
            [
                [connections[5].weight],
            ],
        ]

        expected_biases_matrix = [
            [nodes[0].bias, nodes[1].bias],
            [nodes[2].bias],
            [nodes[3].bias],
        ]

        expected_afs_matrix = [
            [nodes[0].activation_function, nodes[1].activation_function],
            [nodes[2].activation_function],
            [nodes[3].activation_function],
        ]

        # Test
        self.run_test(
            test_case=test_case,
            expected_weights_matrix=expected_weights_matrix,
            expected_biases_matrix=expected_biases_matrix,
            expected_afs_matrix=expected_afs_matrix
        )


if __name__ == '__main__':
    unittest.main()
