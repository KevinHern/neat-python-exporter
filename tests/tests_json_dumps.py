# Models
from neat_python_utility.neat_utility.models.genome_analyzer import GenomeAnalyzer
from neat_python_utility.tests.utils.tests_constants import TestCases

# Utils
from os.path import dirname, join, exists
from os import mkdir
from json import dump

# Testing
import unittest

JSON_DIRECTORY = join(dirname(__file__), 'json_dumps')
if not exists(JSON_DIRECTORY):
    mkdir(JSON_DIRECTORY)


class JsonDumpTestCase(unittest.TestCase):
    def run_test(self, test_case, filename):
        # Extracting values from test_case
        id_inputs = test_case.id_inputs
        id_outputs = test_case.id_outputs
        connections = test_case.connections
        nodes = test_case.nodes

        # Initializing names of special
        inputs_names = ["Input{}".format(x) for x in range(len(id_inputs))]
        outputs_names = ["Output{}".format(x) for x in range(len(id_outputs))]

        # Instantiating a GenomeAnalysis object
        genome_analyzer = GenomeAnalyzer(
            id_inputs=id_inputs, id_outputs=id_outputs,
            connections=connections, nodes=nodes
        )

        # Filter connections and get rid of useless and abandoned nodes
        genome_analyzer.filter_useful_connections()

        # Discover all possible paths
        genome_analyzer.discover_all_connection_paths()

        # Construct layers and deduce inputs per layer
        genome_analyzer.construct_layers()

        # Build weights, biases and activation functions matrices
        genome_analyzer.construct_matrices()

        # Get a Map of the neural network
        neural_network_map = genome_analyzer.network_to_map(inputs_name=inputs_names, outputs_name=outputs_names)

        # Convert the map into a JSON and save it
        with open(filename, "w") as outfile:
            dump(neural_network_map, outfile, indent=4)

        # Asserting JSON exists
        self.assertEqual(exists(filename), True)

    def test_simple_nn(self):
        # Setting filename
        filename = join(JSON_DIRECTORY, "test_simple_nn.json")

        # Test
        self.run_test(
            test_case=TestCases.SIMPLE_NN,
            filename=filename
        )

    def test_multiple_outputs_nn(self):
        # Setting filename
        filename = join(JSON_DIRECTORY, "test_multiple_outputs_nn.json")

        # Test
        self.run_test(
            test_case=TestCases.MULTIPLE_OUTPUTS_NN,
            filename=filename
        )

    def test_two_hidden_layers_nn(self):
        # Setting filename
        filename = join(JSON_DIRECTORY, "test_two_hidden_layers_nn.json")

        # Test
        self.run_test(
            test_case=TestCases.TWO_HIDDEN_LAYERS_NN,
            filename=filename
        )

    def test_skipped_input_to_output_nn(self):
        # Setting filename
        filename = join(JSON_DIRECTORY, "test_skipped_input_to_output_nn.json")

        # Test
        self.run_test(
            test_case=TestCases.SKIPPED_INPUT_TO_OUTPUT_NN,
            filename=filename
        )

    def test_abandoned_nodes_one_nn(self):
        # Setting filename
        filename = join(JSON_DIRECTORY, "test_abandoned_nodes_one_nn.json")

        # Test
        self.run_test(
            test_case=TestCases.ABANDONED_NODES_ONE_NN,
            filename=filename
        )

    def test_abandoned_nodes_two_nn(self):
        # Setting filename
        filename = join(JSON_DIRECTORY, "test_abandoned_nodes_two_nn.json")

        # Test
        self.run_test(
            test_case=TestCases.ABANDONED_NODES_TWO_NN,
            filename=filename
        )

    def test_abandoned_nodes_three_nn(self):
        # Setting filename
        filename = join(JSON_DIRECTORY, "test_abandoned_nodes_three_nn.json")

        # Test
        self.run_test(
            test_case=TestCases.ABANDONED_NODES_THREE_NN,
            filename=filename
        )

    def test_weird_topology_one_nn(self):
        # Setting filename
        filename = join(JSON_DIRECTORY, "test_weird_topology_one_nn.json")

        # Test
        self.run_test(
            test_case=TestCases.WEIRD_TOPOLOGY_ONE_NN,
            filename=filename
        )

    def test_weird_topology_two_nn(self):
        # Setting filename
        filename = join(JSON_DIRECTORY, "test_weird_topology_two_nn.json")

        # Test
        self.run_test(
            test_case=TestCases.WEIRD_TOPOLOGY_TWO_NN,
            filename=filename
        )

    def test_weird_topology_three_nn(self):
        # Setting filename
        filename = join(JSON_DIRECTORY, "test_weird_topology_three_nn.json")

        # Test
        self.run_test(
            test_case=TestCases.WEIRD_TOPOLOGY_THREE_NN,
            filename=filename
        )

    def test_weird_topology_four_nn(self):
        # Setting filename
        filename = join(JSON_DIRECTORY, "test_weird_topology_four_nn.json")

        # Test
        self.run_test(
            test_case=TestCases.WEIRD_TOPOLOGY_FOUR_NN,
            filename=filename
        )

    def test_weird_topology_five_nn(self):
        # Setting filename
        filename = join(JSON_DIRECTORY, "test_weird_topology_five_nn.json")

        # Test
        self.run_test(
            test_case=TestCases.WEIRD_TOPOLOGY_FIVE_NN,
            filename=filename
        )


if __name__ == '__main__':
    unittest.main()
