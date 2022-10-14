from neat_python_utility.neat_utility.genome_to_json import clear_abandoned_nodes, discover_neural_network_layers, build_matrices, network_to_json
from neat_python_utility.neat_utility.models.connection import GenomeConnection
from neat_python_utility.neat_utility.models.node import GenomeNode
from random import randint, uniform, choice
from os.path import dirname, join, exists
from os import mkdir
from json import dump

import unittest

JSON_DIRECTORY = join(dirname(__file__), 'json_dumps')
if not exists(JSON_DIRECTORY):
    mkdir(JSON_DIRECTORY)


class JsonDumpTestCase(unittest.TestCase):
    def test_simple_nn(self):
        # Initializing dummy ID inputs
        id_inputs = set()
        id_inputs.add(-1)
        id_inputs.add(-2)
        id_inputs.add(-3)

        # Initializing dummy ID outputs
        id_outputs = set()
        id_outputs.add(4)

        # Initializing dummy connections
        connections_keys = [
            (-1, 1), (-2, 2), (-3, 3),
            (1, 4), (2, 4), (3, 4),
        ]

        connections = list(
            map(lambda key: GenomeConnection(
                identification_number=key,
                enabled=True,
                weight=randint(0, 1024)
            ), connections_keys)
        )

        filtered_connections = clear_abandoned_nodes(id_inputs=id_inputs, id_outputs=id_outputs,
                                                     connections=connections)

        layers = discover_neural_network_layers(
            id_inputs=id_inputs,
            id_outputs=id_outputs,
            connections=filtered_connections
        )

        # Initializing dummy nodes
        nodes_indexes = [1, 2, 3, 4]

        nodes = list(
            map(lambda node_index: GenomeNode(
                node_id=node_index,
                bias=uniform(0, 30),
                activation_function="tanh"
            ), nodes_indexes)
        )

        # Call function
        weights_matrix, bias_matrix, af_matrix, inputs_per_layer = build_matrices(
            id_inputs=id_inputs,
            layers=layers,
            connections=filtered_connections,
            nodes=nodes
        )

        # Converting to JSON
        neural_network_map = network_to_json(
            inputs_name=["Input1", "Input2", "Input3"],
            outputs_name=["Output1"],
            inputs_per_layer=inputs_per_layer,
            layer_nodes=layers,
            weights=weights_matrix,
            biases=bias_matrix,
            activation_functions=af_matrix,
        )

        # Setting filename
        filename = join(JSON_DIRECTORY, "test_simple_nn.json")

        # Creating JSON
        with open(filename, "w") as outfile:
            dump(neural_network_map, outfile, indent=4)

        # Asserting JSON exists
        self.assertEqual(exists(filename), True)

    def test_multiple_outputs_nn(self):
        # Initializing dummy ID inputs
        id_inputs = set()
        id_inputs.add(-1)
        id_inputs.add(-2)
        id_inputs.add(-3)

        # Initializing dummy ID outputs
        id_outputs = set()
        id_outputs.add(4)
        id_outputs.add(5)
        id_outputs.add(6)

        # Initializing dummy connections
        connections_keys = [
            (-1, 1), (-2, 2), (-3, 3),
            (1, 4), (2, 4), (3, 4), (1, 5), (3, 5), (2, 6),
        ]

        connections = list(
            map(lambda key: GenomeConnection(
                identification_number=key,
                enabled=True,
                weight=randint(0, 1024)
            ), connections_keys)
        )

        filtered_connections = clear_abandoned_nodes(id_inputs=id_inputs, id_outputs=id_outputs,
                                                     connections=connections)

        layers = discover_neural_network_layers(
            id_inputs=id_inputs,
            id_outputs=id_outputs,
            connections=filtered_connections
        )

        # Initializing dummy nodes
        nodes_indexes = [1, 2, 3, 4, 5, 6]

        nodes = list(
            map(lambda node_index: GenomeNode(
                node_id=node_index,
                bias=uniform(0, 30),
                activation_function="tanh"
            ), nodes_indexes)
        )

        # Call function
        weights_matrix, bias_matrix, af_matrix, inputs_per_layer = build_matrices(
            id_inputs=id_inputs,
            layers=layers,
            connections=filtered_connections,
            nodes=nodes
        )

        # Converting to JSON
        neural_network_map = network_to_json(
            inputs_name=["Input1", "Input2", "Input3"],
            outputs_name=["Output1", "Output2", "Output3"],
            inputs_per_layer=inputs_per_layer,
            layer_nodes=layers,
            weights=weights_matrix,
            biases=bias_matrix,
            activation_functions=af_matrix,
        )

        # Setting filename
        filename = join(JSON_DIRECTORY, "test_multiple_outputs_nn.json")

        # Creating JSON
        with open(filename, "w") as outfile:
            dump(neural_network_map, outfile, indent=4)

        # Asserting JSON exists
        self.assertEqual(exists(filename), True)

    def test_two_hidden_layers_nn(self):
        # Initializing dummy ID inputs
        id_inputs = set()
        id_inputs.add(-1)
        id_inputs.add(-2)
        id_inputs.add(-3)
        id_inputs.add(-4)

        # Initializing dummy ID outputs
        id_outputs = set()
        id_outputs.add(5)

        # Initializing dummy connections
        connections_keys = [
            (-1, 1), (-2, 2), (-3, 3), (-4, 4),
            (1, 6), (2, 6), (3, 7), (4, 7), (1, 8), (4, 8),
            (6, 9), (7, 9), (7, 10), (8, 10),
            (9, 5), (10, 5)
        ]

        connections = list(
            map(lambda key: GenomeConnection(
                identification_number=key,
                enabled=True,
                weight=randint(0, 1024)
            ), connections_keys)
        )

        filtered_connections = clear_abandoned_nodes(id_inputs=id_inputs, id_outputs=id_outputs,
                                                     connections=connections)

        layers = discover_neural_network_layers(
            id_inputs=id_inputs,
            id_outputs=id_outputs,
            connections=filtered_connections
        )

        # Initializing dummy nodes
        nodes_indexes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        nodes = list(
            map(lambda node_index: GenomeNode(
                node_id=node_index,
                bias=uniform(0, 30),
                activation_function=choice(["tanh", "sigmoid", "relu"])
            ), nodes_indexes)
        )

        # Call function
        weights_matrix, bias_matrix, af_matrix, inputs_per_layer = build_matrices(
            id_inputs=id_inputs,
            layers=layers,
            connections=filtered_connections,
            nodes=nodes
        )

        # Converting to JSON
        neural_network_map = network_to_json(
            inputs_name=["Input1", "Input2", "Input3", "Input4"],
            outputs_name=["Output1"],
            inputs_per_layer=inputs_per_layer,
            layer_nodes=layers,
            weights=weights_matrix,
            biases=bias_matrix,
            activation_functions=af_matrix,
        )

        # Setting filename
        filename = join(JSON_DIRECTORY, "test_two_complete_hidden_layers_nn.json")

        # Creating JSON
        with open(filename, "w") as outfile:
            dump(neural_network_map, outfile, indent=4)

        # Asserting JSON exists
        self.assertEqual(exists(filename), True)

    def test_weird_topology_one_nn(self):
        # Initializing dummy ID inputs
        id_inputs = set()
        id_inputs.add(-1)
        id_inputs.add(-2)
        id_inputs.add(-3)

        # Initializing dummy ID outputs
        id_outputs = set()
        id_outputs.add(4)

        # Initializing dummy connections
        connections_keys = [
            (-1, 1), (-2, 2), (-3, 3),
            (2, 5), (3, 5),
            (1, 6), (5, 6),
            (1, 4), (3, 4), (5, 4), (6, 4)
        ]

        connections = list(
            map(lambda key: GenomeConnection(
                identification_number=key,
                enabled=True,
                weight=randint(0, 1024)
            ), connections_keys)
        )

        filtered_connections = clear_abandoned_nodes(id_inputs=id_inputs, id_outputs=id_outputs,
                                                     connections=connections)

        layers = discover_neural_network_layers(
            id_inputs=id_inputs,
            id_outputs=id_outputs,
            connections=filtered_connections
        )

        # Initializing dummy nodes
        nodes_indexes = [1, 2, 3, 4, 5, 6]

        nodes = list(
            map(lambda node_index: GenomeNode(
                node_id=node_index,
                bias=uniform(-30, 30),
                activation_function=choice(["tanh", "sigmoid", "relu"])
            ), nodes_indexes)
        )

        # Call function
        weights_matrix, bias_matrix, af_matrix, inputs_per_layer = build_matrices(
            id_inputs=id_inputs,
            layers=layers,
            connections=filtered_connections,
            nodes=nodes
        )

        # Converting to JSON
        neural_network_map = network_to_json(
            inputs_name=["Input1", "Input2", "Input3"],
            outputs_name=["Output1"],
            inputs_per_layer=inputs_per_layer,
            layer_nodes=layers,
            weights=weights_matrix,
            biases=bias_matrix,
            activation_functions=af_matrix,
        )

        # Setting filename
        filename = join(JSON_DIRECTORY, "test_weird_topology_nn_one.json")

        # Creating JSON
        with open(filename, "w") as outfile:
            dump(neural_network_map, outfile, indent=4)

        # Asserting JSON exists
        self.assertEqual(exists(filename), True)

    def test_weird_topology_two_nn(self):
        # Initializing dummy ID inputs
        id_inputs = set()
        id_inputs.add(-1)
        id_inputs.add(-2)
        id_inputs.add(-3)

        # Initializing dummy ID outputs
        id_outputs = set()
        id_outputs.add(4)
        id_outputs.add(5)

        # Initializing dummy connections
        connections_keys = [
            (-1, 1), (-2, 2), (-3, 3),
            (1, 6), (2, 6), (3, 6),
            (1, 7), (6, 7),
            (1, 4), (3, 4), (7, 4), (2, 5), (6, 5)
        ]

        connections = list(
            map(lambda key: GenomeConnection(
                identification_number=key,
                enabled=True,
                weight=randint(0, 1024)
            ), connections_keys)
        )

        filtered_connections = clear_abandoned_nodes(id_inputs=id_inputs, id_outputs=id_outputs,
                                                     connections=connections)

        layers = discover_neural_network_layers(
            id_inputs=id_inputs,
            id_outputs=id_outputs,
            connections=filtered_connections
        )

        # Initializing dummy nodes
        nodes_indexes = [1, 2, 3, 4, 5, 6, 7]

        nodes = list(
            map(lambda node_index: GenomeNode(
                node_id=node_index,
                bias=uniform(-30, 30),
                activation_function=choice(["tanh", "sigmoid", "relu"])
            ), nodes_indexes)
        )

        # Call function
        weights_matrix, bias_matrix, af_matrix, inputs_per_layer = build_matrices(
            id_inputs=id_inputs,
            layers=layers,
            connections=filtered_connections,
            nodes=nodes
        )

        # Converting to JSON
        neural_network_map = network_to_json(
            inputs_name=["Input1", "Input2", "Input3"],
            outputs_name=["Output1", "Output2"],
            inputs_per_layer=inputs_per_layer,
            layer_nodes=layers,
            weights=weights_matrix,
            biases=bias_matrix,
            activation_functions=af_matrix,
        )

        # Setting filename
        filename = join(JSON_DIRECTORY, "test_weird_topology_nn_two.json")

        # Creating JSON
        with open(filename, "w") as outfile:
            dump(neural_network_map, outfile, indent=4)

        # Asserting JSON exists
        self.assertEqual(exists(filename), True)

    def test_skipped_input_to_output_nn(self):
        # Initializing dummy ID inputs
        id_inputs = set()
        id_inputs.add(-1)
        id_inputs.add(-2)

        # Initializing dummy ID outputs
        id_outputs = set()
        id_outputs.add(0)

        # Initializing dummy connections
        connections_keys = [
            (-1, 15), (-2, 15),
            (15, 0), (-1, 0), (-2, 0)
        ]

        connections = list(
            map(lambda key: GenomeConnection(
                identification_number=key,
                enabled=True,
                weight=randint(0, 1024)
            ), connections_keys)
        )

        filtered_connections = clear_abandoned_nodes(id_inputs=id_inputs, id_outputs=id_outputs,
                                                     connections=connections)

        layers = discover_neural_network_layers(
            id_inputs=id_inputs,
            id_outputs=id_outputs,
            connections=filtered_connections
        )

        # Initializing dummy nodes
        nodes_indexes = [15, 0]

        nodes = list(
            map(lambda node_index: GenomeNode(
                node_id=node_index,
                bias=uniform(-30, 30),
                activation_function=choice(["tanh", "sigmoid", "relu"])
            ), nodes_indexes)
        )

        # Call function
        weights_matrix, bias_matrix, af_matrix, inputs_per_layer = build_matrices(
            id_inputs=id_inputs,
            layers=layers,
            connections=filtered_connections,
            nodes=nodes
        )

        # Converting to JSON
        neural_network_map = network_to_json(
            inputs_name=["Input1", "Input2"],
            outputs_name=["Output1"],
            inputs_per_layer=inputs_per_layer,
            layer_nodes=layers,
            weights=weights_matrix,
            biases=bias_matrix,
            activation_functions=af_matrix,
        )

        # Setting filename
        filename = join(JSON_DIRECTORY, "test_skipped_input_to_output_nn.json")

        # Creating JSON
        with open(filename, "w") as outfile:
            dump(neural_network_map, outfile, indent=4)

        # Asserting JSON exists
        self.assertEqual(exists(filename), True)

    def test_abandoned_nodes_one_nn(self):
        # Initializing dummy ID inputs
        id_inputs = set()
        id_inputs.add(-1)
        id_inputs.add(-2)

        # Initializing dummy ID outputs
        id_outputs = set()
        id_outputs.add(0)

        # Initializing dummy connections
        connections_keys = [
            (-1, 41), (-2, 41),
            (41, 1073),
            (-1, 0), (41, 0)
        ]

        connections = list(
            map(lambda key: GenomeConnection(
                identification_number=key,
                enabled=True,
                weight=randint(0, 1024)
            ), connections_keys)
        )

        filtered_connections = clear_abandoned_nodes(id_inputs=id_inputs, id_outputs=id_outputs,
                                                     connections=connections)

        layers = discover_neural_network_layers(
            id_inputs=id_inputs,
            id_outputs=id_outputs,
            connections=filtered_connections
        )

        # Initializing dummy nodes
        nodes_indexes = [41, 1073, 0]

        nodes = list(
            map(lambda node_index: GenomeNode(
                node_id=node_index,
                bias=uniform(-30, 30),
                activation_function=choice(["tanh", "sigmoid", "relu"])
            ), nodes_indexes)
        )

        # Call function
        weights_matrix, bias_matrix, af_matrix, inputs_per_layer = build_matrices(
            id_inputs=id_inputs,
            layers=layers,
            connections=filtered_connections,
            nodes=nodes
        )

        # Converting to JSON
        neural_network_map = network_to_json(
            inputs_name=["Input1", "Input2"],
            outputs_name=["Output1"],
            inputs_per_layer=inputs_per_layer,
            layer_nodes=layers,
            weights=weights_matrix,
            biases=bias_matrix,
            activation_functions=af_matrix,
        )

        # Setting filename
        filename = join(JSON_DIRECTORY, "test_abandoned_nodes_one_nn.json")

        # Creating JSON
        with open(filename, "w") as outfile:
            dump(neural_network_map, outfile, indent=4)

        # Asserting JSON exists
        self.assertEqual(exists(filename), True)

    def test_abandoned_nodes_two_nn(self):
        # Initializing dummy ID inputs
        id_inputs = set()
        id_inputs.add(-1)
        id_inputs.add(-2)

        # Initializing dummy ID outputs
        id_outputs = set()
        id_outputs.add(107)

        # Initializing dummy connections
        connections_keys = [
            (-1, 0), (-2, 0), (-1, 378),
            (0, 421),
            (421, 107), (0, 107), (-1, 107), (552, 107)
        ]

        connections = list(
            map(lambda key: GenomeConnection(
                identification_number=key,
                enabled=True,
                weight=randint(0, 1024)
            ), connections_keys)
        )

        filtered_connections = clear_abandoned_nodes(id_inputs=id_inputs, id_outputs=id_outputs,
                                                     connections=connections)

        layers = discover_neural_network_layers(
            id_inputs=id_inputs,
            id_outputs=id_outputs,
            connections=filtered_connections
        )

        # Initializing dummy nodes
        nodes_indexes = [0, 378, 421, 552, 107]

        nodes = list(
            map(lambda node_index: GenomeNode(
                node_id=node_index,
                bias=uniform(-30, 30),
                activation_function=choice(["tanh", "sigmoid", "relu"])
            ), nodes_indexes)
        )

        # Call function
        weights_matrix, bias_matrix, af_matrix, inputs_per_layer = build_matrices(
            id_inputs=id_inputs,
            layers=layers,
            connections=filtered_connections,
            nodes=nodes
        )

        # Converting to JSON
        neural_network_map = network_to_json(
            inputs_name=["Input1", "Input2"],
            outputs_name=["Output1"],
            inputs_per_layer=inputs_per_layer,
            layer_nodes=layers,
            weights=weights_matrix,
            biases=bias_matrix,
            activation_functions=af_matrix,
        )

        # Setting filename
        filename = join(JSON_DIRECTORY, "test_abandoned_nodes_two_nn.json")

        # Creating JSON
        with open(filename, "w") as outfile:
            dump(neural_network_map, outfile, indent=4)

        # Asserting JSON exists
        self.assertEqual(exists(filename), True)


if __name__ == '__main__':
    unittest.main()
