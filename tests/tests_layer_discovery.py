from neat_python_utility.neat_utility.genome_to_json import discover_neural_network_layers
from neat_python_utility.neat_utility.models.connection import GenomeConnection
from random import randint

import unittest


class LayerDiscoveryTestCase(unittest.TestCase):
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

        expected_layers = [
            {1, 2, 3},
            {4}
        ]

        result = discover_neural_network_layers(
            id_inputs=id_inputs,
            id_outputs=id_outputs,
            connections=connections
        )

        # Assertions
        self.assertEqual(result, expected_layers)

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

        expected_layers = [
            {1, 2, 3},
            {4, 5, 6}
        ]

        result = discover_neural_network_layers(
            id_inputs=id_inputs,
            id_outputs=id_outputs,
            connections=connections
        )

        # Assertions
        self.assertEqual(result, expected_layers)

    def test_two_hidden_layer_nn(self):
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

        expected_layers = [
            {1, 2, 3, 4},
            {6, 7, 8},
            {9, 10},
            {5}
        ]

        result = discover_neural_network_layers(
            id_inputs=id_inputs,
            id_outputs=id_outputs,
            connections=connections
        )

        # Assertions
        self.assertEqual(result, expected_layers)

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
            (1, 4), (6, 4), (5, 4), (3, 4)
        ]

        connections = list(
            map(lambda key: GenomeConnection(
                identification_number=key,
                enabled=True,
                weight=randint(0, 1024)
            ), connections_keys)
        )

        expected_layers = [
            {1, 2, 3},
            {5},
            {6},
            {4}
        ]

        result = discover_neural_network_layers(
            id_inputs=id_inputs,
            id_outputs=id_outputs,
            connections=connections
        )

        # Assertions
        self.assertEqual(result, expected_layers)

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

        expected_layers = [
            {1, 2, 3},
            {6},
            {7},
            {4, 5}
        ]

        result = discover_neural_network_layers(
            id_inputs=id_inputs,
            id_outputs=id_outputs,
            connections=connections
        )

        # Assertions
        self.assertEqual(result, expected_layers)


if __name__ == '__main__':
    unittest.main()
