from neat_python_utility.neat_utility.connection_analysis import ConnectionAnalysis
from neat_python_utility.neat_utility.models.connection import GenomeConnection
from random import randint

import unittest


class ConnectionAnalysisTestCase(unittest.TestCase):
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

        expected_useful_connections = connections_keys.copy()
        expected_useful_connections = list(
            filter(
                lambda x: x.identification_number in expected_useful_connections,
                connections
            )
        )

        # Test
        filtered_connections = ConnectionAnalysis(
            id_inputs=id_inputs, id_outputs=id_outputs,
            connections=connections
        ).filter_useful_connections()

        # Assertions
        self.assertEqual(expected_useful_connections, filtered_connections)

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

        expected_useful_connections = connections_keys.copy()
        expected_useful_connections = list(
            filter(
                lambda x: x.identification_number in expected_useful_connections,
                connections
            )
        )

        # Test
        filtered_connections = ConnectionAnalysis(
            id_inputs=id_inputs, id_outputs=id_outputs,
            connections=connections
        ).filter_useful_connections()

        # Assertions
        self.assertEqual(expected_useful_connections, filtered_connections)

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

        expected_useful_connections = connections_keys.copy()
        expected_useful_connections = list(
            filter(
                lambda x: x.identification_number in expected_useful_connections,
                connections
            )
        )

        # Test
        filtered_connections = ConnectionAnalysis(
            id_inputs=id_inputs, id_outputs=id_outputs,
            connections=connections
        ).filter_useful_connections()

        # Assertions
        self.assertEqual(expected_useful_connections, filtered_connections)

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
            (-1, 0), (15, 0), (-2, 0),
        ]

        connections = list(
            map(lambda key: GenomeConnection(
                identification_number=key,
                enabled=True,
                weight=randint(0, 1024)
            ), connections_keys)
        )

        expected_useful_connections = connections_keys.copy()
        expected_useful_connections = list(
            filter(
                lambda x: x.identification_number in expected_useful_connections,
                connections
            )
        )

        # Test
        filtered_connections = ConnectionAnalysis(
            id_inputs=id_inputs, id_outputs=id_outputs,
            connections=connections
        ).filter_useful_connections()

        # Assertions
        self.assertEqual(expected_useful_connections, filtered_connections)

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

        expected_useful_connections = [
            (-1, 41), (-2, 41),
            (-1, 0), (41, 0)
        ]
        expected_useful_connections = list(
            filter(
                lambda x: x.identification_number in expected_useful_connections,
                connections
            )
        )

        # Test
        filtered_connections = ConnectionAnalysis(
            id_inputs=id_inputs, id_outputs=id_outputs,
            connections=connections
        ).filter_useful_connections()

        # Assertions
        self.assertEqual(expected_useful_connections, filtered_connections)

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

        expected_useful_connections = [
            (-1, 0), (-2, 0),
            (0, 421),
            (421, 107), (0, 107), (-1, 107)
        ]
        expected_useful_connections = list(
            filter(
                lambda x: x.identification_number in expected_useful_connections,
                connections
            )
        )

        # Test
        filtered_connections = ConnectionAnalysis(
            id_inputs=id_inputs, id_outputs=id_outputs,
            connections=connections
        ).filter_useful_connections()

        # Assertions
        self.assertEqual(expected_useful_connections, filtered_connections)

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

        expected_useful_connections = connections_keys.copy()
        expected_useful_connections = list(
            filter(
                lambda x: x.identification_number in expected_useful_connections,
                connections
            )
        )

        # Test
        filtered_connections = ConnectionAnalysis(
            id_inputs=id_inputs, id_outputs=id_outputs,
            connections=connections
        ).filter_useful_connections()

        # Assertions
        self.assertEqual(expected_useful_connections, filtered_connections)

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

        expected_useful_connections = connections_keys.copy()
        expected_useful_connections = list(
            filter(
                lambda x: x.identification_number in expected_useful_connections,
                connections
            )
        )

        # Test
        filtered_connections = ConnectionAnalysis(
            id_inputs=id_inputs, id_outputs=id_outputs,
            connections=connections
        ).filter_useful_connections()

        # Assertions
        self.assertEqual(expected_useful_connections, filtered_connections)

    def test_weird_topology_three_nn(self):
        # Initializing dummy ID inputs
        id_inputs = set()
        id_inputs.add(-1)
        id_inputs.add(-2)

        # Initializing dummy ID outputs
        id_outputs = set()
        id_outputs.add(0)

        # Initializing dummy connections
        connections_keys = [
            (-1, 101), (-2, 101),
            (101, 0), (101, 1751), (-2, 1751),
            (-2, 1718), (1751, 1718),
            (-2, 348), (1718, 348)
        ]

        connections = list(
            map(lambda key: GenomeConnection(
                identification_number=key,
                enabled=True,
                weight=randint(0, 1024)
            ), connections_keys)
        )

        expected_useful_connections = [
            (-1, 101), (-2, 101), (101, 0)
        ]
        expected_useful_connections = list(
            filter(
                lambda x: x.identification_number in expected_useful_connections,
                connections
            )
        )

        # Test
        filtered_connections = ConnectionAnalysis(
            id_inputs=id_inputs, id_outputs=id_outputs,
            connections=connections
        ).filter_useful_connections()

        # Assertions
        self.assertEqual(expected_useful_connections, filtered_connections)

    def test_weird_topology_four_nn(self):
        # Initializing dummy ID inputs
        id_inputs = set()
        id_inputs.add(-1)
        id_inputs.add(-2)

        # Initializing dummy ID outputs
        id_outputs = set()
        id_outputs.add(0)

        # Initializing dummy connections
        connections_keys = [
            (-1, 451), (-2, 451),
            (-1, 0), (-2, 0), (451, 0),
            (0, 1319),
            (-1, 1457), (1583, 1457),
            (1457, 1609),
            (451, 967), (-1, 967), (1457, 967), (1609, 967)
        ]

        connections = list(
            map(lambda key: GenomeConnection(
                identification_number=key,
                enabled=True,
                weight=randint(0, 1024)
            ), connections_keys)
        )

        expected_useful_connections = [
            (-1, 451), (-2, 451),
            (-1, 0), (-2, 0), (451, 0)
        ]
        expected_useful_connections = list(
            filter(
                lambda x: x.identification_number in expected_useful_connections,
                connections
            )
        )

        # Test
        filtered_connections = ConnectionAnalysis(
                    id_inputs=id_inputs, id_outputs=id_outputs,
                    connections=connections
                ).filter_useful_connections()

        # Assertions
        self.assertEqual(expected_useful_connections, filtered_connections)


if __name__ == '__main__':
    unittest.main()
