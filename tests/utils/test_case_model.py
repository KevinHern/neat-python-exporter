from neat_python_utility.neat_utility.models.connection import GenomeConnection
from neat_python_utility.neat_utility.models.node import GenomeNode
from random import randint, uniform, choice


class TestCaseModel:
    def __init__(self, id_inputs, id_outputs, connections_keys, nodes_indexes):
        self.id_inputs = id_inputs
        self.id_outputs = id_outputs

        self.connections_keys = connections_keys
        self.connections = list(
            map(lambda key: GenomeConnection(
                identification_number=key,
                enabled=True,
                weight=randint(0, 1024)
            ), connections_keys)
        )

        self.nodes = list(
            map(lambda node_index: GenomeNode(
                node_id=node_index,
                bias=uniform(-30, 30),
                activation_function=choice(["tanh", "sigmoid", "relu"])
            ), nodes_indexes)
        )