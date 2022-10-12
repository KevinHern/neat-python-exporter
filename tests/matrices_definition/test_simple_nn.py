from neat_exporter_package.neat_utility.genome_to_json import discover_neural_network_layers, build_matrices
from neat_exporter_package.neat_utility.models.connection import GenomeConnection
from neat_exporter_package.neat_utility.models.node import GenomeNode
from random import randint, uniform

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

layers = discover_neural_network_layers(
    id_inputs=id_inputs,
    id_outputs=id_outputs,
    connections=connections
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

expected_af_matrix = [
    [nodes[0].activation_function, nodes[1].activation_function, nodes[2].activation_function],
    [nodes[3].activation_function],
]

expected_inputs_per_layer = [
    [-1, -2, -3],
    [1, 2, 3],
]

# Call function
weights_matrix, bias_matrix, af_matrix, inputs_per_layer = build_matrices(
    id_inputs=id_inputs,
    layers=layers,
    connections=connections,
    nodes=nodes
)

# Assertions
assert expected_weights_matrix == weights_matrix
assert expected_biases_matrix == bias_matrix
assert expected_af_matrix == af_matrix
assert expected_inputs_per_layer == inputs_per_layer

