from neat_exporter_package.neat_utility.genome_to_json import discover_neural_network_layers, build_matrices
from neat_exporter_package.neat_utility.models.connection import GenomeConnection
from neat_exporter_package.neat_utility.models.node import GenomeNode
from random import randint, uniform, choice

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

layers = discover_neural_network_layers(
    id_inputs=id_inputs,
    id_outputs=id_outputs,
    connections=connections
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

expected_af_matrix = [
    [nodes[0].activation_function, nodes[1].activation_function, nodes[2].activation_function, nodes[3].activation_function],
    [nodes[5].activation_function, nodes[6].activation_function, nodes[7].activation_function],
    [nodes[8].activation_function, nodes[9].activation_function],
    [nodes[4].activation_function],
]

expected_inputs_per_layer = [
    [-1, -2, -3, -4],
    [1, 2, 3, 4],
    [6, 7, 8],
    [9, 10],
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

