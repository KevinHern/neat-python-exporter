from neat_exporter_package.neat_utility.genome_to_json import discover_neural_network_layers, build_matrices, network_to_json
from neat_exporter_package.neat_utility.models.connection import GenomeConnection
from neat_exporter_package.neat_utility.models.node import GenomeNode
from random import randint, uniform, choice
from json import dump

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

layers = discover_neural_network_layers(
    id_inputs=id_inputs,
    id_outputs=id_outputs,
    connections=connections
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
    connections=connections,
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

with open("test_weird_topology_nn_two.json", "w") as outfile:
    dump(neural_network_map, outfile, indent=4)
