import numpy as np

# Models
from neat_exporter_package.neat_utility.models.connection import GenomeConnection
from neat_exporter_package.neat_utility.models.node import GenomeNode

# Utils
import json


def get_network_metadata(config):
    # Get number of inputs
    number_inputs = len(config.genome_config.input_keys)

    # Get number of outputs
    number_outputs = len(config.genome_config.output_keys)

    # Getting IDs of inputs
    id_inputs = set()
    for input_key in config.genome_config.input_keys:
        id_inputs.add(input_key)

    # Getting IDs of outputs
    id_outputs = set()
    for input_key in config.genome_config.output_keys:
        id_outputs.add(input_key)

    return number_inputs, number_outputs, id_inputs, id_outputs


def deconstruct_network(genome):
    '''
        genome.nodes is a dictionary!
        {<Int: Node Key>: <Node Object>}

        A Node Object contains the following:
        - .key: The ID of the node
        - .bias: The bias of the node
        - .activation: Activation function used by the net
        - .aggregation: What the neuron does with the values
        - .response: Value coefficient for Weights and biases for the genetic algorithm
    '''
    # Getting all nodes
    nodes = set()
    for node in genome.nodes.values():
        found_node = GenomeNode(
            node_id=node.key,
            bias=node.bias,
            activation_function=node.activation
        )
        nodes.add(found_node)

    # Getting all connections
    connections = set()
    for cg in genome.connections.values():
        found_connection = GenomeConnection(
            identification_number=cg.key,
            enabled=cg.enabled,
            weight=cg.weight
        )
        connections.add(found_connection)

    return nodes, connections


def discover_neural_network_layers(id_inputs, id_outputs, connections):
    # Obtain all the connections' identification_number
    connections_identification_numbers = []
    for connections in connections:
        connections_identification_numbers.append(connections.identification_number)

    # Initialize Layers
    layers = [id_outputs]

    # Do a Backwards analysis
    layer_outputs = id_outputs
    input_layer = set()

    while connections_identification_numbers:
        # Initialize variables
        new_layer = set()
        connections_to_delete = []

        # Check connections to discover the next layers
        for connection in connections_identification_numbers:
            # If the output of the connection_input is in the layer_inputs AND it isn't an input
            # node, then add it to the new layer
            if connection[1] in layer_outputs:
                # If a node is not linked to the inputs and hasn't been discovered, add it to the new layer
                if not (connection[0] in id_inputs or connection[0] in new_layer):
                    new_layer.add(connection[0])
                else:
                    # If a node is linked to the inputs and hasn't been discovered as the input layer, add it
                    if connection[0] in id_inputs and connection[1] not in input_layer:
                        input_layer.add(connection[1])
                connections_to_delete.append(connection)

        # Update connections list
        connections_identification_numbers = list(
            filter(lambda connection: connection not in connections_to_delete, connections_identification_numbers)
        )

        if new_layer:
            # Add layer to layers
            layers.append(new_layer)

            # Update output_layers
            layer_outputs = new_layer.copy()

    # Append input layer
    if input_layer not in layers:
        layers.append(input_layer)

    # Reverse Layers list, so input layer is at the top
    layers.reverse()

    # ------------- CLEAN UP

    # Do a forward pass and eliminate repeating nodes over the layers
    '''
        EXAMPLE:
        [{1, 2, 3}, {1, 2, 3, 5}, {1, 5, 6}, {4}]
        
        It is necessary to do a forward analysis and clean up layers 2 and 3 due for repeating nodes.
    '''

    # Initialize lists
    used_nodes = []
    tidy_layers = []

    # Iterating all layers
    for layer in layers:
        # Checking the current nodes and adding the ones that are not repeated
        cleaned_layer = list(
            filter(lambda node: node not in used_nodes, list(layer))
        )

        # Sorting layer
        cleaned_layer.sort()

        # Appending cleaned up layer
        tidy_layers.append(set(cleaned_layer))

        # Adding to used nodes
        used_nodes += list(cleaned_layer)

    return tidy_layers


def build_matrices(id_inputs, layers, connections, nodes):
    # Initialize matrices
    weights_matrix = []
    bias_matrix = []
    activation_function_matrix = []
    inputs_per_layer = []

    # Initialize inputs
    layer_inputs = list(id_inputs.copy())
    layer_inputs.sort(reverse=True)

    # Iterate all over the layers
    for layer in layers:
        # Initializing some counters
        number_nodes = len(layer)

        # Converting layer to list and sort
        layer = list(layer)
        layer.sort()

        # Filter the potential connections
        potential_connections = list(
            filter(lambda connection: connection.identification_number[1] in layer, connections)
        )

        # Discovering extra inputs
        for connection in potential_connections:
            if connection.identification_number[0] not in layer_inputs:
                layer_inputs.append(connection.identification_number[0])

                # Sorting inputs each time a new input has been discovered
                layer_inputs.sort()
        number_inputs = len(layer_inputs)

        # Appending the ID of the nodes that this layer accepts as inputs
        inputs_per_layer.append(layer_inputs)

        # Initializing weights and bias matrices of this layer
        layer_weights = np.empty([number_nodes, number_inputs], dtype=float)
        layer_weights.fill(0)
        layer_weights = list(layer_weights)

        # Filling up weights matrix
        for node_index, node_id in enumerate(layer):
            # Obtain all connections that contain the node_id
            node_connections = list(
                filter(lambda connection: connection.identification_number[1] == node_id, potential_connections)
            )

            # Convert numpy to just list
            layer_weights[node_index] = list(layer_weights[node_index])

            # Fill up the weights
            for node_connection in node_connections:
                # Obtain the index of the respective column
                index_weight = layer_inputs.index(node_connection.identification_number[0])
                layer_weights[node_index][index_weight] = node_connection.weight if node_connection.enabled else 0

        # ------------

        # Initializing layer biases
        layer_biases = np.empty(number_nodes, dtype=float)
        layer_biases.fill(0)
        layer_biases = list(layer_biases)
        layer_activation_functions = np.empty(number_nodes, dtype=str)
        layer_activation_functions.fill("")
        layer_activation_functions = list(layer_activation_functions)

        # Filtering the nodes that are in the current layer
        layer_nodes = list(
            filter(lambda node: node.node_id in layer, nodes)
        )

        # Iterate all over the nodes
        for layer_node in layer_nodes:
            node_index = layer.index(layer_node.node_id)
            layer_biases[node_index] = layer_node.bias
            layer_activation_functions[node_index] = str(layer_node.activation_function)

        # Append layers
        weights_matrix.append(layer_weights)
        bias_matrix.append(layer_biases)
        activation_function_matrix.append(layer_activation_functions)

        # Changing layer inputs
        layer_inputs = layer.copy()

    return weights_matrix, bias_matrix, activation_function_matrix, inputs_per_layer


def network_to_json(
        inputs_name, outputs_name,
        inputs_per_layer, layer_nodes,
        weights, biases, activation_functions
):
    # Make sure everything is right
    assert len(weights) == len(biases) == len(activation_functions)

    # Formatting inputs
    formatted_inputs = []
    for id_input, input_name in zip(inputs_per_layer[0], inputs_name):
        formatted_inputs.append("{} ({})".format(input_name, id_input))

    # Formatting Outputs
    formatted_outputs = []
    id_outputs = list(layer_nodes[-1])
    id_outputs.sort()
    for id_output, output_name in zip(id_outputs, outputs_name):
        formatted_outputs.append("{} ({})".format(output_name, id_output))

    # Initializing Map
    neural_network_map = {
        "inputs": formatted_inputs,
        "outputs": formatted_outputs,
        "layers": [],
    }

    # Filling weights
    for layer_index in range(len(weights)):
        # Convert to list and sort
        current_layer_nodes = list(layer_nodes[layer_index])
        current_layer_nodes.sort()

        neural_network_map["layers"].append(
            {
                "layer": layer_index,
                "id_node_inputs": inputs_per_layer[layer_index],
                "id_nodes": current_layer_nodes,
                "weights": list(weights[layer_index]),
                "biases": list(biases[layer_index]),
                "afunctions": list(activation_functions[layer_index])
            }
        )

    return neural_network_map


def convert_genome_to_json(filename, config, genome, inputs_name, outputs_name):
    # Getting NN metadata
    number_inputs, number_outputs, id_inputs, id_outputs = get_network_metadata(config=config)

    # Getting NN Connections and Nodes
    nodes, connections = deconstruct_network(genome=genome)

    # Unpacking the resulting network's layers
    layers = discover_neural_network_layers(
        id_inputs=id_inputs,
        id_outputs=id_outputs,
        connections=connections
    )

    # Calculate weight, bias and activation function matrices, and return the inputs needed per layer
    weights_matrix, bias_matrix, activation_function_matrix, inputs_per_layer = build_matrices(
        id_inputs=id_inputs,
        layers=layers,
        connections=connections,
        nodes=nodes
    )

    # Get a Map of the neural network that contains its metadata
    neural_network_map = network_to_json(
        inputs_name=inputs_name,
        outputs_name=outputs_name,
        inputs_per_layer=inputs_per_layer,
        layer_nodes=layers,
        weights=weights_matrix,
        biases=bias_matrix,
        activation_functions=activation_function_matrix
    )

    # Convert the map into a JSON and save it
    with open(filename, "w") as outfile:
        json.dump(neural_network_map, outfile, indent=4)
