# Models
from .models import *

# Utils
import json


def get_network_metadata(genome, config):
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

    # Getting IDs of inputs
    id_inputs = set()
    for input_key in config.genome_config.input_keys:
        id_inputs.add(input_key)

    # Getting IDs of outputs
    id_outputs = set()
    for input_key in config.genome_config.output_keys:
        id_outputs.add(input_key)

    return id_inputs, id_outputs, nodes, connections


def export_genome_to_json(filename, config, genome, inputs_names, outputs_names):
    # Getting NN metadata
    id_inputs, id_outputs, nodes, connections = get_network_metadata(genome=genome, config=config)

    # Instantiating a GenomeAnalysis object
    genome_analyzer_object = GenomeAnalyzer(
        id_inputs=id_inputs, id_outputs=id_outputs,
        connections=connections, nodes=nodes
    )

    # Filter connections and get rid of useless and abandoned nodes
    genome_analyzer_object.filter_useful_connections()

    # Discover all possible paths
    genome_analyzer_object.discover_all_connection_paths()

    # Construct layers and deduce inputs per layer
    genome_analyzer_object.construct_layers()

    # Build weights, biases and activation functions matrices
    genome_analyzer_object.construct_matrices()

    # Get a Map of the neural network
    neural_network_map = genome_analyzer_object.network_to_map(inputs_name=inputs_names, outputs_name=outputs_names)

    # Convert the map into a JSON and save it
    with open(filename, "w") as outfile:
        json.dump(neural_network_map, outfile, indent=4)
