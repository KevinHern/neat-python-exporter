import numpy as np


class GenomeAnalyzer:
    def __init__(self, id_inputs, id_outputs, connections, nodes):
        # Useful Public Variables
        self.id_inputs = id_inputs
        self.id_outputs = id_outputs
        self.connections = connections
        self.nodes = nodes

        self.all_paths = None

        self.layers = None
        self.inputs_per_layer = None

        self.weights_matrix = None
        self.biases_matrix = None
        self.activation_functions_matrix = None

        # Parameters used to filter connections
        self._end_layer = None
        self._analyze_connection_position = None
        self._are_connections_filtered = False

        # Parameters used to find all possible NN paths
        self._all_paths_found = False

        # Parameters used to deduce layers from paths
        self._created_layers = False

        # Parameters used for matrices deduction
        self._deduced_matrices = False

    def _connection_discovery_recursion(self, connection):
        if connection.identification_number[self._analyze_connection_position] in self._end_layer:
            # Check if the connection ends with a node in end_layer, if so, then the connection is useful
            return [connection]
        else:
            # Discover connections
            connections_discovered = list(
                filter(
                    # Connection position is either 0 or 1
                    lambda x: connection.identification_number[self._analyze_connection_position] ==
                              x.identification_number[1 - self._analyze_connection_position],
                    self.connections
                )
            )

            if connections_discovered:
                useful_connections = []

                # More connections found, keep exploring
                for connection_discovered in connections_discovered:
                    result = self._connection_discovery_recursion(
                        connection=connection_discovered
                    )

                    # Add found connections
                    if result:
                        result.append(connection_discovered)
                        useful_connections += list(
                            filter(lambda x: x not in useful_connections, result)
                        )

                return useful_connections
            else:
                # Dead end, connection is useless
                return []

    def _top_down_discover_connections(self, input_connection):
        # Make sure the parameters are at default values
        assert None in [self._analyze_connection_position, self._end_layer]

        # Setting parameters
        self._analyze_connection_position = 1
        self._end_layer = self.id_outputs

        # Filter
        connections = self._connection_discovery_recursion(
            connection=input_connection
        )

        # Resetting parameters
        self._analyze_connection_position = None
        self._end_layer = None

        return connections

    def _bottom_up_discover_connections(self, output_connection):
        # First, make sure the parameters are at default values
        assert None in [self._analyze_connection_position, self._end_layer]

        # Setting parameters
        self._analyze_connection_position = 0
        self._end_layer = self.id_inputs

        # Filter
        connections = self._connection_discovery_recursion(
            connection=output_connection
        )

        # Resetting parameters
        self._analyze_connection_position = None
        self._end_layer = None

        return connections

    def filter_useful_connections(self):
        # Sanity Check: Make sure connections have not been filtered yet
        assert not self._are_connections_filtered

        # TOP DOWN ANALYSIS
        initial_top_down_connections = list(
            filter(lambda x: x.identification_number[0] in self.id_inputs, self.connections)
        )
        top_down_useful_connections = []

        for connection in initial_top_down_connections:
            # Get outputs
            discovered_connections = self._top_down_discover_connections(input_connection=connection) + [connection]

            # Add connections
            top_down_useful_connections += list(
                filter(lambda x: x not in top_down_useful_connections, discovered_connections)
            )

        top_down_useful_connections = list(map(lambda x: x.identification_number, top_down_useful_connections))

        # BOTTOM UP ANALYSIS
        initial_bottom_up_connections = list(
            filter(lambda x: x.identification_number[1] in self.id_outputs, self.connections)
        )
        bottom_up_useful_connections = []

        for connection in initial_bottom_up_connections:
            # Get outputs
            discovered_connections = self._bottom_up_discover_connections(output_connection=connection) + [connection]

            # Add connections
            bottom_up_useful_connections += list(
                filter(lambda x: x not in bottom_up_useful_connections, discovered_connections)
            )
        bottom_up_useful_connections = list(map(lambda x: x.identification_number, bottom_up_useful_connections))

        # -----

        # Obtaining the useful connections
        useful_connections = list(
            filter(
                lambda x: x.identification_number in bottom_up_useful_connections and
                          x.identification_number in top_down_useful_connections,
                self.connections
            )
        )

        # Change self parameters
        self.connections = useful_connections
        self._are_connections_filtered = True

    def _discover_path(self, connection, discovered_path):
        if connection.identification_number[1] in self.id_outputs:
            # End of path, just append the last connection
            discovered_path.append(connection)
            self.all_paths.append(discovered_path.copy())
        else:
            # Discover next connections
            next_connections = list(
                filter(
                    lambda x: connection.identification_number[1] == x.identification_number[0],
                    self.connections
                )
            )

            # Append current connection to the discovered path so far
            discovered_path.append(connection)

            # More connections found, keep exploring
            for next_connection in next_connections:
                self._discover_path(
                    connection=next_connection,
                    discovered_path=discovered_path.copy()
                )

    def discover_all_connection_paths(self):
        # Make sure connections have been filtered
        assert self._are_connections_filtered and not self._all_paths_found

        # Initialize Paths variable
        self.all_paths = []

        # Filter all connections that contain the inputs
        initial_connections = list(
            filter(lambda x: x.identification_number[0] in self.id_inputs, self.connections)
        )

        # Iterate for each connection
        for connection in initial_connections:
            # Get input's path to output
            self._discover_path(
                connection=connection,
                discovered_path=[]
            )

        # Set Flag
        self._all_paths_found = True

    def construct_layers(self):
        # Sanity Check
        assert self._all_paths_found and not self._created_layers

        # Sort all paths based on length and then reverse
        self.all_paths.sort(key=len)
        self.all_paths.reverse()
        sorted_path_keys = list(
            map(
                lambda x: list(
                    map(
                        lambda y: y.identification_number,
                        x
                    )
                ),
                self.all_paths
            )
        )

        # Extracting the or one of the largest paths to use as reference for the picture of the net
        reference_path = sorted_path_keys.pop(0)
        reference_path.reverse()
        layers = list(map(lambda x: [x[1]], reference_path))
        layers[0] = list(self.id_outputs)

        # Iterate over all paths
        while sorted_path_keys:
            # Get next path and reverse it. Analysis start from id_outputs
            reference_path = sorted_path_keys.pop(0)
            reference_path.reverse()

            # Check length of path to help get a vague idea if extra stuff needs to be done
            if len(layers) == len(reference_path):
                # Iterate all connections
                for index in range(len(reference_path)):
                    # Check if output is already in layer
                    if reference_path[index][1] not in layers[index]:
                        layers[index].append(reference_path[index][1])
            else:
                # Getting nodes and inputs
                path_layer_nodes = list(map(lambda x: x[1], reference_path))

                # Iterate all connections
                for node_index, node in enumerate(path_layer_nodes):
                    # Initializing tracker
                    found_in_layer = None

                    # Iterate over all layers and check if the node is already in one layer
                    for layer_index, layer in enumerate(layers):
                        # Check if node is already in layer
                        if node in layer:
                            found_in_layer = layer_index
                            break

                    if found_in_layer is None:
                        # This means the node wasn't found, just add it to the respective layer for the moment
                        layers[node_index].append(node)
                    elif found_in_layer < node_index:
                        # This means the new path is longer and has more connections in between.
                        # Moving the node to the new layer
                        layers[found_in_layer].remove(node)
                        layers[node_index].append(node)

        # Clean Up
        self.layers = list(map(lambda x: set(x), layers))

        # Getting inputs for each layer
        self.inputs_per_layer = []
        for layer in self.layers:
            # Getting all connections that contain as output the nodes of the layer
            inputs_of_layer = list(
                map(
                    lambda x: x.identification_number[0],
                    filter(lambda x: x.identification_number[1] in layer, self.connections)
                )
            )

            # Removing duplicates
            inputs_of_layer = list(dict.fromkeys(inputs_of_layer))

            # Sorting inputs in the layer_inputs, so everything is still in order
            negative_id_inputs = list(
                filter(lambda x: x < 0, inputs_of_layer)
            )
            negative_id_inputs.sort(reverse=True)
            positive_id_inputs = list(
                filter(lambda x: x >= 0, inputs_of_layer)
            )
            positive_id_inputs.sort(reverse=False)
            self.inputs_per_layer.append(negative_id_inputs + positive_id_inputs)

        # Clean Up
        self.layers.reverse()
        self.inputs_per_layer.reverse()

        # Setting flag
        self._created_layers = True

    def construct_matrices(self):
        # Sanity Check: Make sure layers have been deduced
        assert self._created_layers and not self._deduced_matrices

        # Initialize matrices
        self.weights_matrix = []
        self.biases_matrix = []
        self.activation_functions_matrix = []

        # Iterate over all layers
        for layer, layer_inputs in zip(self.layers, self.inputs_per_layer):
            # Converting layer to list and sort
            layer = list(layer)
            layer.sort()

            # Count nodes and number of inputs
            number_nodes = len(layer)
            number_layer_inputs = len(layer_inputs)

            # Initializing weights and bias matrices of this layer
            layer_weights = np.empty([number_nodes, number_layer_inputs], dtype=float)
            layer_weights.fill(0)
            layer_weights = list(layer_weights)

            # Filling up weights matrix
            for node_index, node_id in enumerate(layer):
                # Obtain all connections that contain the node_id
                node_connections = list(
                    filter(lambda connection: connection.identification_number[1] == node_id, self.connections)
                )

                # Convert numpy to just list
                layer_weights[node_index] = list(layer_weights[node_index])

                # Fill up the weights
                for node_connection in node_connections:
                    # Obtain the index of the respective column
                    index_weight = layer_inputs.index(node_connection.identification_number[0])
                    layer_weights[node_index][index_weight] = node_connection.weight if node_connection.enabled else 0

            # Initializing layer biases and activation function matrices
            layer_biases = np.empty(number_nodes, dtype=float)
            layer_biases.fill(0)
            layer_biases = list(layer_biases)
            layer_activation_functions = np.empty(number_nodes, dtype=str)
            layer_activation_functions.fill("")
            layer_activation_functions = list(layer_activation_functions)

            # Filtering the nodes that are in the current layer
            layer_nodes = list(
                filter(lambda node: node.node_id in layer, self.nodes)
            )

            # Iterate over all nodes
            for layer_node in layer_nodes:
                node_index = layer.index(layer_node.node_id)
                layer_biases[node_index] = layer_node.bias
                layer_activation_functions[node_index] = str(layer_node.activation_function)

            # Append layers
            self.weights_matrix.append(layer_weights)
            self.biases_matrix.append(layer_biases)
            self.activation_functions_matrix.append(layer_activation_functions)

        # Set flag
        self._deduced_matrices = True

        # Sanity check: Make sure the length of weights, biases and activation functions are the same
        assert len(self.weights_matrix) == len(self.biases_matrix) == len(self.activation_functions_matrix)

        # Sanity Check: Make sure the output layer of all matrices has the same length as the id_outputs
        assert len(self.weights_matrix[-1]) == len(self.biases_matrix[-1]) \
               == len(self.activation_functions_matrix[-1]) == len(self.id_outputs)

    def network_to_map(self, inputs_name, outputs_name):
        # Sanity Checking
        assert len(self.id_inputs) == len(inputs_name)
        assert len(self.id_outputs) == len(outputs_name)

        # Formatting inputs
        formatted_inputs = []
        for id_input, input_name in zip(self.inputs_per_layer[0], inputs_name):
            formatted_inputs.append("{} ({})".format(input_name, id_input))

        # Formatting Outputs
        formatted_outputs = []
        id_outputs = list(self.layers[-1])
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
        for layer_index, layer_nodes in enumerate(self.layers):
            # Convert to list and sort
            current_layer_nodes = list(layer_nodes)
            current_layer_nodes.sort()

            neural_network_map["layers"].append(
                {
                    "layer": layer_index,
                    "id_node_inputs": self.inputs_per_layer[layer_index],
                    "id_nodes": current_layer_nodes,
                    "weights": list(self.weights_matrix[layer_index]),
                    "biases": list(self.biases_matrix[layer_index]),
                    "afunctions": list(self.activation_functions_matrix[layer_index])
                }
            )

        return neural_network_map
