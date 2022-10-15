
class ConnectionAnalysis:
    def __init__(self, id_inputs, id_outputs, connections):
        self.id_inputs = id_inputs
        self.id_outputs = id_outputs
        self.connections = connections

    def _top_down_discover_connections(self, connection):
        if connection.identification_number[1] in self.id_outputs:
            # Check if the connection ends with an output, if so, then the connection is useful
            return [connection]
        else:
            # Discover connections
            connections_discovered = list(
                filter(
                    lambda x: connection.identification_number[1] == x.identification_number[0],
                    self.connections
                )
            )

            if connections_discovered:
                useful_connections = []

                # More connections found, keep exploring
                for connection in connections_discovered:
                    result = self._top_down_discover_connections(connection=connection)

                    # Add found connections
                    if result:
                        result.append(connection)
                        useful_connections += list(
                            filter(lambda x: x not in useful_connections, result)
                        )

                return useful_connections
            else:
                # Dead end, connection is useless
                return []

    def _bottom_up_discover_connections(self, connection):
        if connection.identification_number[0] in self.id_inputs:
            # Check if the connection starts with an input, if so, then the input_connection is useful
            return [connection]
        else:
            # Discover connections
            connections_discovered = list(
                filter(
                    lambda x: connection.identification_number[0] == x.identification_number[1],
                    self.connections
                )
            )

            if connections_discovered:
                useful_connections = []

                # More connections found, keep exploring
                for connection in connections_discovered:
                    result = self._bottom_up_discover_connections(connection=connection)

                    # Add found connections
                    if result:
                        result.append(connection)
                        useful_connections += list(
                            filter(lambda x: x not in useful_connections, result)
                        )

                return useful_connections
            else:
                # Dead end, connection is useless
                return []

    def filter_useful_connections(self):
        # TOP DOWN ANALYSIS
        initial_top_down_connections = list(
            filter(lambda x: x.identification_number[0] in self.id_inputs, self.connections)
        )
        top_down_useful_connections = []

        for connection in initial_top_down_connections:
            # Get outputs
            discovered_connections = self._top_down_discover_connections(connection=connection) + [connection]

            # print(list(map(lambda x: x.identification_number, top_down_useful_connections)))

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
            discovered_connections = self._bottom_up_discover_connections(connection=connection) + [connection]

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

        return useful_connections
