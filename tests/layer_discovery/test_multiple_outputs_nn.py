from neat_exporter_package.neat_exporter.genome_to_json import discover_neural_network_layers
from neat_exporter_package.neat_exporter.models.connection import GenomeConnection
from random import randint

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

print(result)

assert result == expected_layers

