from neat_exporter_package.neat_utility.models.neat_agent import NeatAgent


class XorAgent(NeatAgent):
    def __init__(self, genome, neural_network):
        super().__init__()
        self.genome = genome
        self.brain = neural_network

    def __lt__(self, other):
        return self.genome.fitness > other.genome.fitness

    def __eq__(self, other):
        return self.genome.fitness == other.genome.fitness

    def think(self, inputs):
        # Forward pass of the neural network
        return self.brain.activate(inputs)
