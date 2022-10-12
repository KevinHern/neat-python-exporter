# AI
from neat.nn import FeedForwardNetwork

# Models
from neat_exporter_package.tests.simulation.xor_agent import XorAgent


class XorGateGame:
    INPUTS = (
        (0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)
    )
    XOR_OUTPUTS = (
        (0.0,), (1.0,), (1.0,), (0.0,)
    )

    def __init__(self):
        self.generation = 0

    @staticmethod
    def create_agent(genome, neural_network):
        return XorAgent(
            genome=genome,
            neural_network=neural_network
        )

    def run(self, generation):
        for xor_agent in generation:
            for inputs, outputs in zip(XorGateGame.INPUTS, XorGateGame.XOR_OUTPUTS):
                output = xor_agent.brain.activate(inputs)
                xor_agent.genome.fitness -= (output[0] - outputs[0]) ** 2

    def simulation(self, genomes, config):
        # Increase generation counter
        self.generation += 1

        # Create Population
        new_generation = []
        for genome_id, genome in genomes:
            # Setting fitness to 4, we will minimize fitness in this example
            genome.fitness = 4.0

            # Creating Agent
            brain = FeedForwardNetwork.create(genome, config)
            xor_agent = XorGateGame.create_agent(genome=genome, neural_network=brain)

            # Appending agent
            new_generation.append(xor_agent)

        # Running simulation
        self.run(generation=new_generation)

        # Sort agents by fitness
        sorted_agents = sorted(new_generation)

        # Return current fittest
        return self.generation, sorted_agents[0]
