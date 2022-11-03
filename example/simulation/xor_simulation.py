# AI
from neat.nn import FeedForwardNetwork

# Models
from neat_python_utility.example.simulation.xor_agent import XorAgent


class XorGateGame:
    INPUTS = (
        (0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)
    )
    XOR_OUTPUTS = (
        (0.0,), (1.0,), (1.0,), (0.0,)
    )

    def __init__(self):
        # THIS IS A MANDATORY VARIABLE, CREATE A GENERATION VARIABLE
        self.generation = 0

    @staticmethod
    def create_agent(genome, neural_network):
        return XorAgent(
            genome=genome,
            neural_network=neural_network
        )

    def run(self, generation):
        # Let agents perform their action
        for xor_agent in generation:
            for xor_input, xor_output in zip(XorGateGame.INPUTS, XorGateGame.XOR_OUTPUTS):
                # Agent processes the inputs and outputs a result
                output = xor_agent.brain.activate(xor_input)

                # Do something with the result, in this case, calculate loss
                xor_agent.loss += (output[0] - xor_output[0]) ** 2

            # Modify fitness
            xor_agent.genome.fitness -= xor_agent.loss

    # This is the equivalent of the eval_genomes(genomes, config) function
    def simulation(self, genomes, config):
        # Increase generation counter
        self.generation += 1

        # Create Population
        new_generation = []
        for genome_id, genome in genomes:
            # Setting fitness to 4, we will maximize fitness in this example
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

        # Return current generation and fittest
        return self.generation, sorted_agents[0]
