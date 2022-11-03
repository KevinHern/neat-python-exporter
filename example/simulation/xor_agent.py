# Models
from neat_python_utility.neat_utility import NeatAgent


class XorAgent(NeatAgent):
    '''
        THIS IS THE CLASS THAT DEFINES AN AGENT FOR THE XOR PROBLEM

        It is mandatory to define your own agent so the library can work properly along with the extra functionalities.

        - FIRST: Inherit from the abstract class 'NeatAgent'
        - SECOND: Overwrite the __lt__ and __eq__ functions. The __lt__ may vary depending if your problem
            is maximization or minimization
        - THIRD: Overwrite the log_stats(agent) by expanding it or just return None
        - FOURTH: The constructor MUST receive a genome and a neural network. You have to save those variables somewhere
        - FIFTH: Create a function similar to 'think(inputs)' that allows the agent process information and then spit
            out a result
    '''
    def __init__(self, genome, neural_network):
        # Initializing super constructor
        super().__init__()

        # Initializing simulation variables
        self.loss = 0
        self.genome = genome
        self.brain = neural_network

    def __lt__(self, other):
        # This is a maximization problem, so __lt__ is structured this way
        return self.genome.fitness > other.genome.fitness

    def __eq__(self, other):
        return self.genome.fitness == other.genome.fitness

    @staticmethod
    def log_stats(agent):
        message = "Loss: {}".format(agent.loss)
        message += "\n"
        return message

    def think(self, inputs):
        # Forward pass of the neural network
        return self.brain.activate(inputs)
