# Models
from neat_exporter_package.tests.usage_example.simulation.xor_simulation import XorGateGame

# AI
import neat
from neat_exporter_package.neat_utility.models.neat_setup import NeatSetup

# utils
from os.path import dirname, join

'''
    THIS EXAMPLE IS TAKEN FROM THE XOR EXAMPLE, LOCATED IN THE EXAMPLES DIRECTORY OF THE 
    ORIGINAL NEAT-PYTHON LIBRARY:
    https://github.com/CodeReclaimers/neat-python/tree/master/examples/xor
    
    So lets keep this short and concise.
    Take this main.py file as the base of your project, since the other stuff will revolve around it.
'''

if __name__ == '__main__':
    '''
        FIRST: Set up the simulation.
        For more information, go to simulation/xor_simulation.py
    '''
    game = XorGateGame()

    '''
        SECOND: Parameters setup
        Lets start with the easiest ones:
        - max_generations: Number of maximum generations the algorithm will run
        - neat_checkpoint_breakpoint: Specifies how many generations the algorithm will run before creating a
            checkpoint
        - file_prefix: This will be used to name all the visualization SVG files of the fittest agent
            from each generation
        - simulation_file: It is a constant. Always pass down the variable '__file__'
    '''
    # INTUITIVE PARAMETERS
    max_generations = 50
    neat_checkpoint_breakpoint = 10
    file_prefix = "xor"
    simulation_file = __file__

    '''
        These parameters are not intuitive at first sight. Here is some explanation:
        - load_checkpoint_number: Look into the generated 'neat-checkpoints' directory for saved checkpoints.
            If you have some, you can select which one to start the algorithm.
            You are only required to pass down the CHECKPOINT NUMBER.
            For example: neat-checkpoint-10, then load_checkpoint_number = 10
            
            If you want to start from scratch, pass down 'None'.
            For example: load_checkpoint_number = None
            
        - config_file: This is the already parsed and instantiated configuration file that the NEAT algorithm needs
            in order to set up the customizable parameters.
            
            Pass down 'None' to use the default configuration file structure (the default structure can be found in the
            artificial_intelligence/config-feedforward.txt file)
            
            Or if you want to be more explicit about certain parameters, you can create your own configuration file and
            pass it down.
            
        - simulation: Basically, the famous eval_genomes function. This is the function that starts the simulation.
            It must receive a 'genomes' and 'config' parameters, respectively.
    '''
    # EASY TO PICK UP PARAMETERS

    load_checkpoint_number = None       # If you want to start from scratch
    # load_checkpoint_number = 10       # If you want to start from the saved checkpoint 'neat-checkpoint-10'

    # If use the default configuration file structure found in artificial_intelligence/config-feedforward.txt
    config_file = None

    # If you want to use your custom configuration file. Don't forget to provide the correct path!
    # config_file = neat.config.Config(
    #     neat.DefaultGenome, neat.DefaultReproduction,
    #     neat.DefaultSpeciesSet, neat.DefaultStagnation,
    #     join(dirname(__file__), 'artificial_intelligence', 'config-feedforward.txt')
    # )

    # Provide the equivalent of eval_genomes(genomes, config) function
    simulation = game.simulation        # Go to simulation/xor_simulation.py to check how to setup the simulation for the problem

    '''
        These parameters can be confusing, here is some help:
        - logging_function: This function will expand the information you want to register in the logs
            This function MUST have a parameter that receives an Object with the datatype of our defined Agent.
            
            Use 'None' if you don't desire to expand said functionality or pass down a function with the structure of
            log_agent(agent).
            
            To check more information on how to set up your own Agent Datatype,
            go to simulation/xor_agent.py as an example
    '''
    # CONFUSING PARAMETERS

    # Setting up NEAT Algorithm
    neatSetup = NeatSetup(
        max_generations=max_generations,
        neat_checkpoint_breakpoint=neat_checkpoint_breakpoint,
        file_prefix=file_prefix,
        simulation_file=simulation_file,

        load_checkpoint_number=load_checkpoint_number,
        config_file=config_file,
        simulation=simulation,

        logging_function=None
    )

    # Run
    neatSetup.run_simulation()
