# Models
from simulation.xor_simulation import XorGateGame

# AI
from neat_exporter_package.neat_utility import *
from neat_exporter_package.neat_utility.models.neat_setup import NeatSetup


if __name__ == '__main__':
    # Setting up Game
    game = XorGateGame()

    # Setting up NEAT Algorithm
    neatSetup = NeatSetup(
        simulation=game.simulation,
        max_generations=50,
        neat_checkpoint_breakpoint=11,
        file_prefix="xor",
        simulation_file=__file__,
        load_checkpoint_number=None,
        config_file=None,
        logging_function=None
    )

    # Run
    neatSetup.run_simulation()
