# AI
import neat

# utils
from neat_utility.visualize import draw_net
import shutil
from os.path import join, dirname, exists
from os import mkdir, listdir
import logging
from datetime import datetime


class NeatSetup:
    CONFIG_PATH = join(dirname(dirname(__file__)), 'artificial_intelligence', 'config-feedforward.txt')
    LOG_PATH = join(dirname(dirname(__file__)), 'neat_logs')
    CHECKPOINT_PATH = join(dirname(dirname(__file__)), 'neat_checkpoints')

    NEAT_CHECKPOINT_FILE_PREFIX = "neat-checkpoint"

    MAIN_PY_DIRECTORY = dirname(dirname(dirname(__file__)))
    SVG_PATH = join(dirname(dirname(dirname(__file__))), 'growth')

    def __init__(
            self,
            simulation, max_generations, neat_checkpoint_breakpoint,
            file_prefix,
            load_checkpoint_number=None, config_file=None, logging_function=None
    ):
        # Sanity Checking: Make sure main parameters are not None
        assert None not in [simulation, max_generations, neat_checkpoint_breakpoint, file_prefix]

        # Sanity checking: Making sure the following parameters are integers
        assert isinstance(max_generations, int) and isinstance(neat_checkpoint_breakpoint, int)

        # Sanity checking: Making sure the following parameters are strings
        assert isinstance(file_prefix, str)

        # Sanity checking: Making sure the following parameters, if not None, are the expected type
        assert True if load_checkpoint_number is None else isinstance(load_checkpoint_number, int)
        assert True if config_file is None else isinstance(load_checkpoint_number, neat.config.Config)

        # Initializing parameters for simulations
        self.simulation = simulation

        self.max_generations = max_generations
        self.neat_checkpoint_breakpoint = neat_checkpoint_breakpoint
        self.load_checkpoint_number = load_checkpoint_number

        self.file_prefix = file_prefix

        # Parsing configuration file
        if config_file is None:
            self.config_file = neat.config.Config(
                neat.DefaultGenome, neat.DefaultReproduction,
                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                NeatSetup.CONFIG_PATH
            )
        else:
            self.config_file = config_file

        # Setting up Logging function
        self.logging_function = logging_function

        # Create checkpoint directory if it doesn't exist
        if not exists(NeatSetup.CHECKPOINT_PATH):
            mkdir(NeatSetup.CHECKPOINT_PATH)

        # Create SVG directory if it doesn't exist
        if not exists(NeatSetup.SVG_PATH):
            mkdir(NeatSetup.SVG_PATH)

        # Create logs directory if it doesn't exist
        if not exists(NeatSetup.LOG_PATH):
            mkdir(NeatSetup.LOG_PATH)

        # Setting up logger
        today_date = datetime.today().strftime('%Y_%m_%d-%H_%M')
        logging.basicConfig(
            filename=join(NeatSetup.LOG_PATH, 'training_log_neat_{}.log'.format(today_date)),
            filemode='w',
            level=logging.INFO,
            format='\n%(asctime)s-%(levelname)s> %(message)s',
            datefmt='%d-%b-%y %H:%M:%S'
        )

    @staticmethod
    def move_checkpoints():
        # Moving checkpoint files
        for file in listdir(NeatSetup.MAIN_PY_DIRECTORY):
            if file.startswith(NeatSetup.NEAT_CHECKPOINT_FILE_PREFIX):
                shutil.move(
                    src=join(NeatSetup.MAIN_PY_DIRECTORY, file),
                    dst=join(NeatSetup.CHECKPOINT_PATH, file)
                )

    @staticmethod
    def move_svg_visualization(file_prefix):
        # Moving SVGs
        for file in listdir(NeatSetup.MAIN_PY_DIRECTORY):
            if file.startswith(file_prefix):
                shutil.move(
                    src=join(NeatSetup.MAIN_PY_DIRECTORY, file),
                    dst=join(NeatSetup.SVG_PATH, file)
                )

    def log_stats(self, winner_genome):
        # Creating message to log
        message = "\n---END OF GENERATION {}---\n".format(winner_genome)
        message += "Fittest Score: {}\n".format(winner_genome.genome.fitness)
        message += "" if self.logging_function is None else self.logging_function()

        # Logging
        logging.info(
            msg=message
        )

    def _neat_simulation(self, genomes, config):
        # Play simulation
        fittest_genome = self.simulation(genomes=genomes, config=config)

        # Logging results
        self.log_stats(winner_genome=fittest_genome)

        # Move SVG files
        NeatSetup.move_svg_visualization(file_prefix=self.file_prefix)

        # Move Checkpoints
        NeatSetup.move_checkpoints()

    def start_simulation(self):
        # Create population
        if self.load_checkpoint_number is None:
            # Creating population from config file
            generation = neat.Population(self.config_file)
        else:
            # Obtaining the path of the checkpoint
            checkpoint_path = join(
                NeatSetup.CHECKPOINT_PATH,
                '{}-{}'.format(NeatSetup.NEAT_CHECKPOINT_FILE_PREFIX, self.load_checkpoint_number)
            )

            # Creating population
            generation = neat.Checkpointer.restore_checkpoint(checkpoint_path)

        # Adding statistics and logging capabilities
        generation.add_reporter(neat.StdOutReporter(True))
        generation.add_reporter(neat.StatisticsReporter())
        generation.add_reporter(neat.Checkpointer(self.neat_checkpoint_breakpoint))

        # Run for up to max_generations generations.
        winner = generation.run(self._neat_simulation, self.max_generations)

        # Show stats of the best Genome
        print('\nBest genome:\n{!s}'.format(winner))

        # Visualize best genome
        draw_net(
            config=self.config_file,
            genome=winner,
            view=True,
            filename=self.file_prefix + "_fittest",
            show_disabled=True,
            fmt='svg'
        )