# AI
import neat

# utils
from neat_exporter_package.neat_utility.visualize import draw_net
import shutil
from os.path import join, dirname, exists
from os import mkdir, listdir
import logging
from datetime import datetime


class NeatSetup:
    NEAT_CHECKPOINT_FILE_PREFIX = "neat-checkpoint"

    def __init__(
            self,
            simulation, max_generations, neat_checkpoint_breakpoint,
            file_prefix, simulation_file,
            logging_function=None,
            load_checkpoint_number=None, config_file=None
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

        # Setting directories
        self.root_directory = dirname(simulation_file)

        self.config_path = join(self.root_directory, 'artificial_intelligence', 'config-feedforward.txt')

        self.logs_path = join(self.root_directory, 'neat_logs')
        self.checkpoint_path = join(self.root_directory, 'neat_checkpoints')

        self.svg_path = join(self.root_directory, 'svg_growth')

        # Parsing configuration file
        if config_file is None:
            self.config_file = neat.config.Config(
                neat.DefaultGenome, neat.DefaultReproduction,
                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                self.config_path
            )
        else:
            self.config_file = config_file

        # Setting up Logging function
        self.logging_function = logging_function

        # Create checkpoint directory if it doesn't exist
        if not exists(self.checkpoint_path):
            mkdir(self.checkpoint_path)

        # Create SVG directory if it doesn't exist
        if not exists(self.svg_path):
            mkdir(self.svg_path)

        # Create logs directory if it doesn't exist
        if not exists(self.logs_path):
            mkdir(self.logs_path)

        # Setting up logger
        today_date = datetime.today().strftime('%Y_%m_%d-%H_%M')
        logging.basicConfig(
            filename=join(self.logs_path, 'training_log_neat_{}.log'.format(today_date)),
            filemode='w',
            level=logging.INFO,
            format='\n%(asctime)s-%(levelname)s> %(message)s',
            datefmt='%d-%b-%y %H:%M:%S'
        )

    def move_checkpoints(self):
        # Moving checkpoint files
        for file in listdir(self.root_directory):
            if file.startswith(NeatSetup.NEAT_CHECKPOINT_FILE_PREFIX):
                shutil.move(
                    src=join(self.root_directory, file),
                    dst=join(self.checkpoint_path, file)
                )

    def move_svg_visualization(self):
        # Moving SVGs
        for file in listdir(self.root_directory):
            if file.startswith(self.file_prefix):
                shutil.move(
                    src=join(self.root_directory, file),
                    dst=join(self.svg_path, file)
                )

    def log_stats(self, generation, winner_genome):
        # Creating message to log
        message = "\n---END OF GENERATION {}---\n".format(generation)
        message += "Fittest Score: {}\n".format(winner_genome.genome.fitness)
        message += "" if self.logging_function is None else self.logging_function()

        # Logging
        logging.info(
            msg=message
        )

    def _neat_simulation(self, genomes, config):
        # Play simulation
        generation, fittest_genome = self.simulation(genomes=genomes, config=config)

        # Logging results
        self.log_stats(generation=generation, winner_genome=fittest_genome)

        # Visualize best genome
        draw_net(
            config=self.config_file,
            genome=fittest_genome.genome,
            view=False,
            filename=self.file_prefix + "_{}".format(generation),
            show_disabled=True,
            fmt='svg'
        )
        self.move_svg_visualization()

        # Move Checkpoints
        self.move_checkpoints()

    def run_simulation(self):
        # Create population
        if self.load_checkpoint_number is None:
            # Creating population from config file
            generation = neat.Population(self.config_file)
        else:
            # Obtaining the path of the checkpoint
            checkpoint_path = join(
                self.checkpoint_path,
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
            view=False,
            filename=self.file_prefix + "_fittest",
            show_disabled=True,
            fmt='svg'
        )
        self.move_svg_visualization()
