# AI
import neat

# Utils
from .visualize import draw_net
from .genome_to_json import export_genome_to_json
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
            inputs_name, outputs_name,
            is_feedforward_network=False,
            logging_function=None,
            load_checkpoint_number=None, config_file=None
    ):
        # Sanity Checking: Making sure the following parameters are functions
        assert callable(simulation)

        # Sanity checking: Making sure the following parameters are integers and have valid values
        assert isinstance(max_generations, int) and max_generations > 1
        assert isinstance(neat_checkpoint_breakpoint, int) and neat_checkpoint_breakpoint >= 0

        # Sanity checking: Making sure the following parameters are strings and are nor empty
        assert isinstance(file_prefix, str) and len(file_prefix) > 0
        assert isinstance(simulation_file, str) and len(simulation_file) > 0

        # Sanity Checking: Making sure the following parameters are tuples and are not empty
        assert (isinstance(inputs_name, list) or isinstance(inputs_name, tuple)) and inputs_name
        assert (isinstance(outputs_name, list) or isinstance(outputs_name, tuple)) and outputs_name

        # Sanity checking: Making sure the following parameters are booleans
        assert isinstance(is_feedforward_network, bool)

        # Sanity checking: Making sure the following parameters, if not None, are the expected type
        assert True if load_checkpoint_number is None else\
            (isinstance(load_checkpoint_number, int) and load_checkpoint_number >= 0)
        assert True if config_file is None else isinstance(config_file, neat.config.Config)

        # Initializing parameters for simulations
        self.simulation = simulation

        self.max_generations = max_generations
        self.neat_checkpoint_breakpoint = neat_checkpoint_breakpoint
        self.load_checkpoint_number = load_checkpoint_number

        self.file_prefix = file_prefix

        self.is_feedforward_network = is_feedforward_network

        self.inputs_name = inputs_name
        self.outputs_name = outputs_name

        # Setting directories
        self.root_directory = dirname(simulation_file)

        self.artificial_intelligence_path = join(self.root_directory, 'artificial_intelligence')

        self.logs_path = join(self.root_directory, 'neat_logs')
        self.checkpoint_path = join(self.root_directory, 'neat_checkpoints')

        self.svg_path = join(self.root_directory, 'svg_growth')

        # Parsing configuration file
        if config_file is None:
            self.config_file = neat.config.Config(
                neat.DefaultGenome, neat.DefaultReproduction,
                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                join(self.artificial_intelligence_path, 'config-feedforward.txt')
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
        message = "---END OF GENERATION {}---\n".format(generation)
        message += "Fittest Score: {}\n".format(winner_genome.genome.fitness)
        message += "" if self.logging_function is None else self.logging_function(agent=winner_genome)

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

        # Move remaining checkpoints
        self.move_checkpoints()

        # Export model
        if self.is_feedforward_network:
            export_genome_to_json(
                filename=join(self.artificial_intelligence_path, 'network_dump.json'),
                config=self.config_file,
                genome=winner,
                inputs_names=self.inputs_name,
                outputs_names=self.outputs_name
            )
