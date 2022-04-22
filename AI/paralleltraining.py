"""Functions to train the AI in parallel by using multiprocessing"""


import pickle
import multiprocessing
import neat
from pathlib import Path
import sys
from . import fitness_function
#this is a bad way to do this, I should fix it later
PARENTPATH = str(Path(__file__).parent.parent)
sys.path.append(PARENTPATH + "/2048-python")
####################################################

import logic, puzzle, constants


def eval_genome_increasing_greedy(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    game = puzzle.GameGrid()
    game.reset()
    return fitness_function.play_game(game, net, reward_function=fitness_function.increasing_row_col_greedy_fitness)
    




def eval_genome_greedy(genome, config):
    """Evaluate one genome by making it play 3 games and setting its fitness to the average of the games, using the
        greedy reward function
    """
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    game = puzzle.GameGrid()

    fitness = 0
    for i in range(3):
        game.reset()
        fitness += fitness_function.play_game(game, net)

    return fitness / 3


def train(config_file, checkpoint='0', generations=10, processors = multiprocessing.cpu_count(), folder = 'checkpoints',
        winner_file = 'best.pickle', eval_function = eval_genome_greedy, generation_interval=1):
    """Train the AI from checkpoint to the number of generations given"""

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)

    pop = neat.Population(config) if checkpoint == '0' else neat.Checkpointer.restore_checkpoint(folder+'/neat-checkpoint-' + checkpoint)


    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.Checkpointer(generation_interval = generation_interval, filename_prefix=folder+'/neat-checkpoint-'))

    #will train genomes in parallel
    pe = neat.ParallelEvaluator(processors, eval_function)

    winner = pop.run(pe.evaluate, generations)

    #saving the best AI

    with open(winner_file, "wb") as f:
        pickle.dump(winner, f)

