"""Functions to train the AI in parallel by using multiprocessing"""


import pickle
import tkinter
import multiprocessing
import os
import neat
import sequentialtraining

def eval_genome(genome, config):
    """Evaluate one genome by making it play 3 games and setting its fitness to the average of the games"""
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    game = sequentialtraining.puzzle.GameGrid()

    fitness = 0
    for i in range(3):
        game.reset()
        fitness += sequentialtraining.play_game(game, net)

    return fitness / 3


def train(config_file, checkpoint='0', generations=40, processors = multiprocessing.cpu_count(), folder = 'checkpoints',
        winner_file = 'best.pickle', eval_function = eval_genome, generation_interval=10):
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

