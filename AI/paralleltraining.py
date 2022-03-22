"""Functions to train the AI in parallel by using multiprocessing"""

import pickle
import tkinter
import multiprocessing
import os
import neat
import training

def eval_genome(genome, config):
    """Evaluate one genome by making it play 3 games and setting its fitness to the average of the games"""
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    game = training.game

    fitness = 0
    for i in range(3):
        game.reset()
        fitness += training.play_game(game, net)

    return fitness / 3


def run(config_file, checkpoint='0', generations=40):
    """Train the AI from checkpoint to the number of generations given"""
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)

    pop = neat.Population(config) if checkpoint == '0' else neat.Checkpointer.restore_checkpoint('checkpoints2/neat-checkpoint-' + checkpoint)


    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.Checkpointer(generation_interval = 1, filename_prefix='checkpoints2/neat-checkpoint-'))

    #will train 8 genomes in parallel
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)

    winner = pop.run(pe.evaluate, generations)

    #saving the best AI

    with open("winner2.pickle", "wb") as f:
        pickle.dump(winner, f)



if __name__ == '__main__':

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')

    run(config_path, checkpoint='90')
    
