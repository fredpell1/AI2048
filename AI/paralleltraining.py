import pickle
import tkinter
import multiprocessing
import os
import neat
import training

def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    game = training.game

    fitness = training.play_game(game, net)
    game.reset()

    return fitness


def run(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)

    pop = neat.Population(config)


    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.Checkpointer(generation_interval = 25, filename_prefix='checkpoints/neat-checkpoint-'))

    #will train 4 genomes in parallel
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count() // 2, eval_genome)

    winner = pop.run(pe.evaluate, 50)

    #saving the best AI

    with open("winner.pickle", "wb") as f:
        pickle.dump(winner, f)



if __name__ == '__main__':

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')

    run(config_path)
    
