"""All the functions to train the model sequentially, test it or play the game"""


import neat
import pickle 
from pathlib import Path
import sys
import fitness_function

#this is a bad way to do this, I should fix it later
PARENTPATH = str(Path(__file__).parent.parent)
sys.path.append(PARENTPATH + "/2048-python")
####################################################

import logic, puzzle, constants


def eval_genomes(genomes, config):
    """Evaluate the genomes by making each of them play 2 games per generation and setting the fitness to the average of the games"""
    game = puzzle.GameGrid()
    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness += fitness_function.play_game(game, net)
        game.reset()
        genome.fitness += fitness_function.play_game(game, net)
        game.reset()
        genome.fitness /= 2 #taking the average of the two games





def train_ai(config_file, checkpoint = '0', generations = 100, folder = 'checkpoints',
        winner_file = 'best.pickle', eval_function = eval_genomes, generation_interval=10):

    """Train the population from checkpoint during the given number of generations"""
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    
    pop = neat.Population(config) if checkpoint == '0' else neat.Checkpointer.restore_checkpoint(folder + '/neat-checkpoint-' + checkpoint) 
    #Add a stdout reporter to show progress in the terminal
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.Checkpointer(generation_interval = generation_interval, filename_prefix= folder+'/neat-checkpoint-'))


    winner = pop.run(eval_function, generations)

    #saving the best AI
    with open(winner_file, "wb") as f:
        pickle.dump(winner, f)







