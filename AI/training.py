"""All the functions to train the model sequentially, test it or play the game"""

from time import sleep, time
import random
import tkinter
import neat
import pickle 
from pathlib import Path
import sys
import os

#this is a bad way to do this, I should fix it later
PARENTPATH = str(Path(__file__).parent.parent)
sys.path.append(PARENTPATH + "/2048-python")
####################################################

import logic, puzzle, constants

game = puzzle.GameGrid()

def eval_genomes(genomes, config):
    """Evaluate the genomes by making each of them play 2 games per generation and setting the fitness to the average of the games"""

    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness += play_game(game, net)
        game.reset()
        genome.fitness += play_game(game, net)
        game.reset()
        genome.fitness /= 2 #taking the average of the two games


def play_game(game: puzzle.GameGrid, net=None, max_move = 1000, time_delay = False):
    """Play one game of 2048 with the neural net as the player"""
    running = True
    fitness = 0
    score_history = []
    num_move = 0
    while running:
        
        try:
            status, old_score, score = game.one_loop()
            if time_delay: sleep(0.1)
            if status == 'lose': 
                fitness -= 100
                running = False

            if status == 'win':
                fitness += 2048
                running = False

            if num_move > max_move: running = False

            fitness += (score - old_score) if score != old_score else -100
            score_history.append(score - old_score)
            if net is not None:
                num_move += 1
                flatten_grid = [element for row in game.matrix for element in row]
                
                output = net.activate(flatten_grid)
                
                move = output.index(max(output))

                
             

                if len(score_history) > 10:
                    score_history.pop(0)

                #do a random move if the AI is stuck in making a move that doesn't modify the board
                if all(x == 0 for x in score_history) and len(score_history) == 10:
                    move = random.randint(0,3)

                if move == 0:
                    game.process_input(constants.KEY_UP)
                elif move ==1:
                    game.process_input(constants.KEY_DOWN)
                elif move ==2:
                    game.process_input(constants.KEY_LEFT)
                elif move == 3:
                    game.process_input(constants.KEY_RIGHT)
                else:
                    print("something's wrong") #shouldnt happen
            
        except tkinter.TclError:
            running = False
    
    return fitness


def test_ai(config_file):
    """Test the best AI trained so far"""
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)

    with open("winner2.pickle", "rb") as f:
        best = pickle.load(f)


    net = neat.nn.FeedForwardNetwork.create(best, config)
    game.reset()
    play_game(game, net, max_move=sys.maxsize, time_delay=True)
    game.reset()



def train_ai(config_file, checkpoint = '0', generations = 100):
    """Train the population from checkpoint during the given number of generations"""
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    
    pop = neat.Population(config) if checkpoint == '0' else neat.Checkpointer.restore_checkpoint('checkpoints/neat-checkpoint-' + checkpoint) 
    #Add a stdout reporter to show progress in the terminal
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.Checkpointer(generation_interval = 25, filename_prefix='checkpoints/neat-checkpoint-'))


    winner = pop.run(eval_genomes, generations)

    #saving the best AI
    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)

    

if __name__ == "__main__":
    
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    #train_ai(config_path, '114')
    test_ai(config_path)       







