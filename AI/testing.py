import os
import neat
import sys
from pathlib import Path
#this is a bad way to do this, I should fix it later
PARENTPATH = str(Path(__file__).parent.parent)
sys.path.append(PARENTPATH + "/2048-python")
####################################################
import logic, puzzle, constants
import pickle
from . import fitness_function


def test_ai(config_file, best_file='best/winner.pickle'):
    """Test the best AI trained so far"""
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)

    with open(best_file, "rb") as f:
        best = pickle.load(f)

    game = puzzle.GameGrid()
    net = neat.nn.FeedForwardNetwork.create(best, config)
    fitness_function.play_game(game, net, max_move=sys.maxsize, time_delay=True)
    game.reset()

