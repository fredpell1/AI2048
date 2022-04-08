from time import sleep
import tkinter
import sys
from pathlib import Path
#this is a bad way to do this, I should fix it later
PARENTPATH = str(Path(__file__).parent.parent)
sys.path.append(PARENTPATH + "/2048-python")
####################################################
import logic, puzzle, constants
import random


def normalize_max(flatten_grid):
    max_value = max(flatten_grid)
    return [i / max_value for i in flatten_grid]

def greedy_fitness(score, old_score, matrix):
    return score - old_score if score != old_score else -1



def play_game(game: puzzle.GameGrid, net=None, max_move = 1000, time_delay = False, reward_function = greedy_fitness, 
    normalize = True):

    """Play one game of 2048 with the neural net as the player"""
    running = True
    fitness = 0
    score_history = []
    num_move = 0
    while running:
        
        try:
            game.one_loop()
            if net is not None:
                
                
                flatten_grid = normalize_max([element for row in game.matrix for element in row]) \
                    if normalize else \
                    [element for row in game.matrix for element in row]
                print(flatten_grid)
                output = net.activate(flatten_grid)
                
                move = output.index(max(output))

                
                if len(score_history) > 10:
                    score_history.pop(0)
                
                #do a random move if the AI is stuck in making a move that doesn't modify the board
                if all(x == 0 for x in score_history) and len(score_history) == 10:
                    
                    move = random.randint(0,3)
                
                if move == 0:
                    status, old_score, score = game.process_input(constants.KEY_UP)
                    
                elif move ==1:
                    status, old_score, score = game.process_input(constants.KEY_DOWN)
                    
                elif move ==2:
                    status, old_score, score = game.process_input(constants.KEY_LEFT)
                    
                elif move == 3:
                    status, old_score, score = game.process_input(constants.KEY_RIGHT)
        
                else:
                    print("something's wrong") #shouldnt happen
                
                num_move += 1
                if len(score_history) > 10:
                    score_history.pop(0)      
                        
                if time_delay: sleep(0.1)

                if status == 'lose': 
                    fitness -= 100
                    running = False

                if status == 'win':
                     fitness += 2048
                     running = False

                if num_move > max_move: running = False

                fitness += reward_function(score, old_score, game.matrix)
                score_history.append(score - old_score)
                
        except tkinter.TclError:
            running = False
    
    return fitness

