from time import sleep
import tkinter
import sys
import numpy as np
from pathlib import Path
#this is a bad way to do this, I should fix it later
PARENTPATH = str(Path(__file__).parent.parent)
sys.path.append(PARENTPATH + "/2048-python")
####################################################
import logic, puzzle, constants
import random


def __normalize_max(flatten_grid):
    max_value = max(flatten_grid)
    return [i / max_value for i in flatten_grid]

def greedy_fitness(score, old_score, matrix):
    return score - old_score if score != old_score else -1

def increasing_row_fitness(score, old_score, matrix):
    fitness = 0
    best = lambda row : row[0] == row[1] and row[2] == row[3] and 2 * row[0] == row[3]
    second_best = lambda row: row[0] == row[1] and row[2] == row[3] and row[0] == row[3]
    last = lambda row: row[0] == row[1] or row[1] == row[2] or row[2] == row[3]
    for row in matrix: 
        if best(row) or best(row[::-1]):
            fitness += 10
        elif second_best(row) or second_best(row[::-1]):
            fitness += 5
        elif last(row) or last(row[::-1]):
            fitness += 2.5

    return fitness


def increasing_col_fitness(score, old_score, matrix):
    transposed = np.array(matrix).T
    return increasing_row_fitness(score, old_score, transposed)
        

def increasing_row_col_greedy_fitness(score, old_score, matrix):
    fitness = 0
    fitness += greedy_fitness(score, old_score, matrix)
    fitness += increasing_row_fitness(score, old_score, matrix)
    fitness += increasing_col_fitness(score, old_score, matrix)
    return fitness



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
                
                
                flatten_grid = __normalize_max([element for row in game.matrix for element in row]) \
                    if normalize else \
                    [element for row in game.matrix for element in row]
            
                output = net.activate(flatten_grid)
                
                move = output.index(max(output))
                
                
                #do a random move if the AI is stuck in making a move that doesn't modify the board
                if all(x == 0 for x in score_history) and len(score_history) == 10:
                    fitness -= 1
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
                if len(score_history) >= 10:
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
    
    game.destroy()
    return fitness

