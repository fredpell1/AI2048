import tkinter
import neat
import pickle 
from pathlib import Path
import sys

#this is a bad way to do this, I should fix it later
PARENTPATH = str(Path(__file__).parent.parent)
sys.path.append(PARENTPATH + "/2048-python")
####################################################

import logic, puzzle, constants




if __name__ == "__main__":
    game = puzzle.GameGrid()
    
    running = True

    while running:
        try:
            game.one_loop()
        except tkinter.TclError:
            running = False







