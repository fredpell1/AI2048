import neat
import pickle 
from pathlib import Path
import sys

#this is a bad way to do this, I should fix it later
PARENTPATH = str(Path(__file__).parent.parent)
sys.path.append(PARENTPATH + "/2048-python")
####################################################

import logic, puzzle, constants








