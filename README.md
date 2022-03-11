# AI2048

This is an ongoing project. 

I took the UI for the game [there](https://github.com/yangshun/2048-python) and I refactored parts of the code to adapt it to an AI playing the game. 
However, since the project was read-only, when I pushed my changes it broke the link of the git submodule. 

## Performance
 
Currently, the agent is pretty terrible and reaches a 128 or 256 cell before losing. The evaluation function is pretty general and does not explicitly give rules of the game through
its rewards. Each genome plays two games per generation. The agent usually get stuck in a local minima and compulsively repeats the same move. To avoid this, 
I select a random move when the agent has been doing the same thing for 10 consecutive moves. 

## Dependencies

Look at this [file](https://github.com/fredpell1/AI2048/blob/main/requirements.txt)


