# AI2048

I am currently refactoring the code to make it more modular, slightly more object oriented, to better parameterize different reward functions that will be implemented. 

UI for the game was taken [there](https://github.com/yangshun/2048-python) and I refactored parts of the code to adapt it to an AI playing the game. 
However, since the project was read-only, when I pushed my changes it broke the link of the git submodule. 

## Reward function

The reward function used to evaluate the genomes makes each genome play 2 (or 3, see Training) games and sets the fitness to the average performance of the games. It rewards each move by the points just made and heavily penalizes moves that don't change the state of the board. If the model does a move that does not change the state of the game 10 times in a row, a random move is made. 

## Training

### Parallel training

If you wish to take advantage of your multicore computer to increase the speed of training, use the functions in this [file](https://github.com/fredpell1/AI2048/blob/main/AI/paralleltraining.py). It will use all the cores available on your computer and train several genomes in parallel. However, it will use a lot of resources and your computer may get slow. 

When training in parallel, each genome plays three games per generation and is given a fitness value corresponding to its average fitness score during the three games. 

### Sequential training

If you wish to train the model sequentially, use the functions in this [file](https://github.com/fredpell1/AI2048/blob/main/AI/training.py). It will take a lot more time but you will be able to use your computer at the same time without much impact on performance. 

When training sequentially, each genome plays two games per generation and is given a fitness value corresponding to its average fitness score during the two games.

## Performance
 
After training for 91 generations, the model is playing at a level greatly inferior to humans. It may be due to a lack of training, bad configuration parameters of the NEAT algorithm, a too general reward function, the absence of data preprocessing. 

## Comments

Although the results are disappointing, this project was a great learning experience where I learned more about the NEAT algorithm, parallel programming in Python and applying AI concepts to video games. 

## Potential next steps

- Normalize the inputs to avoid giving unnecessarily importance to large values.
- Test different reward functions, from general to really specific.
- Deal differently with when the model makes a move that does not update the board.


## Dependencies

Look at this [file](https://github.com/fredpell1/AI2048/blob/main/requirements.txt)


