# AI2048

UI for the game was taken [there](https://github.com/yangshun/2048-python) and I refactored parts of the code to adapt it to an AI playing the game. 
However, since the project was read-only, when I pushed my changes it broke the link of the git submodule. 

## Reward functions

There are several possible reward functions you can use: 
- A "greedy" one that only rewards the immediate change of score
- One that looks for patterns in blocks
- A combination of both

The most successful has been a combination of both the greedy and block pattern (called increasing_row_col_greedy_fitness) with a normalization of the input. 

## Training

### Parallel training

If you wish to take advantage of your multicore computer to increase the speed of training, use the functions in this [file](https://github.com/fredpell1/AI2048/blob/main/AI/paralleltraining.py). It will use all the cores available on your computer and train several genomes in parallel. However, it will use a lot of resources and your computer may get slow. 


### Sequential training

If you wish to train the model sequentially, use the functions in this [file](https://github.com/fredpell1/AI2048/blob/main/AI/training.py). It will take a lot more time but you will be able to use your computer at the same time without much impact on performance. 

When training sequentially, each genome plays two games per generation and is given a fitness value corresponding to its average fitness score during the two games.

### Learning

Using the function in either parallel training or sequential training, you can make the model learn by running this [file](https://github.com/fredpell1/AI2048/blob/main/AI/learning.py)

## Performance
 
After training for only 10 generations, the model achieves average performance and can reach a tile of 256. 

## Potential next steps

- Add some command line input
- refactor the sequential training and parallel training files to remove duplicate codes in both

## Dependencies

Look at this [file](https://github.com/fredpell1/AI2048/blob/main/requirements.txt)


