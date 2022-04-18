import paralleltraining
import os


def main():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    paralleltraining.train(config_path, folder='checkpoints', winner_file='best_increasing_greedy.pickle',
    eval_function=paralleltraining.eval_genome_increasing_greedy, checkpoint='9')


if __name__ == '__main__':
    main()