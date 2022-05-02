"""Program to execute to manage the model: train or test it"""



import os
import AI

def __get_eval_function():

    methods = [x for x in dir(AI.paralleltraining) if x.startswith('eval')]
    print('The following evaluation functions are available to train the model: ')
    for method in methods:
        print(method, end = ', ')
    while(True):
        method = input('Which one do you want to use? ')
        if hasattr(AI.paralleltraining, method):
            break
        else:
            print('Invalid method name. Please try again. ')
    return getattr(AI.paralleltraining, method)

def __create_checkpoints_directory():
    while(True):
        directory = input('Where do you want to save the checkpoints of this new model? ')
        try:
            os.mkdir(directory)
            break
        except FileExistsError:
            print("This directory already exists. Please try again.")
    return directory
        

def __find_folder():
    while(True):
        folder = input('In what folder are the checkpoints for the model? ')
        print(folder)
        if folder not in os.listdir():
            print('invalid folder, please try again')
        else:
            break

    return folder


def __find_checkpoint(folder):
    
    print('We found the following checkpoints: ')
    for checkpoint in sorted(os.listdir(folder), key = lambda x: int(x.split('-')[-1])):
        print(checkpoint, end = ', ')

    checkpoint = input('\nFrom which do you want to start, enter the ' \
        ' number only: ')

    return checkpoint

def test(config):
    print('The following models are available to test: ')
    for file in os.listdir('best'):
        print(file)
    file = input('\nWhich one do you want? ')
    while(True):
        if file in os.listdir('best'):
            AI.test_ai(config, f'best/{file}')
            break
        else:
            print('invalid file')

def train(config):
    new_model = input('Do you want to train a new model or an existing one (new/existing) ? ')
    if new_model.lower() == 'new':
        folder = __create_checkpoints_directory()
        num_generations = int(input('For how many generations do you want to train the model? '))
        winner_file = input('In which file do you want to save the best generation? ')
        eval_function = __get_eval_function()
        AI.paralleltraining.train(config_file=config, generations=num_generations, folder=folder, 
        winner_file=f'best/{winner_file}', eval_function=eval_function)
    
    elif new_model.lower() == 'existing':
        folder = __find_folder()
        checkpoint =__find_checkpoint(folder)
        num_generations = int(input('For how many generations do you want to train the model? '))
        winner_file = input('In which file do you want to save the best generation? ')
        AI.paralleltraining.train(config_file = config, checkpoint = checkpoint, folder = folder, generations = num_generations,
         winner_file=f'best/{winner_file}')
        
        
    else:
        print('invalid answer')

    

def main():
    #finding the config file
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'AI/config.txt')

    #greetings

    print("""
  _   _            _      __             ___   ___  _  _   ___  
 | \ | |          | |    / _|           |__ \ / _ \| || | / _ \ 
 |  \| | ___  __ _| |_  | |_ ___  _ __     ) | | | | || || (_) |
 | . ` |/ _ \/ _` | __| |  _/ _ \| '__|   / /| | | |__   _> _ < 
 | |\  |  __/ (_| | |_  | || (_) | |     / /_| |_| |  | || (_) |
 |_| \_|\___|\__,_|\__| |_| \___/|_|    |____|\___/   |_| \___/
                                                                                
                                                                            """)


    #selecting the task to do 
    while(True):
        task =input('What do you want to do (test or train)? ')
        if task.lower() == 'train': 
            train(config_path)
            break
        elif task.lower() == 'test':
            test(config_path)
            break
        else:
            print('invalid command, please try again')




if __name__ == '__main__':
    main()