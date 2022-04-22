"""Program to execute to manage the model: train or test it"""


from importlib import invalidate_caches
import os
import AI





def test(config):
    print('The following models are available to test: ')
    for file in os.listdir('best'):
        print(file)
    file = input('Which one do you want? ')
    if file in os.listdir('best'):
        AI.test_ai(config, f'best/{file}')
    else:
        print('invalid file')

def train(config):
    pass


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
        if task == 'train': 
            train(config_path)
            break
        elif task == 'test':
            test(config_path)
            break
        else:
            print('invalid command, please try again')




if __name__ == '__main__':
    main()