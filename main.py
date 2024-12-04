import os
import h5py
from multiprocessing import Pool, cpu_count
import environment

def main():
    # Training SETUP
    train_or_test = 'train'  #'train' or 'test'
    has_continuous_action_space = True 
    continuous_training = False
    
    if train_or_test == 'train':
        print("Starting training...")
        env = environment.Env(root, train_or_test, has_continuous_action_space, continuous_training)
        environment.train(env)

    elif train_or_test == 'test':
        load_pt_file_name = ''
        env = environment.Env(root, train_or_test, has_continuous_action_space, continuous_training, load_pt_file_name = load_pt_file_name)
        environment.test(env)

if __name__ == '__main__':
    main()
