import os
import h5py
from multiprocessing import Pool, cpu_count

def main():
    # Training SETUP
    train_or_test = 'train'  #'train' or 'test'
    has_continuous_action_space = True

    if train_or_test == 'train':
        continue_training = False
        load_pt_file_name = ''
        import environment
        env = environment.Env(root, train_or_test_or_write, has_continuous_action_space, continue_training = False, load_pt_file_name = '')
        environment.train(env)

    elif train_or_test == 'test':
        load_pt_file_name = ''
        # load simulation

if __name__ == '__main__':
    main()
