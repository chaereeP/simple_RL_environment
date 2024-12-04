import Simulation #import simulation as Simulation
import os
import torch
import numpy as np
import math
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import PPO
env_name = '' # set environment name here


class Set_env():
    def __init__(self,node,train_or_test,has_continuous_action_space, continuous_training , load_pt_file_name ):
        # torch.set_num_threads(6)
        if train_or_test !='test':
            from scenecreater import createScene
            self.root = createScene(node,train_or_test,has_continuous_action_space)
            Simulation.init(self.root)   # simulation initiaion
            self.node = self.root.solverNode.reducedModel.model

        self.t=1
        self.ep_reward = 0
        self.ep = 1
        self.test_running_reward = 0
        self.iterator = 0
        self.max_ep_len=2024
        self.K_epochs = 80               # update policy for K epochs
        self.eps_clip = 0.2              # clip parameter for PPO
        self.gamma = 0.99                # discount factor
        self.lr_actor = 0.0003           # learning rate for actor
        self.lr_critic = 0.001           # learning rate for critic
        global multi_animate_step
        self.multi_animate_step=multi_animate_step
        #####################################################
        # state space dimension
        self.state_dim =10
        self.value_state_dim=10
        # action space dimension
        self.action_dim = 5
        self.train_or_test = train_or_test
        self.has_continuous_action_space=has_continuous_action_space
        self.iterator = 0

    def _reward(self):
        # Calculate reward here
        return reward

    def get_velocity(self):
        # load velocity
        return vel

    def get_position(self):
        # load position
        return  x, y, z

    def take_observation(self):
        # load state, value_state
        return state, value_state

    def compute_action(self):
        state, value_state=self.take_observation()
        action = self.ppo_agent.select_action(state,value_state)
        return action

    def _done(self):
        # for max_episode steps
        done1 = bool(self.iterator == self.max_episode_steps)
        # for additional terminal case
        # done2=True if (case) else False
        done2 = False
        return done1 or done2

    def reset(self):
        Simulation.reset(self.root) # reset simulation here
        self.iterator = 0
        obs, value_states = self.take_observation()
        return obs, value_states


class Env(Set_env):
    def __init__(self, node, train_or_test,has_continuous_action_space , continue_training , load_pt_file_name):
        super(Env, self).__init__(node, train_or_test,has_continuous_action_space, continue_training , load_pt_file_name)

    def step(self, a):
        for i in range(self.multi_animate_step):
            Simulation.animate(self.root, a)
        done = self._done()
        # reward = -1.0 if not done else 0.0
        reward = self._reward()
        self.iterator +=1
        obs, value_states = self.take_observation()
        return (obs, reward, done, {}, value_states)

    def reset(self):
        obs, value_states = super().reset()
        return obs, value_states




def train(env):
    writer=SummaryWriter()

    print("============================================================================================")
    ####### initialize environment hyperparameters ######
    max_training_timesteps = int(9e6)   # break training loop if timeteps > max_training_timesteps

    print_freq = env.max_ep_len * 1 #10        # print avg reward in the interval (in num timesteps)
    log_freq = env.max_ep_len * 2           # log avg reward in the interval (in num timesteps)
    save_model_freq = int(1e4)          # save model frequency (in num timesteps)

    action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
    ################ PPO hyperparameters ################
    update_timestep = env.max_ep_len * 2      # update policy every n timesteps
    random_seed = 0         # set random seed if required (0 = no random seed)
    #####################################################

    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten
    log_dir = "PPO_logs"
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)
    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    #### create new log file for each run
    log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)
    #####################################################

    ################### checkpointing ###################
    directory = "PPO_preTrained"
    if not os.path.exists(directory):
          os.makedirs(directory)
    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
          os.makedirs(directory)

    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", env.max_ep_len)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("state space dimension : ", env.state_dim)
    print("value state space dimension : ", env.value_state_dim)
    print("action space dimension : ", env.action_dim)
    if env.has_continuous_action_space:
        print("Initializing a continuous action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("starting std of action distribution : ", action_std)
        print("decay rate of std of action distribution : ", action_std_decay_rate)
        print("minimum std of action distribution : ", min_action_std)
        print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")
    else:
        print("continuous action",env.has_continuous_action_space)
        print("Initializing a discrete action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", env.K_epochs)
    print("PPO epsilon clip : ", env.eps_clip)
    print("discount factor (gamma) : ", env.gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", env.lr_actor)
    print("optimizer learning rate critic : ", env.lr_critic)
    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    #####################################################

    print("============================================================================================")
    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent = PPO.PPO(env.state_dim, env.action_dim,env.value_state_dim, env.lr_actor, env.lr_critic, env.gamma, env.K_epochs, env.eps_clip,env.has_continuous_action_space, action_std)
    if load_pt_file:
        print('######################################loat_trained_file###############################')
        directory = "PPO_preTrained" + '/' + env_name + '/'
        print("loaded")
        checkpoint_path = directory + load_model_pt
        print("loading network from : " + checkpoint_path)
        ppo_agent.load_for_train(checkpoint_path)
    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    current_time = start_time
    print("Started training at (GMT) : ", start_time)
    print("============================================================================================")

    # logging file
    log_f = open(log_f_name,"w+")
    log_f.write('episode,timestep,reward\n')

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 1
    print_running_time_step =0
    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0
    checkpoint_path = directory + "PPO_" +"{}_state_{}_value_{}_{}.pth".format(env_name,env.state_dim, env.value_state_dim,datetime.now().replace(
                                                                                              microsecond=0))
    best_reward = 0

    # training loop
    while time_step <= max_training_timesteps:
        current_ep_reward = 0
        state, value_states = env.reset()
        for t in range(1, env.max_ep_len+1):
            # select action with policy
            action = ppo_agent.select_action(state,value_states)
            state, reward, done, _ ,value_states= env.step(action)
            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step +=1
            print_running_time_step += 1
            current_ep_reward += reward
            print_running_reward +=reward
            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()

            # if continuous action space; then decay action std of ouput action distribution
            # log in logging file
            if env.has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)
            if time_step % log_freq == 0:

                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0

            # printing average reward
            if time_step % print_freq == 0:

                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)
                print_avg_reward_step = print_running_reward / print_running_time_step
                print_avg_reward_step = round(print_avg_reward_step, 2)
                print("Episode : {}\t Timestep : {}\t\t Average Reward per episode : {}\t Average Reward per step : {} \t\tTime : {}\tTotal : {}".format(i_episode, time_step, print_avg_reward,print_avg_reward_step,datetime.now().replace(microsecond=0) - current_time,datetime.now().replace(microsecond=0) - start_time))
                current_time = datetime.now().replace(microsecond=0)
                writer.add_scalar("Reward/episode", print_avg_reward, i_episode)
                print_running_reward = 0
                print_running_episodes = 0
                print_running_time_step = 0
                # save model weights
                if time_step >= save_model_freq and best_reward < print_avg_reward_step:
                    best_reward = print_avg_reward_step
                    print("--------------------------------------------------------------------------------------------")
                    print("saving model at : " + checkpoint_path)
                    ppo_agent.save(checkpoint_path)
                    print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                    print("--------------------------------------------------------------------------------------------")
            if done:
                break

        print_running_episodes += 1
        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

    log_f.close()
    env.close()
    writer.close()
    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")
