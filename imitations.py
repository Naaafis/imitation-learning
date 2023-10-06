import os
import numpy as np
import gym
from pyglet.window import key


def load_imitations(data_folder):
    """
    1.1 a)
    Given the folder containing the expert imitations, the data gets loaded and
    stored it in two lists: observations and actions.
                    N = number of (observation, action) - pairs
    data_folder:    python string, the path to the folder containing the
                    observation_%05d.npy and action_%05d.npy files
    return:
    observations:   python list of N numpy.ndarrays of size (96, 96, 3)
    actions:        python list of N numpy.ndarrays of size 3
    """
    
    observations = []
    actions = []
    
    files = os.listdir(data_folder)
    
    for filename in files:
        if filename.startswith('observation'):
            observations.append(np.load(data_folder + filename))
        elif filename.startswith('action'):
            actions.append(np.load(data_folder + filename))
    
    return observations, actions


def save_imitations(data_folder, actions, observations):
    """
    1.1 f)
    Save the lists actions and observations in numpy .npy files that can be read
    by the function load_imitations.
                    N = number of (observation, action) - pairs
    data_folder:    python string, the path to the folder containing the
                    observation_%05d.npy and action_%05d.npy files
    observations:   python list of N numpy.ndarrays of size (96, 96, 3)
    actions:        python list of N numpy.ndarrays of size 3
    """
    #pass
    # Ensure the directory exists
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    
    # Save each observation and action pair
    for idx, (obs, act) in enumerate(zip(observations, actions)):
        obs_filename = os.path.join(data_folder, f'observation_{idx:05d}.npy')
        act_filename = os.path.join(data_folder, f'action_{idx:05d}.npy')
        
        np.save(obs_filename, obs)
        np.save(act_filename, act)
        
# chatGPT code below 

def save_dagger_imitations(data_folder, actions, observations):
    """
    Save the lists actions and observations in numpy .npy files that can be read
    by the function load_imitations.
                    N = number of (observation, action) - pairs
    data_folder:    python string, the path to the folder containing the
                    observation_%05d.npy and action_%05d.npy files
    observations:   python list of N numpy.ndarrays of size (96, 96, 3)
    actions:        python list of N numpy.ndarrays of size 3
    """
    # Ensure the directory exists
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    
    # Determine the starting index by checking existing files
    existing_files = os.listdir(data_folder)
    existing_obs_files = [f for f in existing_files if 'observation_' in f]
    
    if existing_obs_files:
        latest_index = max([int(f.split('_')[1].split('.')[0]) for f in existing_obs_files])
        start_idx = latest_index + 1
    else:
        start_idx = 0
    
    # Save each observation and action pair starting from the determined index
    for idx, (obs, act) in enumerate(zip(observations, actions), start=start_idx):
        obs_filename = os.path.join(data_folder, f'observation_{idx:05d}.npy')
        act_filename = os.path.join(data_folder, f'action_{idx:05d}.npy')
        
        np.save(obs_filename, obs)
        np.save(act_filename, act)



class ControlStatus:
    """
    Class to keep track of key presses while recording imitations.
    """

    def __init__(self):
        self.stop = False
        self.save = False
        self.quit = False
        self.steer = 0.0
        self.accelerate = 0.0
        self.brake = 0.0
        self.switch = 0  # 0 for AI driving, 1 for user driving
        self.reset = False

    def key_press(self, k, mod):
        #print(f"Key pressed: {k}")  # Debug statement
        if k == key.ESCAPE: self.quit = True
        if k == key.SPACE: self.stop = True
        if k == key.TAB: self.save = True
        if k == key.LEFT: self.steer = -0.8
        if k == key.RIGHT: self.steer = +0.8
        if k == key.UP: self.accelerate = +0.4
        if k == key.DOWN: self.brake = +0.8
        if k == key.S: self.switch = 1 - self.switch  # Toggle between 0 and 1
        if k == key.R: self.reset = True

    def key_release(self, k, mod):
        #print(f"Key released: {k}")  # Debug statement
        if k == key.LEFT and self.steer < 0.0: self.steer = 0.0
        if k == key.RIGHT and self.steer > 0.0: self.steer = 0.0
        if k == key.UP: self.accelerate = 0.0
        if k == key.DOWN: self.brake = 0.0
        if k == key.R: self.reset = False



def record_imitations(imitations_folder):
    """
    Function to record own imitations by driving the car in the gym car-racing
    environment.
    imitations_folder:  python string, the path to where the recorded imitations
                        are to be saved

    The controls are:
    arrow keys:         control the car; steer left, steer right, gas, brake
    ESC:                quit and close
    SPACE:              restart on a new track
    TAB:                save the current run
    """
    env = gym.make('CarRacing-v0').env
    status = ControlStatus()
    total_reward = 0.0

    while not status.quit:
        observations = []
        actions = []
        # get an observation from the environment
        observation = env.reset()
        print("Environment reset.")  # Debug statement
        env.render()

        # set the functions to be called on key press and key release
        env.viewer.window.on_key_press = status.key_press
        env.viewer.window.on_key_release = status.key_release

        while not status.stop and not status.save and not status.quit:
            # collect all observations and actions
            observations.append(observation.copy())
            actions.append(np.array([status.steer, status.accelerate,
                                     status.brake]))
            print(f"Added observation and action: {status.steer, status.accelerate, status.brake}")  # Debug statement
            # submit the users' action to the environment and get the reward
            # for that step as well as the new observation (status)
            observation, reward, done, info = env.step([status.steer,
                                                        status.accelerate,
                                                        status.brake])
            total_reward += reward
            env.render()

        if status.save:
            save_imitations(imitations_folder, actions, observations)
            print("Imitations saved.")  # Debug statement
            status.save = False

        status.stop = False
        env.close()
