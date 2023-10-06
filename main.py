import os
import re
import sys
import numpy as np
import torch
import gym
import shutil
import random

import matplotlib.pyplot as plt
from training import train
from imitations import record_imitations
from imitations import ControlStatus
from imitations import save_imitations
from imitations import load_imitations
from imitations import save_dagger_imitations

directory = "/Users/cheffbcookin/Desktop/518/hw1"  ######## change that! ########
trained_network_file = os.path.join(directory, '/Users/cheffbcookin/Desktop/518/hw1/data/train.t7')
imitations_folder = os.path.join(directory, '/Users/cheffbcookin/Desktop/518/hw1/data/teacher/')
# imitations_folder = os.path.join(directory, 'data/grey_teacher/')
# imitations_folder = os.path.join(directory, '/Users/cheffbcookin/Desktop/518/hw1/data/cropped_teacher/')

import os

def count_actions(actions):
    """
    Count the occurrences of each unique action in the provided list.
    actions:        python list of N numpy.ndarrays of size 3
    return:         dictionary with action tuples as keys and counts as values
    """
    # Define the possible actions
    possible_actions = [
        (-0.8, 0.0, 0.0),  # Steer left
        (0.8, 0.0, 0.0),   # Steer right
        (0.0, 0.4, 0.0),   # Accelerate
        (0.0, 0.0, 0.8),   # Brake
        (0.0, 0.0, 0.0),   # No action
        (-0.8, 0.4, 0.0),  # Steer left and accelerate
        (0.8, 0.4, 0.0),   # Steer right and accelerate
        (-0.8, 0.0, 0.8),  # Steer left and brake
        (0.8, 0.0, 0.8)    # Steer right and brake
    ]
    
    # Initialize a dictionary to store the counts
    action_counts = {action: 0 for action in possible_actions}
    
    # Count each action
    for action in actions:
        action_tuple = tuple(action)
        if action_tuple in action_counts:
            action_counts[action_tuple] += 1
    
    return action_counts

# Function below written with a really good GPT-4 prompt:
def remove_do_nothing_data(data_folder):
    """
    Remove 9/10 of the "do nothing" data from the dataset directory.
    """
    # Load the imitations
    _, actions = load_imitations(data_folder)
    
    # Define the "do nothing" action
    do_nothing_action = (0.0, 0.0, 0.0)
    
    # Count the occurrences of "do nothing" action
    do_nothing_count = sum(1 for action in actions if np.array_equal(tuple(action), do_nothing_action))
    
    # Calculate the number of "do nothing" occurrences to remove
    to_remove = int(9/10.0 * do_nothing_count)
    
    # Counter to keep track of the number of "do nothing" occurrences removed
    removed_count = 0
    
    # Create a list of indices to remove
    indices_to_remove = []
    indices_to_remove = [i for i, action in enumerate(actions) if tuple(action) == do_nothing_action]
    
    # Shuffle the indices to ensure randomness
    random.shuffle(indices_to_remove)
    
    # Only keep the first 'to_remove' indices
    indices_to_remove = indices_to_remove[:to_remove]
    
    # Delete the files for the selected indices
    for idx in indices_to_remove:
        obs_filename = os.path.join(data_folder, f'observation_{idx:05d}.npy')
        act_filename = os.path.join(data_folder, f'action_{idx:05d}.npy')
        
        if os.path.exists(obs_filename):
            os.remove(obs_filename)
            if not os.path.exists(obs_filename):  # Check if file was actually removed
                removed_count += 1
                #print(f"Deleted observation file: {obs_filename}")
        
        if os.path.exists(act_filename):
            os.remove(act_filename)
            # if not os.path.exists(act_filename):  # Check if file was actually removed
            #     #print(f"Deleted action file: {act_filename}")
            
    print(f"Removed {removed_count} of {do_nothing_count} 'do nothing' occurrences.")
    
    
    # # Sort and reindex files
    # all_obs_files = sorted([f for f in os.listdir(data_folder) if 'observation_' in f])
    # all_act_files = sorted([f for f in os.listdir(data_folder) if 'action_' in f])
    # for i, (obs_file, act_file) in enumerate(zip(all_obs_files, all_act_files)):
    #     new_obs_name = os.path.join(data_folder, f'observation_{i:05d}.npy')
    #     new_act_name = os.path.join(data_folder, f'action_{i:05d}.npy')
    #     os.rename(os.path.join(data_folder, obs_file), new_obs_name)
    #     os.rename(os.path.join(data_folder, act_file), new_act_name)

# Intended to be used to test a single iteration of the DAGGER algorithm. Written with a really good GPT-4 prompt:
'''
We need to figure out how to let the AI drive but we take control by pressing a button. In main.py, 
the evaluate function steps through the environment by getting each frame with env.render then having 
the model predict what to do at that frame. What we will do is combine this mechanism with one from 
record imitation that records our own user data while also letting us press keys to control the 
environment. Essentially, we need to add a new control functionality in imitation.py that takes in a 
new key and sets a new state called switch. The key for this will be "S". Look into the pyglet library 
and figure out the key.S actual value. Anyways, write me a function that basically adds an additional 
state into control status for switch, and if switch == 1, then we follow the user and record imitations,
as record imitations already does, we can just call the function itself to be honest. when s == 0, the 
AI is driving as it does in the evaluate function. This would essentially be a fourth possible argument 
one could run with main.py called "dagger"
'''
def switch():
    infer_action = torch.load(trained_network_file, map_location='cpu')
    infer_action.eval()
    env = gym.make('CarRacing-v0')
    device = torch.device('cpu')
    infer_action = infer_action.to(device)
    status = ControlStatus()

    for episode in range(5):
        observations = []
        actions = []
        observation = env.reset()

        # env.viewer.window.on_key_press = status.key_press
        # env.viewer.window.on_key_release = status.key_release
        '''
        The error you're seeing is due to the fact that the env object is wrapped by TimeLimit, 
        which doesn't have the viewer attribute directly accessible. The underlying environment, 
        however, does have this attribute.

        To fix this, you can access the underlying environment using env.unwrapped. 
        '''
        env.unwrapped.viewer.window.on_key_press = status.key_press
        env.unwrapped.viewer.window.on_key_release = status.key_release


        reward_per_episode = 0
        while not status.quit:
            env.render()
            
            # Crop the observation for CutNetwork
            center_y, center_x = observation.shape[0] // 2, observation.shape[1] // 2
            top_left_y, top_left_x = center_y - 30, center_x - 30
            bottom_right_y, bottom_right_x = center_y + 30, center_x + 30
            observation = observation[top_left_y:bottom_right_y, top_left_x:bottom_right_x, :]

            
            # Check for reset
            if status.reset:
                observation = env.reset()
                observations = []
                actions = []
                continue

            if status.switch == 0:  # AI driving
                observation = torch.Tensor(np.ascontiguousarray(observation[None])).permute(0, 3, 1, 2).to(device)
                action_scores = infer_action(observation)
                steer = action_scores[0][0].detach().item()
                gas = action_scores[0][1].detach().item()
                brake = action_scores[0][2].detach().item()
            else:  # User driving
                steer = status.steer
                gas = status.accelerate
                brake = status.brake
                observations.append(observation.copy())
                actions.append(np.array([steer, gas, brake]))

            observation, reward, done, info = env.step([steer, gas, brake])
            reward_per_episode += reward

        if status.save:
            save_imitations(imitations_folder, actions, observations)
            observations, actions = load_imitations(imitations_folder)
            action_counts = count_actions(actions)
            # Print the counts
            for action, count in action_counts.items():
                print(f"Action {action}: {count} occurrences")
            status.save = False

        print('episode %d \t reward %f' % (episode, reward_per_episode))

'''
When we call dagger, we have to somehow get a measure this "regret"

First describe how the term regret works in online learning. You may want to describe a bit about how online working is done. Our method of collecting data while the model is performing is a method of online learning. Our goal is to ultimately fully implement this dagger functionality in main.py, with the final regret being plotted. Not sure how to go about that
'''

def dagger():
    """
    Implement the DAGGER algorithm to iteratively collect data and train the model.
    """
    # regrets = []  # List to store cumulative regret values
    # cumulative_regret = 0  # Initialize cumulative regret to 0

    # # Load the initial policy πˆ1 (e.g., a pre-trained model or a random policy)
    # infer_action = torch.load(trained_network_file, map_location='cpu')
    # infer_action.eval()
    # env = gym.make('CarRacing-v0')
    # status = ControlStatus()
    # device = torch.device('cpu')
    # infer_action = infer_action.to(device)
    
     # Check if checkpoints directory exists, if not, create it
    if not os.path.exists('./data/checkpoints'):
        os.makedirs('./data/checkpoints')

    checkpoints_folder = './data/checkpoints'
    
    checkpoints = [f for f in os.listdir(checkpoints_folder) if re.match(r'checkpoint_\d+.t7', f)]
    start_iteration = max([int(re.findall(r'\d+', chkpt)[0]) for chkpt in checkpoints], default=0) 
    print(f"Starting from iteration {start_iteration}")
    
    # Load regrets from checkpoints if exists
    if os.path.exists('./data/checkpoints/regrets.npy'):
        regrets = np.load('./data/checkpoints/regrets.npy').tolist()
    else:
        regrets = []
        
    cumulative_regret = sum(regrets) if regrets else 0
    
    if start_iteration > 0:
        # load the model weights from the last checkpoint
        infer_action = torch.load(os.path.join(checkpoints_folder, f'checkpoint_{start_iteration}.t7'), map_location='cpu')

    # train for the next iteration onwards
    start_iteration += 1

    N = 10  # Number of iterations for DAGGER
    #for i in range(N):
    for i in range(start_iteration, N):
        # Note that reloading the model at each iteration helps propagate the improvements of the DAGGER algorithm
        # re-render the environment at each iteration fully because the close function is a bit buggy
        # Load the initial policy πˆ1 (e.g., a pre-trained model or a random policy)
        infer_action = torch.load(trained_network_file, map_location='cpu')
        infer_action.eval()
        env = gym.make('CarRacing-v0')
        status = ControlStatus()
        device = torch.device('cpu')
        infer_action = infer_action.to(device)
        observations = []
        actions = []
        expert_actions = []

        # Sample T-step trajectories using πi (not gonna lie, I haven't been keeping track of the number of steps at each iteration)
        observation = env.reset()
        env.render()
        
        env.unwrapped.viewer.window.on_key_press = status.key_press
        env.unwrapped.viewer.window.on_key_release = status.key_release

        while not status.stop and not status.save and not status.quit:
            env.render()
            
            # # Crop the observation for CutNetwork
            # center_y, center_x = observation.shape[0] // 2, observation.shape[1] // 2
            # top_left_y, top_left_x = center_y - 30, center_x - 30
            # bottom_right_y, bottom_right_x = center_y + 30, center_x + 30
            # observation = observation[top_left_y:bottom_right_y, top_left_x:bottom_right_x, :]
            
            # Check for reset
            if status.reset:
                observation = env.reset()
                observations = []
                actions = []
                expert_actions = []
                continue

            observation_tensor = torch.Tensor(np.ascontiguousarray(observation[None])).permute(0, 3, 1, 2).to(device)
            # action_scores = infer_action(observation_tensor)
            # action_scores = infer_action(observation_tensor)
            # steer = action_scores[0][0].detach().item()
            # gas = action_scores[0][1].detach().item()
            # brake = action_scores[0][2].detach().item()
            
            action_scores = infer_action(observation_tensor)

            steer, gas, brake = infer_action.scores_to_action(action_scores) # for classification task we need to convert classes to actions

            # If switch is on, use expert's action
            if status.switch:
                action = [status.steer, status.accelerate, status.brake]
                expert_actions.append(action)
            else:
                action = [steer, gas, brake]
                expert_actions.append([status.steer, status.accelerate, status.brake])

            observations.append(observation.copy())
            actions.append(action)

            observation, _, _, _ = env.step(action)

        # Compute regret for this iteration
        regret = sum([loss_fn(a, e_a) for a, e_a in zip(actions, expert_actions)])
        cumulative_regret += regret
        regrets.append(cumulative_regret)
        # Convert the regrets list to a numpy array
        regrets_array = np.array(regrets)


        # # Save observations and expert actions to the imitations folder
        # save_imitations(imitations_folder, expert_actions, observations)
        if status.save:
            #save_imitations(imitations_folder, actions, observations)
            save_dagger_imitations(imitations_folder, actions, observations)
            print("Imitations saved.")  # Debug statement
            # balance_dataset(imitations_folder)
            # remove_do_nothing_data(imitations_folder)
            observations, actions = load_imitations(imitations_folder)
            action_counts = count_actions(actions)
            # Print the counts for each action
            for action, count in action_counts.items():
                print(f"Action {action}: {count} occurrences")
            status.save = False
            
            # Save the regrets array to the checkpoints directory
            np.save('data/checkpoints/regrets.npy', regrets_array)
            print(f"Iteration {i}: regret = {regret}, cumulative regret = {cumulative_regret}")

        status.stop = False
        env.close()

        # Train classifier πˆi+1 on the collected data
        train(imitations_folder, trained_network_file)
        
        # Save the model weights as a checkpoint after every iteration
        checkpoint_path = os.path.join(checkpoints_folder, f"checkpoint_{i}.t7")
        torch.save(infer_action, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    # Plotting the cumilative regret
    import matplotlib.pyplot as plt
    plt.plot(regrets)
    plt.xlabel('Iterations')
    plt.ylabel('Cumulative Regret')
    plt.title('DAGGER Cumulative Regret over Iterations')
    plt.show()
    
    # plotting the regret per iteration
    plt.plot([regrets[i] - regrets[i-1] for i in range(1, len(regrets))])
    plt.xlabel('Iterations')
    plt.ylabel('Regret')
    plt.title('DAGGER Regret per Iteration')
    plt.show()
    



# Assuming a simple loss function (MSE) for computing regret
def loss_fn(action, expert_action):
    return np.sum((np.array(action) - np.array(expert_action))**2)

        
# Had GPT write this for convinience
def clear_teacher_directory():
    """
    Removes the teacher directory.
    """
    if os.path.exists(imitations_folder):
        shutil.rmtree(imitations_folder)
        print(f"'{imitations_folder}' has been removed.")
        # create a new empty directory called 'teacher'
        os.makedirs(imitations_folder)
        print(f"'{imitations_folder}' has been created.")
    else:
        print(f"'{imitations_folder}' does not exist.")
        os.makedirs(imitations_folder)
        print(f"'{imitations_folder}' has been created.")


def evaluate():
    """
    """
    infer_action = torch.load(trained_network_file, map_location='cpu')
    infer_action.eval()
    env = gym.make('CarRacing-v0')
    # you can set it to torch.device('cuda') in case you have a gpu
    device = torch.device('cpu')
    infer_action = infer_action.to(device)


    for episode in range(5):
        observation = env.reset()

        reward_per_episode = 0
        for t in range(500):
            env.render()
            # action_scores = infer_action(torch.Tensor(
            #     np.ascontiguousarray(observation[None])).to(device))
            
            # # Crop the observation for CutNetwork
            # center_y, center_x = observation.shape[0] // 2, observation.shape[1] // 2
            # top_left_y, top_left_x = center_y - 30, center_x - 30
            # bottom_right_y, bottom_right_x = center_y + 30, center_x + 30
            # observation = observation[top_left_y:bottom_right_y, top_left_x:bottom_right_x, :]
            # print(observation.shape)
            
            observation = torch.Tensor(np.ascontiguousarray(observation[None])).permute(0, 3, 1, 2).to(device)
            # for grey network
            # observation = torch.mean(observation, dim=1, keepdim=True)
            # for cut network
            action_scores = infer_action(observation)

            steer, gas, brake = infer_action.scores_to_action(action_scores) # for classification task we need to convert classes to actions
            print("(action_scores): ", action_scores[0][0], action_scores[0][1], action_scores[0][2]) # debug
            # steer, gas, brake = action_scores[0][0], action_scores[0][1], action_scores[0][2] # for regression task we don't need to convert classes to actions
            # Detach the tensors and convert them to Python scalars
            # steer = action_scores[0][0].detach().item()
            # gas = action_scores[0][1].detach().item()
            # brake = action_scores[0][2].detach().item()
            print("(steer, gas, brake): ", steer, gas, brake) # debug
            observation, reward, done, info = env.step([steer, gas, brake])
            reward_per_episode += reward

        print('episode %d \t reward %f' % (episode, reward_per_episode))


def calculate_score_for_leaderboard():
    """
    Evaluate the performance of the network. This is the function to be used for
    the final ranking on the course-wide leader-board, only with a different set
    of seeds. Better not change it.
    """
    infer_action = torch.load(trained_network_file, map_location='cpu')
    infer_action.eval()
    env = gym.make('CarRacing-v0')
    # you can set it to torch.device('cuda') in case you have a gpu
    device = torch.device('cpu')

    seeds = [22597174, 68545857, 75568192, 91140053, 86018367,
             49636746, 66759182, 91294619, 84274995, 31531469]
    total_reward = 0

    for episode in range(10):
        env.seed(seeds[episode])
        observation = env.reset()
        reward_per_episode = 0
        
        for t in range(600):
            env.render()
            action_scores = infer_action(torch.Tensor(
                np.ascontiguousarray(observation[None])).to(device))

            steer, gas, brake = infer_action.scores_to_action(action_scores)
            observation, reward, done, info = env.step([steer, gas, brake])
            reward_per_episode += reward
        
        # Crop the observation for CutNetwork
        # center_y, center_x = observation.shape[0] // 2, observation.shape[1] // 2
        # top_left_y, top_left_x = center_y - 30, center_x - 30
        # bottom_right_y, bottom_right_x = center_y + 30, center_x + 30
        # observation = observation[top_left_y:bottom_right_y, top_left_x:bottom_right_x, :]
        # print(observation.shape)
        
        # observation = torch.Tensor(np.ascontiguousarray(observation[None])).permute(0, 3, 1, 2).to(device)
        # # for grey network
        # # observation = torch.mean(observation, dim=1, keepdim=True)
        # # for cut network
        # action_scores = infer_action(observation)
        # print("(action_scores): ", action_scores[0][0], action_scores[0][1], action_scores[0][2]) # debug
        # steer, gas, brake = action_scores[0][0], action_scores[0][1], action_scores[0][2] # for regression task we don't need to convert classes to actions
        # Detach the tensors and convert them to Python scalars
        # steer = action_scores[0][0].detach().item()
        # gas = action_scores[0][1].detach().item()
        # brake = action_scores[0][2].detach().item()
        
        print("(steer, gas, brake): ", steer, gas, brake) # debug
        observation, reward, done, info = env.step([steer, gas, brake])
        reward_per_episode += reward

        print('episode %d \t reward %f' % (episode, reward_per_episode))
        total_reward += reward_per_episode

    print('---------------------------')
    print(' total score: %f' % (total_reward / 10))
    print('---------------------------')


if __name__ == "__main__":
    if len(sys.argv) == 1 or sys.argv[1] == "train":
        train(imitations_folder, trained_network_file)
    elif sys.argv[1] == "teach":
        record_imitations(imitations_folder)
    elif sys.argv[1] == "test":
        evaluate()
    elif sys.argv[1] == "score":
        calculate_score_for_leaderboard()
    elif sys.argv[1] == "dagger":
        dagger()
    elif sys.argv[1] == "clear":
        clear_teacher_directory()
    elif sys.argv[1] == "switch":
        switch()


    else:
        print('This command is not supported, valid options are: train, teach, '
              'test and score.')
