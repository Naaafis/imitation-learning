import torch
import torch.nn.functional as F

import torch.nn as nn
import torch.nn.functional as F

class GreyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        # convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2)
        self.bn4 = nn.BatchNorm2d(256)
        
        # fully connected layers
        self.fc1 = nn.Linear(256 * 5 * 5, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc_out = nn.Linear(128, 3)  # 3 nodes for steer, gas, and brake
        self.dropout = nn.Dropout(0.5)

    def forward(self, observation):
        # convolutional layers
        x = F.relu(self.bn1(self.conv1(observation)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # fully connected layers
        x = x.reshape(-1, 256 * 5 * 5)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc_out(x)
        
        return x


class CutNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        # convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2)
        self.bn4 = nn.BatchNorm2d(256)
        
        # fully connected layers
        # self.fc1 = nn.Linear(256 * 2 * 2, 512)
        self.fc1 = nn.Linear(256 * 2 * 2 + 4 + 1 + 1 + 1, 512)  # +4 for ABS, +1 for speed, +1 for steering, +1 for gyroscope
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc_out = nn.Linear(128, 3)  # 3 nodes for steer, gas, and brake
        self.dropout = nn.Dropout(0.5)
        
        '''
        Input shape: (batch_size, 3, 60, 60)
        After conv1: (batch_size, 32, 29, 29) # (60-3)/2 + 1 = 29
        After conv2: (batch_size, 64, 14, 14) # (29-3)/2 + 1 = 14
        After conv3: (batch_size, 128, 6, 6) # (14-3)/2 + 1 = 6
        After conv4: (batch_size, 256, 2, 2) # (6-3)/2 + 1 = 2
        '''

    def forward(self, observation):
        # convolutional layers
        
       # Ensure the input is a 4D tensor with shape: (batch_size, channels, height, width)
        assert observation.dim() == 4, f"Expected 4D tensor, got shape {observation.shape}"
        
        speed, abs_sensors, steering, gyroscope = self.extract_sensor_values(observation, observation.shape[0])
        
        # Crop the observation for CutNetwork
        center_y, center_x = observation.shape[2] // 2, observation.shape[3] // 2
        top_left_y, top_left_x = center_y - 30, center_x - 30
        bottom_right_y, bottom_right_x = center_y + 30, center_x + 30
        
        # Ensure the cropping indices are valid
        assert top_left_y >= 0 and top_left_x >= 0 and bottom_right_y <= observation.shape[2] and bottom_right_x <= observation.shape[3], "Invalid cropping indices"
        
        # Crop the observation
        x = observation[:, :, top_left_y:bottom_right_y, top_left_x:bottom_right_x]
        
        # print("CutNetwork input shape:", x.shape)
        
        x = F.relu(self.bn1(self.conv1(x)))
        #print("Shape after conv1:", x.shape)
        
        x = F.relu(self.bn2(self.conv2(x)))
        #print("Shape after conv2:", x.shape)
        
        x = F.relu(self.bn3(self.conv3(x)))
        #print("Shape after conv3:", x.shape)
        
        x = F.relu(self.bn4(self.conv4(x)))
        #print("Shape after conv4:", x.shape)
        
        # fully connected layers
        x = x.reshape(-1, 256 * 2 * 2)
        # x = x.reshape(-1, 256 * 5 * 5)
        
        # Concatenate sensor values to the input of the first fully connected layer
        x = torch.cat((x, speed, abs_sensors, steering, gyroscope), dim=1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc_out(x)
        
        # print("CutNetwork output shape:", x.shape)
        
        return x
    
    ''' # Need to have a scores to action for regression task but it is simple
    # steer = action_scores[0][0].detach().item()
    # gas = action_scores[0][1].detach().item()
    # brake = action_scores[0][2].detach().item()
    
    action_scores = infer_action(observation_tensor)

    steer, gas, brake = infer_action.scores_to_action(action_scores) # for classification task we need to convert classes to actions
    '''
    
    def scores_to_action(self, scores):
        steer = scores[0][0].detach().item()
        gas = scores[0][1].detach().item()
        brake = scores[0][2].detach().item()
        
        return steer, gas, brake
    
    def extract_sensor_values(self, observation, batch_size):
        """
        observation:    python list of batch_size many torch.Tensors of size
                        (96, 96, 3)
        batch_size:     int
        return          torch.Tensors of size (batch_size, 1),
                        torch.Tensors of size (batch_size, 4),
                        torch.Tensors of size (batch_size, 1),
                        torch.Tensors of size (batch_size, 1)
        """
        # print("Observation shape:", observation.shape)
        # sliced_tensor = observation[:, 84:94, 18:25:2, 2]
        # print("Sliced tensor shape:", sliced_tensor.shape)

        speed_crop = observation[:, 84:94, 12, 0].reshape(batch_size, -1)
        speed = speed_crop.sum(dim=1, keepdim=True) / 255
        # abs_crop = observation[:, 84:94, 18:25:2, 2].reshape(batch_size, 10, 4)
        # keeping 3 channels to start and taking mean across them
        observation = observation.permute(0, 2, 3, 1)  # Change shape to [batch_size, height, width, channels]
        abs_crop = observation[:, 84:94, 18:25:2, 2]  # Extract the third channel
        # abs_crop = observation[:, 84:94, 18:25:2].reshape(batch_size, 10, 4, 3)

        # abs_crop = observation[:, 84:94, 18:25:2].mean(dim=1).reshape(batch_size, 10, 4)

        abs_sensors = abs_crop.sum(dim=1) / 255
        steer_crop = observation[:, 88, 38:58, 1].reshape(batch_size, -1)
        steering = steer_crop.sum(dim=1, keepdim=True)
        gyro_crop = observation[:, 88, 58:86, 0].reshape(batch_size, -1)
        gyroscope = gyro_crop.sum(dim=1, keepdim=True)
        return speed, abs_sensors.reshape(batch_size, 4), steering, gyroscope



class RegressionNetwork(torch.nn.Module):

    def __init__(self):
        super().__init__()
        
        # convolutional layers
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=2)
        self.bn1 = torch.nn.BatchNorm2d(32)
        
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.bn2 = torch.nn.BatchNorm2d(64)
        
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.bn3 = torch.nn.BatchNorm2d(128)
        
        self.conv4 = torch.nn.Conv2d(128, 256, kernel_size=3, stride=2)
        self.bn4 = torch.nn.BatchNorm2d(256)
        
        # fully connected layers
        self.fc1 = torch.nn.Linear(256 * 5 * 5 + 4 + 1 + 1 + 1, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 128)
        self.fc_out = torch.nn.Linear(128, 3)  # 3 nodes for steer, gas, and brake
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, observation):
        # Extract sensor values
        speed, abs_sensors, steering, gyroscope = self.extract_sensor_values(observation, observation.shape[0])
        
        # convolutional layers
        x = F.relu(self.bn1(self.conv1(observation)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # fully connected layers
        x = x.reshape(-1, 256 * 5 * 5)
        
        # concatenate extra sensor values into the fully connected layer
        x = torch.cat((x, speed, abs_sensors, steering, gyroscope), dim=1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc_out(x)
        
        return x
    
    

    def extract_sensor_values(self, observation, batch_size):
        """
        observation:    python list of batch_size many torch.Tensors of size
                        (96, 96, 3)
        batch_size:     int
        return          torch.Tensors of size (batch_size, 1),
                        torch.Tensors of size (batch_size, 4),
                        torch.Tensors of size (batch_size, 1),
                        torch.Tensors of size (batch_size, 1)
        """
        # print("Observation shape:", observation.shape)
        # sliced_tensor = observation[:, 84:94, 18:25:2, 2]
        # print("Sliced tensor shape:", sliced_tensor.shape)

        speed_crop = observation[:, 84:94, 12, 0].reshape(batch_size, -1)
        speed = speed_crop.sum(dim=1, keepdim=True) / 255
        # abs_crop = observation[:, 84:94, 18:25:2, 2].reshape(batch_size, 10, 4)
        # keeping 3 channels to start and taking mean across them
        observation = observation.permute(0, 2, 3, 1)  # Change shape to [batch_size, height, width, channels]
        abs_crop = observation[:, 84:94, 18:25:2, 2]  # Extract the third channel
        # abs_crop = observation[:, 84:94, 18:25:2].reshape(batch_size, 10, 4, 3)

        # abs_crop = observation[:, 84:94, 18:25:2].mean(dim=1).reshape(batch_size, 10, 4)

        abs_sensors = abs_crop.sum(dim=1) / 255
        steer_crop = observation[:, 88, 38:58, 1].reshape(batch_size, -1)
        steering = steer_crop.sum(dim=1, keepdim=True)
        gyro_crop = observation[:, 88, 58:86, 0].reshape(batch_size, -1)
        gyroscope = gyro_crop.sum(dim=1, keepdim=True)
        return speed, abs_sensors.reshape(batch_size, 4), steering, gyroscope


class MultiClassNetwork(torch.nn.Module):
    def __init__(self):
        """
        1.1 d)
        Implementation of the network layers. The image size of the input
        observations is 96x96 pixels.
        """
        super().__init__()
        gpu = torch.device('cpu')
        
        # convolutional layers
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=2)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=2)
        
        # fully connected layers
        #self.fc1 = torch.nn.Linear(128 * 11 * 11, 128)
        
        # Modify the first fully connected layer to accept the additional sensor inputs
        # 4 additional inputs for speed, steering, gyroscope, and 4 for abs_sensors
        self.fc1 = torch.nn.Linear(128 * 11 * 11 + 4 + 1 + 1 + 1, 128)
        
        self.fc2 = torch.nn.Linear(128, 64)
        #self.fc3 = torch.nn.Linear(64, 9)
        #self.fc3 = torch.nn.Linear(64, 5)  # Reducing number of classes to 5
        self.fc_out = torch.nn.Linear(64, 4)  # 4 nodes for 4 arrow keys
    
    def forward(self, observation):
        """
        1.1 e)
        The forward pass of the network. Returns the prediction for the given
        input observation.
        observation:   torch.Tensor of size (batch_size, 96, 96, 3)
        return         torch.Tensor of size (batch_size, number_of_classes)
        """
        # Extract sensor values
        speed, abs_sensors, steering, gyroscope = self.extract_sensor_values(observation, observation.shape[0])
        
        # convolutional layers
        x = F.relu(self.conv1(observation))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # fully connected layers
        x = x.reshape(-1, 128 * 11 * 11)
        
        # concatonate extra sensor values into the fully connected layer
        x = torch.cat((x, speed, abs_sensors, steering, gyroscope), dim=1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        
        # # softmax to get probabilities over action-classes
        # x = F.softmax(x, dim=1)
        # Output layer with sigmoid activation for multi-class classification
        x = torch.sigmoid(self.fc_out(x))
        
        return x
    
    def actions_to_classes(self, actions):
        """
        Maps a given set of actions to its corresponding action-class representation.
        actions:        python list of N torch.Tensors of size 3
        return          python list of N torch.Tensors of size number_of_classes
        """
        possible_actions = []
        
        for action in actions:
            steer = action[0]
            gas = action[1]
            brake = action[2]
            
            # Initialize the classes with zeros
            classes = torch.zeros(4)
            
            # Set the values for each class based on the action
            classes[0] = 1 if steer < 0 else 0  # Steer left
            classes[1] = 1 if steer > 0 else 0  # Steer right
            classes[2] = 1 if gas > 0 else 0  # Gas
            classes[3] = 1 if brake > 0 else 0  # Brake
            
            possible_actions.append(classes)
                   
        return possible_actions
    
    def scores_to_action(self, scores):
        """
        Maps the scores predicted by the network to an action-class and returns
        the corresponding action [steer, gas, brake].
        scores:         python list of torch.Tensors of size number_of_classes
        return          (float, float, float)
        """
        threshold = 0.35 
        # Threshold for the classifications to activate controls                 
        # This value was chosen based on the results of the network during training
        # found that it was better to dynamically change the threshold based on the
        # frequency of the scores selected for each class
        # found that the vehicle would stuck on a non-action mode in certain situations
        
        # Check if scores for each class are above the threshold
        steer_left = scores[0][0] > 0.30
        steer_right = scores[0][1] > 0.23
        gas = scores[0][2] > 0.4
        brake = scores[0][3] > 0.08
        
        # Handle conflicting scenarios
        if steer_left and steer_right:
            steer = 0
            gas_val = 0.3
        else:
            steer = -0.8 if steer_left else (0.8 if steer_right else 0)
            gas_val = 1 if gas else 0
        
        brake_val = 1 if brake else 0
        
        return (steer, gas_val, brake_val)

    def extract_sensor_values(self, observation, batch_size):
        """
        observation:    python list of batch_size many torch.Tensors of size
                        (96, 96, 3)
        batch_size:     int
        return          torch.Tensors of size (batch_size, 1),
                        torch.Tensors of size (batch_size, 4),
                        torch.Tensors of size (batch_size, 1),
                        torch.Tensors of size (batch_size, 1)
        """
        # print("Observation shape:", observation.shape)
        # sliced_tensor = observation[:, 84:94, 18:25:2, 2]
        # print("Sliced tensor shape:", sliced_tensor.shape)

        speed_crop = observation[:, 84:94, 12, 0].reshape(batch_size, -1)
        speed = speed_crop.sum(dim=1, keepdim=True) / 255
        # abs_crop = observation[:, 84:94, 18:25:2, 2].reshape(batch_size, 10, 4)
        # keeping 3 channels to start and taking mean across them
        observation = observation.permute(0, 2, 3, 1)  # Change shape to [batch_size, height, width, channels]
        abs_crop = observation[:, 84:94, 18:25:2, 2]  # Extract the third channel
        # abs_crop = observation[:, 84:94, 18:25:2].reshape(batch_size, 10, 4, 3)

        # abs_crop = observation[:, 84:94, 18:25:2].mean(dim=1).reshape(batch_size, 10, 4)

        abs_sensors = abs_crop.sum(dim=1) / 255
        steer_crop = observation[:, 88, 38:58, 1].reshape(batch_size, -1)
        steering = steer_crop.sum(dim=1, keepdim=True)
        gyro_crop = observation[:, 88, 58:86, 0].reshape(batch_size, -1)
        gyroscope = gyro_crop.sum(dim=1, keepdim=True)
        return speed, abs_sensors.reshape(batch_size, 4), steering, gyroscope

class ClassificationNetwork(torch.nn.Module):
    def __init__(self):
        """
        1.1 d)
        Implementation of the network layers. The image size of the input
        observations is 96x96 pixels.
        """
        super().__init__()
        gpu = torch.device('cpu')
        
        # convolutional layers
        
        # Input:
        # Size: (batch_size, 3, 96, 96)
        # After conv1:
        # Kernel size: 3x3, Stride: 2
        # Output size: (batch_size, 32, (96-3)/2 + 1, (96-3)/2 + 1) = (batch_size, 32, 47, 47)
        # After conv2:
        # Kernel size: 3x3, Stride: 2
        # Output size: (batch_size, 64, (47-3)/2 + 1, (47-3)/2 + 1) = (batch_size, 64, 23, 23)
        # After conv3:
        # Kernel size: 3x3, Stride: 2
        # Output size: (batch_size, 128, (23-3)/2 + 1, (23-3)/2 + 1) = (batch_size, 128, 11, 11)
        
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=2)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=2)
        
        # dropout layer for regularization
        # self.dropout = torch.nn.Dropout(0.5)  # Dropout layer with 50% drop probability

        # fully connected layers
        #self.fc1 = torch.nn.Linear(128 * 11 * 11, 128)
        
        # Modify the first fully connected layer to accept the additional sensor inputs
        # 4 additional inputs for speed, steering, gyroscope, and 4 for abs_sensors
        self.fc1 = torch.nn.Linear(128 * 11 * 11 + 4 + 1 + 1 + 1, 128)
        
        self.fc2 = torch.nn.Linear(128, 64)
        #self.fc3 = torch.nn.Linear(64, 9)
        self.fc3 = torch.nn.Linear(64, 5)  # Reducing number of classes to 5



    def forward(self, observation):
        """
        1.1 e)
        The forward pass of the network. Returns the prediction for the given
        input observation.
        observation:   torch.Tensor of size (batch_size, 96, 96, 3)
        return         torch.Tensor of size (batch_size, number_of_classes)
        """
        # Extract sensor values
        speed, abs_sensors, steering, gyroscope = self.extract_sensor_values(observation, observation.shape[0])
        
        
        # convolutional layers
        x = F.relu(self.conv1(observation))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # x = self.dropout(x)  # Apply dropout
        
        # print(x.shape)
        # fully connected layers
        x = x.reshape(-1, 128 * 11 * 11)
        
        # concatonate extra sensor values into the fully connected layer
        x = torch.cat((x, speed, abs_sensors, steering, gyroscope), dim=1)
        
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        # softmax to get probabilities over action-classes
        x = F.softmax(x, dim=1)
        
        return x

    def actions_to_classes(self, actions):
        """
        1.1 c)
        For a given set of actions map every action to its corresponding
        action-class representation. Assume there are number_of_classes
        different classes, then every action is represented by a
        number_of_classes-dim vector which has exactly one non-zero entry
        (one-hot encoding). That index corresponds to the class number.
        actions:        python list of N torch.Tensors of size 3
        return          python list of N torch.Tensors of size number_of_classes
        """
        number_of_classes = 5
        possible_actions = []
        
        for action in actions:
            steer = action[0]
            gas = action[1]
            brake = action[2]
            one_hot = torch.zeros(number_of_classes)
            
            # method below copied over from GPT-4 so I don't have to type as much
            # Define the classes based on the action values
            # if steer < 0:  # Steer left
            #     if gas > 0:
            #         one_hot[4] = 1  # Steer left and gas
            #     elif brake > 0:
            #         one_hot[6] = 1  # Steer left and brake
            #     else:
            #         one_hot[0] = 1  # Steer left
            # elif steer > 0:  # Steer right
            #     if gas > 0:
            #         one_hot[5] = 1  # Steer right and gas
            #     elif brake > 0:
            #         one_hot[7] = 1  # Steer right and brake
            #     else:
            #         one_hot[1] = 1  # Steer right
            # else:  # No steering
            #     if gas > 0:
            #         one_hot[2] = 1  # Gas
            #     elif brake > 0:
            #         one_hot[3] = 1  # Brake
            #     else:
            #         one_hot[8] = 1  # No action
            
            if steer < 0:
                one_hot[0] = 1  # Steer left
            elif steer > 0:
                one_hot[1] = 1  # Steer right
            elif gas > 0:
                one_hot[2] = 1  # Gas
            elif brake > 0:
                one_hot[3] = 1  # Brake
            else:
                one_hot[4] = 1  # No action
            
            # Lowering options for vehicle control
            
                    
            possible_actions.append(one_hot)
                   
        return possible_actions

    def scores_to_action(self, scores):
        """
        1.1 c)
        Maps the scores predicted by the network to an action-class and returns
        the corresponding action [steer, gas, brake].
        scores:         python list of torch.Tensors of size number_of_classes
        return          (float, float, float)
        """
        # Map the highest score to its corresponding action (Dictionary idea taken from GPT-4)
        # class_idx = torch.argmax(scores)
        class_idx = torch.argmax(scores).item()
        # action_map = {
        #     0: (-1, 0, 0),  # Steer left
        #     1: (1, 0, 0),   # Steer right
        #     2: (0, 1, 0),   # Gas
        #     3: (0, 0, 1),   # Brake
        #     4: (-1, 1, 0),  # Steer left and gas
        #     5: (1, 1, 0),   # Steer right and gas
        #     6: (-1, 0, 1),  # Steer left and brake
        #     7: (1, 0, 1),   # Steer right and brake
        #     8: (0, 0, 0)    # No action
        # }
        action_map = {
            0: (-1, 0, 0),  # Steer left
            1: (1, 0, 0),   # Steer right
            2: (0, 1, 0),   # Gas
            3: (0, 0, 1),   # Brake
            4: (0, 0, 0)    # No action
        }
        
        return action_map[class_idx]

    def extract_sensor_values(self, observation, batch_size):
        """
        observation:    python list of batch_size many torch.Tensors of size
                        (96, 96, 3)
        batch_size:     int
        return          torch.Tensors of size (batch_size, 1),
                        torch.Tensors of size (batch_size, 4),
                        torch.Tensors of size (batch_size, 1),
                        torch.Tensors of size (batch_size, 1)
        """
        # print("Observation shape:", observation.shape)
        # sliced_tensor = observation[:, 84:94, 18:25:2, 2]
        # print("Sliced tensor shape:", sliced_tensor.shape)

        speed_crop = observation[:, 84:94, 12, 0].reshape(batch_size, -1)
        speed = speed_crop.sum(dim=1, keepdim=True) / 255
        # abs_crop = observation[:, 84:94, 18:25:2, 2].reshape(batch_size, 10, 4)
        # keeping 3 channels to start and taking mean across them
        observation = observation.permute(0, 2, 3, 1)  # Change shape to [batch_size, height, width, channels]
        abs_crop = observation[:, 84:94, 18:25:2, 2]  # Extract the third channel
        # abs_crop = observation[:, 84:94, 18:25:2].reshape(batch_size, 10, 4, 3)

        # abs_crop = observation[:, 84:94, 18:25:2].mean(dim=1).reshape(batch_size, 10, 4)

        abs_sensors = abs_crop.sum(dim=1) / 255
        steer_crop = observation[:, 88, 38:58, 1].reshape(batch_size, -1)
        steering = steer_crop.sum(dim=1, keepdim=True)
        gyro_crop = observation[:, 88, 58:86, 0].reshape(batch_size, -1)
        gyroscope = gyro_crop.sum(dim=1, keepdim=True)
        return speed, abs_sensors.reshape(batch_size, 4), steering, gyroscope
