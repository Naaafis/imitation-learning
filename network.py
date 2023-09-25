import torch
import torch.nn.functional as F

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
        
        # self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=2)
        # self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=2)
        # self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=2)
        
        # convolutional layers
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=2)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.conv4 = torch.nn.Conv2d(128, 256, kernel_size=3, stride=2)  # New convolutional layer

        # dropout layer for regularization
        self.dropout = torch.nn.Dropout(0.5)  # Dropout layer with 50% drop probability

        
        # fully connected layers
        self.fc1 = torch.nn.Linear(256 * 5 * 5, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 9)
        


    def forward(self, observation):
        """
        1.1 e)
        The forward pass of the network. Returns the prediction for the given
        input observation.
        observation:   torch.Tensor of size (batch_size, 96, 96, 3)
        return         torch.Tensor of size (batch_size, number_of_classes)
        """
        
        # # convolutional layers
        # x = F.relu(self.conv1(observation))
        # x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        
        # convolutional layers
        x = F.relu(self.conv1(observation))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))  # New convolutional layer

        x = self.dropout(x)  # Apply dropout
        
        # print(x.shape)
        # fully connected layers
        x = x.reshape(-1, 256 * 5 * 5)
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
        number_of_classes = 9
        possible_actions = []
        
        for action in actions:
            steer = action[0]
            gas = action[1]
            brake = action[2]
            one_hot = torch.zeros(number_of_classes)
            
            # method below copied over from GPT-4 so I don't have to type as much
            # Define the classes based on the action values
            if steer < 0:  # Steer left
                if gas > 0:
                    one_hot[4] = 1  # Steer left and gas
                elif brake > 0:
                    one_hot[6] = 1  # Steer left and brake
                else:
                    one_hot[0] = 1  # Steer left
            elif steer > 0:  # Steer right
                if gas > 0:
                    one_hot[5] = 1  # Steer right and gas
                elif brake > 0:
                    one_hot[7] = 1  # Steer right and brake
                else:
                    one_hot[1] = 1  # Steer right
            else:  # No steering
                if gas > 0:
                    one_hot[2] = 1  # Gas
                elif brake > 0:
                    one_hot[3] = 1  # Brake
                else:
                    one_hot[8] = 1  # No action
                    
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
        action_map = {
            0: (-1, 0, 0),  # Steer left
            1: (1, 0, 0),   # Steer right
            2: (0, 1, 0),   # Gas
            3: (0, 0, 1),   # Brake
            4: (-1, 1, 0),  # Steer left and gas
            5: (1, 1, 0),   # Steer right and gas
            6: (-1, 0, 1),  # Steer left and brake
            7: (1, 0, 1),   # Steer right and brake
            8: (0, 0, 0)    # No action
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
        speed_crop = observation[:, 84:94, 12, 0].reshape(batch_size, -1)
        speed = speed_crop.sum(dim=1, keepdim=True) / 255
        abs_crop = observation[:, 84:94, 18:25:2, 2].reshape(batch_size, 10, 4)
        abs_sensors = abs_crop.sum(dim=1) / 255
        steer_crop = observation[:, 88, 38:58, 1].reshape(batch_size, -1)
        steering = steer_crop.sum(dim=1, keepdim=True)
        gyro_crop = observation[:, 88, 58:86, 0].reshape(batch_size, -1)
        gyroscope = gyro_crop.sum(dim=1, keepdim=True)
        return speed, abs_sensors.reshape(batch_size, 4), steering, gyroscope
