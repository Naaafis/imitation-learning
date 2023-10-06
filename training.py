import torch
import random
import time
from network import ClassificationNetwork
from network import MultiClassNetwork
from network import RegressionNetwork
from network import GreyNetwork
from network import CutNetwork  
from imitations import load_imitations
import torch.nn.functional as F
import tqdm

def train(data_folder, trained_network_file):
    """
    Function for training the network.
    """
    #infer_action = ClassificationNetwork()
    #infer_action = MultiClassNetwork()
    #infer_action = RegressionNetwork()
    #infer_action = GreyNetwork()
    infer_action = CutNetwork()
    # criterion = torch.nn.BCELoss()  # Binary Cross-Entropy Loss for Multi-Class Classification
    criterion = torch.nn.MSELoss()  # Mean Squared Error Loss for Regression task 
    optimizer = torch.optim.Adam(infer_action.parameters(), lr=1e-3)
    observations, actions = load_imitations(data_folder)
    
    # # Crop the observations for CutNetwork
    # cropped_observations = []
    # for observation in observations:
    #     center_y, center_x = observation.shape[0] // 2, observation.shape[1] // 2
    #     top_left_y, top_left_x = center_y - 30, center_x - 30
    #     bottom_right_y, bottom_right_x = center_y + 30, center_x + 30
    #     cropped_observation = observation[top_left_y:bottom_right_y, top_left_x:bottom_right_x, :]
    #     cropped_observations.append(cropped_observation)

    # observations = [torch.Tensor(observation) for observation in cropped_observations]
    observations = [torch.Tensor(observation) for observation in observations]
    actions = [torch.Tensor(action) for action in actions]

    # permute the data to have observation shape be (batch_size, 3, 96, 96)
    # for observation in observations:
    #     observation = observation.permute(0, 3, 1, 2)

    # batches = [batch for batch in zip(observations, infer_action.actions_to_classes(actions))]
    batches = [batch for batch in zip(observations, actions)] # for regression task we don't need to convert actions to classes
    gpu = torch.device('cpu')

    nr_epochs = 50
    batch_size = 64
    number_of_classes = 3  # needs to be changed
    start_time = time.time()

    for epoch in range(nr_epochs):
        random.shuffle(batches)

        total_loss = 0
        batch_in = []
        batch_gt = []
        
        #for batch_idx, batch in enumerate(batches):
        for batch_idx, batch in tqdm.tqdm(enumerate(batches), total=len(batches)):
            batch_in.append(batch[0])
            #print(batch[1].shape)  # debug
            batch_gt.append(batch[1])
            
            # for tensor in batch_gt:
            #     print(tensor.shape)

            if (batch_idx + 1) % batch_size == 0 or batch_idx == len(batches) - 1:
                batch_in = torch.reshape(torch.cat(batch_in, dim=0),
                                         (-1, 96, 96, 3))
                # batch_in = torch.reshape(torch.cat(batch_in, dim=0),
                #                          (-1, 96, 96, 1))
                # batch_in = torch.reshape(torch.cat(batch_in, dim=0),
                #                          (-1, 60, 60, 3))
                # permute the data to have observation shape be (batch_size, 3, 96, 96)
                batch_in = batch_in.permute(0, 3, 1, 2)
                batch_gt = torch.reshape(torch.cat(batch_gt, dim=0),
                                         (-1, number_of_classes))

                #print("Batch Inputs shape:", batch_in.shape)
                batch_out = infer_action(batch_in)
                
                # Debug: Print the predicted outputs for each class
                # print("Predicted Outputs shape:", batch_out.shape)
                # print("Ground Truth shape:", batch_gt.shape)
                
                # loss = cross_entropy_loss(batch_out, batch_gt) # for single class classification
                loss = criterion(batch_out, batch_gt)  # for multi-class classification and regression

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss

                batch_in = []
                batch_gt = []

        time_per_epoch = (time.time() - start_time) / (epoch + 1)
        time_left = (1.0 * time_per_epoch) * (nr_epochs - 1 - epoch)
        print("Epoch %5d\t[Train]\tloss: %.6f \tETA: +%fs" % (
            epoch + 1, total_loss, time_left))

    torch.save(infer_action, trained_network_file)


def cross_entropy_loss(batch_out, batch_gt):
    """
    Calculates the cross entropy loss between the prediction of the network and
    the ground truth class for one batch.
    batch_out:      torch.Tensor of size (batch_size, number_of_classes)
    batch_gt:       torch.Tensor of size (batch_size, number_of_classes)
    return          float
    """
    # Convert one-hot encoded ground truth to class indices
    batch_gt_indices = torch.argmax(batch_gt, dim=1)
    return F.cross_entropy(batch_out, batch_gt_indices)
    
