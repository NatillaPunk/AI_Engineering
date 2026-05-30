import torch
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

# from pathlib import Path
# import os

torch.manual_seed(42)

#Dataset configurations ------------------------------------------------------------------------------------------------------------------------
DataSetPath = "Datasets\\Superstore Sales\\sample_-_superstore.csv"
df = pd.read_csv(DataSetPath)

#Set Targets, Features and Device:
df["Profitable"] = (df["Profit"] > 0).astype(int) #Since we miss a clear target column, we set it in this line of code
Features = df.drop(columns=["Profitable","Profit"]) #Drops the column specified in flag value and returns the rest of them as input features.
FeaturesNum = pd.get_dummies(Features,drop_first=True) #Converts each unique value into a column and assing 1 or 0 (Drop_first)
FeaturesNum = FeaturesNum.astype("float32")

Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# print(df[Flag])
# print(len(Features.columns))
# print(len(FeaturesNum.columns))
# print(Device)

# print(FeaturesNum.select_dtypes(include=["object"]).columns)
# print(FeaturesNum.dtypes)
# print("Testing Target Distribution:")
# print(df["Profitable"].value_counts().to_frame("count").assign(pct=lambda x: (x["count"] / x["count"].sum() * 100).round(2)).assign(test="Profit"))


#Tensor Creation ------------------------------------------------------------------------------------------------------------------------
# For X we give the values of the feature columns and convert them to numpyarray and tensor

# X_test = torch.from_numpy(FeaturesNum.values).float().to(Device) # Method 1 to create tensor from Dataframes

Xtensor = torch.tensor(FeaturesNum.values, dtype=torch.float32).to(Device) #Method 2 to create tensor from dataframes
Ytensor = torch.tensor(df["Profitable"].values, dtype=torch.float32).view(-1,1).to(Device) #View is needed to set the correct size of the tensor expected in further methods from (batch_size,) to (batch_size,1)
                       
# print(Xtensor)


Yarray = Ytensor.cpu().numpy().ravel()

# Creation of dataloader and datatensor  ------------------------------------------------------------------------------------------------------------------------

DfDataset = TensorDataset(Xtensor, Ytensor)
DfLoader = DataLoader(DfDataset, batch_size=32, shuffle=True)


# Creation of MLP  ------------------------------------------------------------------------------------------------------------------------


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()


    def forward(self,x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x



# Set of Sizes  ------------------------------------------------------------------------------------------------------------------------

input_features = FeaturesNum.shape[1]
hidden_neurons = 360
output_classes = 1


MLPModel = MLP(input_features, hidden_neurons, output_classes)
MLPModel.to(Device)


#Loss Function and Optimizer ------------------------------------------------------------------------------------------------------------------------

Loss_fn = nn.BCEWithLogitsLoss()
Optimizer = torch.optim.Adam(MLPModel.parameters(),lr=0.001)






#Training loop ------------------------------------------------------------------------------------------------------------------------

# epoch_loss: Training loss per epoch.
# num_batches: Number of batches per epoch.
# correct: Number of correct predictions.
# total: Total number of predictions.



MLPModel.train()

num_epochs = 10

for Epoch in range(num_epochs):
    Epoch_Loss = 0
    Num_batches = 0
    AccCorrect = 0
    Total = 0

    for X,Y in DfLoader:
        #Gets the logits by passing X through the forward pass
        Logits = MLPModel(X)

        # Computes loss comparing results given by the logits with the real targets
        Loss = Loss_fn(Logits,Y)

        # Backpropagation
        Optimizer.zero_grad()
        Loss.backward()

        #Optimizer weight update
        Optimizer.step()

        #Loss tracking
        Epoch_Loss += Loss.item()
        Num_batches += 1

        #Translate logits to probabilities
        Probs = torch.sigmoid(Logits)
        Preds = (Probs >= 0.5).float()

        #Accuracy
        AccCorrect += (Preds == Y).sum().item()
        Total += Y.size(0)

        #Averages
        AvgLoss = Epoch_Loss / Num_batches
        AvgAccuracy = AccCorrect / Total

    print(f'Epoch # [{Epoch + 1}/{num_epochs}]')
    print(f'BCE Loss {AvgLoss:4f}, Accuracy {AvgAccuracy:4f}')





#Save the trained model:

torch.save(MLPModel.state_dict(),"MLP_Weights.pt")





