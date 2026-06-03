import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import sys
import matplotlib.pyplot as plt
from collections import Counter



print(sys.executable)

#Dataset configurations ------------------------------------------------------------------------------------------------------------------------
DataSetPath = "Datasets\\Superstore Sales\\sample_-_superstore.csv"
df = pd.read_csv(DataSetPath)
torch.manual_seed(42)

#Set Inference Enviroment: 
# python -m venv sklearn-env
# sklearn-env\Scripts\activate  # activate
# pip install -U scikit-learn


#Set Targets, Features and Device:
df["Profitable"] = (df["Profit"] > 0).astype(int) #Since we miss a clear target column, we set it in this line of code
Features = df.drop(columns=["Profitable","Profit"]) #Drops the column specified in flag value and returns the rest of them as input features.
FeaturesNum = pd.get_dummies(Features,drop_first=True) #Converts each unique value into a column and assing 1 or 0 (Drop_first)
FeaturesNum = FeaturesNum.astype("float32")

Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Tensor Creation ------------------------------------------------------------------------------------------------------------------------

Xtensor = torch.tensor(FeaturesNum.values, dtype=torch.float32).to(Device) #Method 2 to create tensor from dataframes
Ytensor = torch.tensor(df["Profitable"].values, dtype=torch.float32).view(-1,1).to(Device) #View is needed to set the correct size of the tensor expected in further methods from (batch_size,) to (batch_size,1)
y_true = Ytensor.cpu().numpy().ravel()

# Creation of dataloader and datatensor  ------------------------------------------------------------------------------------------------------------------------

DfDataset = TensorDataset(Xtensor, Ytensor)
DfLoader = DataLoader(DfDataset, batch_size=32, shuffle=True)


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

# Load parameters
StateDict = torch.load(
    "MLP_Weights.pt",
    map_location=Device,
    weights_only=True
)


# Set of Sizes  ------------------------------------------------------------------------------------------------------------------------

InputFeatures = FeaturesNum.shape[1]
HiddenNeurons = 360
OutputClasses = 1


MLPModel = MLP(InputFeatures, HiddenNeurons, OutputClasses)
MLPModel.load_state_dict(StateDict)
MLPModel.to(Device)


print("hola")
#Prediction and Inference ------------------------------------------------------------------------------------------------------------------------

def PredictGenerator(MLPModel,DfLoader, Device):
    MLPModel.eval()
    LogitsList = []
    PredsList = []

    
    print("hola2")

    with torch.no_grad():
        for X_batch,Y_batch in DfLoader: #We Get the X (Features) and Y (Flags)
            X_batch,Y_batch = X_batch.to(Device),Y_batch.to(Device)
            
            #Gets the logits
            Logits = MLPModel(X_batch)

            #Gets the probabilities
            Probs = torch.sigmoid(Logits)
            Preds = (Probs >= 0.5).float()

            LogitsList.append(Logits.cpu().numpy())
            PredsList.append(Preds.cpu().numpy())

    y_logits = np.concatenate(LogitsList).ravel()
    y_pred = np.concatenate(PredsList).ravel()

    return y_logits,y_pred




y_logits_relu, y_pred_relu = PredictGenerator(MLPModel, DfLoader, Device)

test_accuracy = accuracy_score(y_true, y_pred_relu)
report = classification_report(y_true, y_pred_relu, target_names=['No Profitable', 'Profitable']) 

print(f"Accuracy: {test_accuracy:.4f}")
print(report)

# Prediction counts
class_map = {0: "Not Canceled", 1: "Canceled"}
counts = {class_map[k]: v for k, v in Counter(y_pred_relu).items()}
print(counts)

# Visualize counts
plt.bar(counts.keys(), counts.values())
plt.title("Prediction Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()








