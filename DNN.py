import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
from autocoder import Autocoder

class loadBankDataset(Dataset):
    def __init__(self, file_name):
        self.data = pd.read_csv(file_name)

        # self.data['BalanceSalaryRatio'] = self.data.Balance / self.data.EstimatedSalary
        # unwanted_cols = ['CustomerId', 'Geography', 'Tenure', 'Balance', 'HasCrCard', 'EstimatedSalary']
        unwanted_cols = ['CustomerId']
        self.data = self.data.drop(unwanted_cols, axis=1)

        # Length of data
        self.len = len(self.data)

        # Create CreditLevel list for one hot encoding later
        self.credit_ls = self.data['CreditLevel'].unique()

        # Create Geography dict to map country to number
        unique_geo = self.data['Geography'].unique()
        self.geo_dict = dict()
        for idx, geo in enumerate(unique_geo):
            self.geo_dict[geo] = idx

        # Convert country to number
        self.data['Geography'] = self.data['Geography'].apply(lambda x: self.geo_dict[x])

        # Normalzie Balance and EstimatedSalary
        self.data['Balance'] = (self.data['Balance'] - self.data['Balance'].min()) \
                               / (self.data['Balance'].max() - self.data['Balance'].min())
        # self.data['BalanceSalaryRatio'] = (self.data['BalanceSalaryRatio'] - self.data['BalanceSalaryRatio'].min()) \
        #                        / (self.data['BalanceSalaryRatio'].max() - self.data['BalanceSalaryRatio'].min())

        self.data['EstimatedSalary'] = (self.data['EstimatedSalary'] - self.data['EstimatedSalary'].min()) \
                                       / (self.data['EstimatedSalary'].max() - self.data['EstimatedSalary'].min())



    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # Define CreditLevel as label
        label = self.data.iloc[idx, 8]  ##
        label = label - 1
        # Retreive attributes
        attribute = self.data.iloc[idx, 0:8].to_numpy()  ##
        return attribute, label


# class NN(nn.Module):
#     def __init__(self, input_size, num_class):
#         super(NN, self).__init__()
#         self.fc1 = nn.Linear(input_size, 50)
#         self.fc2 = nn.Linear(50, 40)
#         self.fc3 = nn.Linear(40, 25)
#         self.fc4 = nn.Linear(25, num_class)
#
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = self.fc4(x)
#         return x

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

bank_dataset = loadBankDataset("BankChurners.csv")
data_len = bank_dataset.len

# Define test data ratio
test_len = int(data_len * 0.3)
train_len = data_len - test_len

# Split the data and set random seed to 123 for reproduceable results
train_dataset, test_dataset = random_split(bank_dataset,
                                           [train_len, test_len],
                                           generator=torch.Generator().manual_seed(24))

# Hyparameters
learning_rate = 0.01
num_epochs = 10
batch_size = 100

# Initialzie neural network
input_size = 8  ##
num_class = len(bank_dataset.credit_ls)
# model = NN(input_size, num_class).double().to(device)
model = Autocoder()
model.to(device).double()
#%%
# Initialize loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

# Initialzie dataloader
train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

# Train Network
total_batch = len(train_dataloader)
for epoch in range(num_epochs):
    for idx, (attribute, label) in enumerate(train_dataloader):

        attribute = attribute.to(device)
        label = label.to(device)
        # Forward propagation
        _,outputs = model(attribute)
        loss = criterion(outputs, attribute)

        # Backward propagation
        optimizer.zero_grad()
        loss.backward()

        # Gradient descent update
        optimizer.step()

        # Output training status
        print('Epoch [{} / {}], Batch [{} / {}], Loss : {:.4f}'
              .format(epoch + 1, num_epochs, idx + 1, total_batch, loss.item()))
#%%
# Test for accuracy
# def check_accuracy(loader, model):
#     num_correct = 0
#     num_samples = 0
#     model.eval() # Tell torch this is evaluation mode
#
#     with torch.no_grad():
#         for attribute, label in loader:
#             _,logits = model.encoder(attribute)
#             # _, prediction = logits.max(1) # compute max at dimension 1 and retreive indices
#             # print(prediction)
#             num_correct += (logits.data == attribute.data).sum()
#             num_samples += logits.size(0)
#     print(f"{num_correct} / {num_samples} correct with accuracy = {float(num_correct)/float(num_samples)*100:.2f}%")
#
# check_accuracy(train_dataloader, model)
# check_accuracy(test_dataloader, model)

model_file = 'model.pth'
torch.save(model, model_file)
print(f'Model saved to {model_file}.')


