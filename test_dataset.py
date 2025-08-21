import torch.nn as nn
import torch
import pickle
import numpy as np

with open("C:\\Users\\david\\Documents\\svd_results.pkl", "rb") as f:
    svd_results = pickle.load(f)

print(svd_results.keys())
U_in = svd_results["U_in"]
s_in = svd_results["s_in"]
V_in = svd_results["V_in"]
U_out = svd_results["U_out"]
s_out = svd_results["s_out"]
V_out = svd_results["V_out"]

inputs = U_in @ np.diag(s_in) @ V_in

outputs = U_out @ np.diag(s_out) @ V_out

training_inputs = inputs[:500, :]
training_outputs = outputs[:500, :]
validation_inputs = inputs[500:600, :]
validation_outputs = outputs[500:600, :]
testing_inputs = inputs[600:, :]
testing_outputs = outputs[600:, :]

# Model
class UNET (nn.Module):
    def __init__(self, in_channels):
        super(UNET, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=in_channels,out_channels= in_channels * 2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=in_channels * 2,out_channels=  in_channels * 4, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=in_channels * 4, out_channels= in_channels * 8, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.ConvTranspose1d(in_channels=in_channels * 8,out_channels=  in_channels * 8, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv1d(in_channels=in_channels * 8, out_channels= in_channels * 16, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv1d(in_channels=in_channels * 16, out_channels= in_channels * 32, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv1d(in_channels=in_channels * 32, out_channels= in_channels * 64, kernel_size=3, stride=1, padding=1)
        self.conv8= nn.ConvTranspose1d(in_channels=in_channels * 64, out_channels= in_channels * 64, kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv1d(in_channels=in_channels * 64, out_channels= in_channels * 128, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv1d(in_channels=in_channels * 128, out_channels= in_channels * 256, kernel_size=3, stride=1, padding=1)
        self.conv11 = nn.Conv1d(in_channels=in_channels * 256, out_channels= in_channels * 512, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change shape from (batch_size, seq_len, features) to (batch_size, features, seq_len)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = x.permute(0, 2, 1)  # Change shape back to (batch_size, seq_len, features)
        return x

class Transformer(nn.Module):
    def __init__(self, in_channels, d_model, num_heads, num_encoder_layers, num_classes):
        super(Transformer, self).__init__()
        self.TransformerEncoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads), num_encoder_layers)
        self.regressor = nn.Linear(1536, num_classes)

        self.the_UNET = UNET(in_channels)
    def forward(self, x):
        x = self.TransformerEncoder(x)
        x = x.reshape(1, x.shape[0], x.shape[1])
        x = self.the_UNET(x)
        x = self.regressor(x)
        return x


#Define hyperparameters
epochs = 500
loss_func = nn.MSELoss()
Model = Transformer(in_channels=3, d_model=128, num_heads=1, num_encoder_layers=2, num_classes=1)
learning_rate = 0.0001
optimizer = torch.optim.Adam(Model.parameters(), lr = learning_rate)
# Identifying tracked values

train_loss = []


# Training loop
inputs_tensor = torch.from_numpy(training_inputs).float()
outputs_tensor = torch.from_numpy(training_outputs).float()
validation_inputs_tensor = torch.from_numpy(validation_inputs).float()
validation_outputs_tensor = torch.from_numpy(validation_outputs).float()
testing_inputs_tensor = torch.from_numpy(testing_inputs).float()
testing_outputs_tensor = torch.from_numpy(testing_outputs).float()

for epoch in range(epochs):
    Model.train()
    optimizer.zero_grad()
    
    # Convert training inputs and outputs to tensors
    inputs_tensor = torch.tensor(training_inputs, dtype=torch.float32)
    outputs_tensor = torch.tensor(training_outputs, dtype=torch.float32)
    
    # Forward pass
    predictions = Model(inputs_tensor)
    
    # Compute loss
    loss = loss_func(predictions, outputs_tensor)
    
    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    
    train_loss.append(loss.item())
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")



print("Test Loss: ", loss_func(Model(testing_inputs_tensor), testing_outputs_tensor).item())

