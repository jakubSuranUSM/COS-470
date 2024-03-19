import torch
from torch import nn

num_features = 19
num_genres = 6


class GenreClassifier(nn.Module):
    def __init__(self):
        super(GenreClassifier, self).__init__()
        self.input_size = num_features
        self.fc1 = nn.Linear(num_features, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, num_genres)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x


#
# model = GenreClassifier()
#
# # Define loss function and optimizer
# criterion = nn.CrossEntropyLoss()  # For multi-class classification
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Learning rate adjustment might be needed
#
# # Sample training data (replace with your actual data)
# # Assuming features is a tensor of size (num_samples, num_features)
# # and labels is a tensor of size (num_samples) with genre indices
#
# # Training loop (replace with your training data and epochs)
# for epoch in range(10):  # Adjust the number of epochs based on your data
#     for features, labels in training_data:
#         # Forward pass
#         outputs = model(features)
#         loss = criterion(outputs, labels)
#
#         # Backward pass and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#     # Print training progress (optional)
#     print(f'Epoch: {epoch+1}/{10}, Loss: {loss.item():.4f}')
#
# # Prediction (replace with your new lyrics features)
# new_lyrics_features = ...  # Prepare your new lyrics features as a tensor
#
# # Get predictions
# predictions = model(new_lyrics_features)
#
# # Get the genre with the highest probability
# _, predicted_genre = torch.max(predictions.data, 1)
#
# print(f'Predicted Genre: {predicted_genre.item()}')  # Assuming predicted_genre is a single value tensor
