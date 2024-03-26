import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
from torcheval.metrics.functional import multiclass_f1_score

num_features = 20
num_genres = 6


class GenreClassifier(nn.Module):
    def __init__(self, input_size=num_features, output_size=num_genres):
        super(GenreClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, output_size),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        out = self.model(x)
        return out


def get_train_val_dataloaders(dataset, batch_size, val_ratio=0.2, shuffle=False):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=42)

    train_index, val_index = next(sss.split(dataset.features, dataset.data["Genre"]))

    train_sampler = torch.utils.data.SubsetRandomSampler(train_index)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_index)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, shuffle=shuffle)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, shuffle=shuffle)

    return train_loader, val_loader


def train_one_epoch(model, train_data, loss_func, optimizer):
    loss_values = []
    for X, y in train_data:
        # Forward pass
        out = model(X)
        loss = loss_func(out, y)
        loss_values.append(loss.item())

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = sum(loss_values) / len(loss_values)
    return avg_loss


def validate(model, val_data, loss_func):
    loss_values = []
    with torch.no_grad():
        for X, y in val_data:
            out = model(X)
            loss = loss_func(out, y)
            loss_values.append(loss)

    avg_loss = sum(loss_values) / len(loss_values)
    return avg_loss


def train(model, train_data, val_data, loss_func, optimizer, num_epochs):
    loss_train = []
    loss_valid = []

    for epoch in tqdm(range(num_epochs)):
        avg_loss = train_one_epoch(model, train_data, loss_func, optimizer)
        loss_train.append(avg_loss)

        avg_loss_valid = validate(model, val_data, loss_func)
        loss_valid.append(avg_loss_valid)

    return loss_train, loss_valid


def evaluate(model, X, y):
    out = model(X)
    y = torch.argmax(y, dim=1)
    return (multiclass_f1_score(out, y, num_classes=num_genres, average='macro'),
            multiclass_f1_score(out, y, num_classes=num_genres, average=None))
