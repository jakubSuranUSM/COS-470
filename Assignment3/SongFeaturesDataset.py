import torch
from torch.utils.data import Dataset
from torch.nn.functional import one_hot


class SongFeaturesDataset(Dataset):
    def __init__(self, df, label):
        self.data = df

        self.features = torch.Tensor(df.loc[:, df.columns != label].values)
        self.labels = one_hot(torch.Tensor(df[label].values).to(torch.int64)).to(torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.features[idx]
        label = self.labels[idx]

        return features, label
