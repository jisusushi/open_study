import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class TrainTestDataset(Dataset):
    def __init__(self, file, transform=None):
        self.data= pd.read_csv(file)
        self.data= self.data.iloc[:, 1:]
        self.transform= transform

        if transform is not None:
            self.data= self.transform(np.array(self.data))

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, ind):
        user_vector= self.data.data[0][ind]

        return user_vector
