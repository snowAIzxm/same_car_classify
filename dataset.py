# demo
#  car_id_image_feature
import random

from typing import List

from bean import CarData

from torch.utils.data import Dataset
class CustomerDataset(Dataset):
    def __init__(self, car_data_list: List[CarData]):
        self.data = car_data_list
        self.size = len(car_data_list)
        self.idx_list = list(range(self.size))
        self.same_ratio = 0.3

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = random.choice(self.data[idx].feature_list)

        if random.random() < self.same_ratio:
            other_index = random.choice(self.idx_list)
            if other_index == idx:
                is_same = True
            else:
                is_same = False
            y = random.choice(self.data[other_index].feature_list)
        else:
            is_same = True
            y = random.choice(self.data[idx].feature_list)
        return x, y, is_same
