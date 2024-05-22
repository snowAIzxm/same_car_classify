import torch

from dataset import CustomerDataset
from model import SameCarModel
from torch.utils.data import DataLoader

class ModelTrainer:
    def __init__(self):
        self.model = SameCarModel()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

    def train(self, train_data_list, test_data_list):
        train_dataset = CustomerDataset(train_data_list)
        test_dataset = CustomerDataset(test_data_list)
        train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        for epoch in range(10):
            for x, y, is_same in train_dataloader:
                x, y, is_same = x.to(self.device), y.to(self.device), is_same.to(self.device)
                optimizer.zero_grad()
                out = self.model(x, y)
                loss = criterion(out, is_same)
                loss.backward()
                optimizer.step()
            self.model.eval()
            with torch.no_grad():
                for x, y, is_same in test_dataloader:
                    x, y, is_same = x.to(self.device), y.to(self.device), is_same.to(self.device)
                    out = self.model(x, y)
                    loss = criterion(out, is_same)
                    print(f"epoch:{epoch},loss:{loss.item()}")
            self.model.train()
