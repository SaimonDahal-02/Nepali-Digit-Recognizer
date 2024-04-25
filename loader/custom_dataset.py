
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

class NepaliDigitDataset(Dataset):
    def __init__(self, dir, transform):
        self.dir = dir
        self.transform = transform
        self.data = ImageFolder(root=dir, transform=transform)

    def __len__(self,):
        return len(self.data)
    
    def __getitem__(self, index):
        imag, label = self.data[index]
        return imag, label