from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from loader.custom_dataset import NepaliDigitDataset


BATCH_SIZE = 64
def dataloader():
    train_path = 'data/train/'
    test_path = 'data/test/'

    transform = transforms.Compose([

        transforms.ToTensor(),
        transforms.Resize((32, 32)),
        transforms.Grayscale(),
        transforms.Normalize(mean=0, std=1)
    ])

    train_dataset = NepaliDigitDataset(train_path, transform=transform)
    test_dataset = NepaliDigitDataset(test_path, transform=transform)

    train_size = int(0.8 * len(train_dataset))
    val_size = int(0.2 * len(train_dataset))

    train_set, val_set = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    return train_loader, val_loader, test_loader