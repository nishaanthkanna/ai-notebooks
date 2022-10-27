from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms


# Dataloader functions


def get_dataloader(dataset="cifar", batch_size=64):
    
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if dataset == "cifar":
        cifar_training_data = datasets.CIFAR10(
            root="data",
            train=True,
            download=True,
            transform=transform
        )

        cifar_testing_data = datasets.CIFAR10(
            root="data",
            train=False,
            download=True,
            transform=transform
        )

        train_dl = DataLoader(cifar_training_data, batch_size=batch_size)
        test_dl = DataLoader(cifar_testing_data, batch_size=batch_size)

        return train_dl, test_dl
