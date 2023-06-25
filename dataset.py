import torchvision 
import torch 
from torch.utils.data import DataLoader 
from torchvision.transforms import Compose, Lambda, ToTensor 

def download_dataset(): 
    mnist = torchvision.datasets.MNIST(root="./data/mnist", download=True)
    idx = 7 
    img, label = mnist[idx]
    tensor = ToTensor()(img)
    print(f"mnist img shape {tensor.shape}")
    
def get_dataloader(batch_size: int): 
    # change input from [0,1] to [-1,1] since we want a normal distribution 
    transforms = Compose([ToTensor(), Lambda(lambda x: (x-0.5)*2)])
    dataset = torchvision.datasets.MNIST(root="./data/mnist", transform=transforms)

    subset_indices = range(1000)
    subset_dataset = torch.utils.data.Subset(dataset, subset_indices)
    
    # return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return DataLoader(subset_dataset, batch_size=batch_size, shuffle=True)
    
def get_img_shape(): 
    return (1,28,28)

if __name__ == "__main__": 
    download_dataset()
    