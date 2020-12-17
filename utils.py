import torch
from torch.distributions import MultivariateNormal
from torch.utils.data import Dataset


def tensor_list_prod(tensor_list):
    product = tensor_list[0].T
    for i in range(1, len(tensor_list)):
        product = product @ tensor_list[i].T
    return product

def make_random_affine_dependency_dataset(num_points, x_size, y_size, device):

    R = torch.randn(size=(x_size, y_size))
    
    mn = MultivariateNormal(torch.zeros(x_size), torch.eye(x_size))
    x = mn.sample((num_points,))

    # x = torch.linspace(-10, 10, num_points).unsqueeze(1)
    y = (x @ R).to(device)
    x = x.to(device)
    return (list(zip(x, y)), R)

class NoisyDataset(Dataset):

    def __init__(self, which, dataset, alpha):
        self.alpha = alpha
        self.dataset = dataset
        assert which in ['x', 'y']
        self.is_x = True if which == 'x' else False
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, i):
        x, y = self.dataset[i]
        if self.is_x:
            return x + torch.randn_like(x) * self.alpha, y
        return x, y + torch.randn_like(y) * self.alpha
