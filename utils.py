import torch
from torch.distributions import MultivariateNormal


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
