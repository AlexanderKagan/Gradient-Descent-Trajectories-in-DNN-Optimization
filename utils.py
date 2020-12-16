def tensor_list_prod(tensor_list):
    product = tensor_list[0].T
    for i in range(1, len(tensor_list)):
        product = product @ tensor_list[i].T
    return product