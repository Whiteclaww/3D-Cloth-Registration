from chamferdist import ChamferDistance
import torch
from conversion import tofloat

def dist1():
    chamfer = ChamferDistance()
    lista = [[[1, 1, 1]], [[3, 3, 3]]]
    listb = [[[0, 1, 0], [1, 0, 0], [0, 0, 0]], [[3, 0, 0], [0, 3, 0], [0, 0, 0]]]
    res = []
    for i in range(len(lista)):
        for j in range(len(listb)):
            a = lista[i]
            b = listb[j]
            a = torch.Tensor([a])
            b = torch.Tensor([b])
            result = chamfer.forward(source_cloud=a, target_cloud=b)
            res.append([i, j, result.detach().cpu().item()])
    return result

def dist2():
    chamfer = ChamferDistance()
    lista = [[[1, 1, 1]]]
    listb = [[[1.1, 1.1, 0], [1.1, 0, 0], [0, 1.1, 0]], [[1.1, 1.1, 0], [1.1, 0, 0], [1.1, 1.2, 0]]]
    res = []
    for i in range(len(lista)):
        for j in range(len(listb)):
            a = lista[i]
            b = listb[j]
            a = torch.Tensor([a])
            b = torch.Tensor([b])
            result = chamfer.forward(source_cloud=a, target_cloud=b)
            res.append([i, j, result.detach().cpu().item()])
    return result



def dist3():
    cham = ChamferDistance()
    #torch_garment = torch.Tensor([garment.vertices])
    #torch_smpl = torch.Tensor([smpl.vertices])
    lista = [[[1, 1, 1]]]
    listb = [[[1.1, 1.1, 0], [1.1, 0, 0], [0, 1.1, 0]], [[1.1, 1.1, 0], [1.1, 0, 0], [1.1, 1.4, 0]]]
    res = []
    for i in range(len(lista)):
        for j in range(len(listb)):
            a = lista[i]
            b = listb[j]
            a = torch.Tensor([a])
            b = torch.Tensor([b])
            result = tofloat(cham.forward(source_cloud=a, target_cloud=b))
            res.append([i, j, result])
            