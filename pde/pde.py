import torch


class PDE:
    @staticmethod
    def diff_x(x, u):
        return torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]

    @staticmethod
    def diff_xx(x, u):
        return torch.autograd.grad(PDE.diff_x(x, u), x, torch.ones_like(u), create_graph=True)[0]

    @staticmethod
    def diff_xxx(x, u):
        return torch.autograd.grad(PDE.diff_xx(x, u), x, torch.ones_like(u), create_graph=True)[0]

    @staticmethod
    def diff_xxxx(x, u):
        return torch.autograd.grad(PDE.diff_xxx(x, u), x, torch.ones_like(u), create_graph=True)[0]