import torch
from .model import select


def load_net(path, device='cpu'):
    model = torch.load(path, map_location=device)
    net = select[model['meta']['v']](**model['meta']['init_params'])
    net.load_state_dict(model['model'])
    net = net.eval().to(device)
    for param in net.parameters():
        param.requires_grad = False
    return net


def wrap(clf, size):
    def predict(cropped):
        return clf(cropped.get_rectangles_array(size))
    return predict


def wrapnet(net):
    device = next(net.parameters()).device

    def clf(imgs):
        return (net(torch.from_numpy(imgs).to(device)) < .5).cpu().numpy()
    return wrap(clf, net.input_size)


def load(path: str, device='cpu'):
    if path.endswith('t7'):
        return wrapnet(load_net(path, device))
    else:
        raise NotImplementedError
