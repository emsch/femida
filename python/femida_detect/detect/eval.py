import torch
from .model import select


def load(path, device='cpu'):
    model = torch.load(path, map_location=device)
    net = select[model['meta']['v']](**model['meta']['init_params'])
    net.load_state_dict(model['model'])
    net = net.eval().to(device)
    for param in net.parameters():
        param.requires_grad = False

    def predict(cropped):
        return (net(torch.from_numpy(
            cropped.get_rectangles_array(model['meta']['init_params']['input_size'])
        ).to(device)) < .5).cpu().numpy()
    return predict
