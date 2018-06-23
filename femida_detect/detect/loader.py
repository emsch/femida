from torchvision import datasets, transforms
import random
import numpy as np
import torch.utils
from imgaug import augmenters

__all__ = [
    'dataset',
    'data_loader'
]


class RandomAugAndShuffle(transforms.RandomApply):
    def __call__(self, img):
        order = list(range(len(self.transforms)))
        random.shuffle(order)
        for i in order:
            if self.p < random.random():
                img = self.transforms[i](img)
        return img


def dataset(root, augment=False, shape=28, paug=.5):
    if isinstance(shape, int):
        shape = (shape, shape)
    h, w = shape
    if augment:
        data_transform = transforms.Compose([
            transforms.RandomChoice([
                transforms.Compose([
                    transforms.Resize((int(h*1.15), int(h*1.15))),
                    transforms.RandomCrop(shape),
                ]),
                transforms.Resize(shape),
                transforms.Compose([
                    transforms.Resize((int(h*1.1), int(h*1.1))),
                    transforms.Pad((int(h*.05), int(h*.05)), padding_mode='reflect'),
                    transforms.RandomCrop(shape),
                ])
            ]),
            RandomAugAndShuffle([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip()
            ]),
            np.asarray,
            RandomAugAndShuffle([
                augmenters.AddToHueAndSaturation(value=(-5, 5)).augment_image,
                augmenters.AdditiveGaussianNoise(scale=.05*255).augment_image,
                augmenters.SaltAndPepper(0.01).augment_image,
                augmenters.ContrastNormalization([0.8, 1.2]).augment_image,
                augmenters.Affine(deterministic=False, mode='reflect', rotate=[-5, 5]).augment_image
            ], p=1-paug**(1/5)),
            augmenters.Sometimes(0.5, augmenters.Grayscale()).augment_image,
            transforms.ToTensor(),
        ])
    else:
        data_transform = transforms.Compose([
            transforms.Resize(shape),
            transforms.ToTensor(),
        ])
    return datasets.ImageFolder(root=root, transform=data_transform)


def data_loader(root, batch_size=64, shuffle=True, augment=False, shape=28, paug=.5, workers=1, ):
    dataset_loader = torch.utils.data.DataLoader(
        dataset(root, augment=augment, shape=shape, paug=paug),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers
    )
    return dataset_loader


if __name__ == '__main__':
    img = dataset('../../data/train', augment=True)[0]

