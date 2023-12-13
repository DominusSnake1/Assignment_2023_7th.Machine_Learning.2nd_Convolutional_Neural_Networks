from torchvision import transforms


class Transformations:
    def __init__(self, m, n):
        self.m = m
        self.n = n
        self.transform = transforms.Compose([
            transforms.Resize((self.m, self.n)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
