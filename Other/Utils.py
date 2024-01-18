from torchvision import transforms


def transformation(m, n):
    transform = transforms.Compose([
        transforms.Resize((m, n)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    return transform
