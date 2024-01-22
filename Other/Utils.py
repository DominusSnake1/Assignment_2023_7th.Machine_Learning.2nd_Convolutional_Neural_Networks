from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torchvision import transforms


def transformations(m=0, n=0):
    data_transform = {
        'standard': transforms.Compose([
            transforms.Resize((m, n)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    return data_transform


def model_statistics(predictions, true_labels):
    accuracy = accuracy_score(true_labels, predictions)
    class_report = classification_report(true_labels, predictions)
    confusion_mat = confusion_matrix(true_labels, predictions)

    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{class_report}")
    print(f"Confusion Matrix:\n{confusion_mat}")
