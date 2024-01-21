from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torchvision import transforms


def transformation(m, n):
    transform = transforms.Compose([
        transforms.Resize((m, n)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    return transform


def model_statistics(predictions, true_labels):
    accuracy = accuracy_score(true_labels, predictions)
    class_report = classification_report(true_labels, predictions)
    confusion_mat = confusion_matrix(true_labels, predictions)

    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{class_report}")
    print(f"Confusion Matrix:\n{confusion_mat}")
