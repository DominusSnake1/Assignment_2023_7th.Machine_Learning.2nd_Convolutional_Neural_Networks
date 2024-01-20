from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from Classes.NetTrainAndTest import NetTrainAndTest
from torchsummary import summary
from torchvision import transforms
import torch


def transformation(m, n):
    transform = transforms.Compose([
        transforms.Resize((m, n)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    return transform


def model_train_and_stats(model, train_loader, test_loader, epochs,  learning_rate):
    trainer = NetTrainAndTest(
        model=model,
        trainloader=train_loader,
        testloader=test_loader,
        epochs=epochs,
        optimizer=torch.optim.SGD(model.parameters(), lr=learning_rate),
        loss_fn=torch.nn.CrossEntropyLoss()
    )

    summary(model, (3, 50, 62))

    trainer.train_net()
    predictions, true_labels = trainer.test_net()

    accuracy = accuracy_score(true_labels, predictions)
    class_report = classification_report(true_labels, predictions)
    confusion_mat = confusion_matrix(true_labels, predictions)

    print(f"======[Statistics for {model.__class__.__name__}]======")
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{class_report}")
    print(f"Confusion Matrix:\n{confusion_mat}")
