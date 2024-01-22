from Classes.MLProjectToDataset import split_dataset
from Other.Utils import model_statistics, transformations
from Classes.NetTrainAndTest import train_model
from Models.Demographic import Demographic
from Models.ComplexCNN import ComplexCNN
from Models.SimpleCNN import SimpleCNN
from Models.ResNet import ResNet


def main():
    # Simple CNN
    transforms = transformations(m=50, n=62)
    train_loader, val_loader, test_loader = split_dataset(transformations=transforms['standard'])

    simplecnn_predictions, simplecnn_labels = train_model(
        model=SimpleCNN(),
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=20,
        learning_rate=0.1
    )

    model_statistics(predictions=simplecnn_predictions, true_labels=simplecnn_labels)

    # Complex CNN
    transforms = transformations(m=100, n=62)
    train_loader, val_loader, test_loader = split_dataset(transformations=transforms['standard'])

    complexcnn_predictions, complexcnn_labels = train_model(
        model=ComplexCNN(),
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=20,
        learning_rate=1e-3
    )

    model_statistics(predictions=complexcnn_predictions, true_labels=complexcnn_labels)

    # ResNet34
    transforms = transformations()
    train_loader, _, _ = split_dataset(transformations=transforms['train'])
    _, val_loader, test_loader = split_dataset(transformations=transforms['val'])

    resnet_predictions, resnet_labels = train_model(
        model=ResNet(),
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=5,
        learning_rate=1e-3,
        momentum=0.9
    )

    model_statistics(predictions=resnet_predictions, true_labels=resnet_labels)

    # # Demographic
    # transforms = transformations(m=50, n=62)
    # train_loader, val_loader, test_loader = split_dataset(transformations=transforms['standard'])
    #
    # resnet_predictions, resnet_labels = train_model(
    #     model=Demographic(selected_model='SimpleCNN'),
    #     train_loader=train_loader,
    #     test_loader=test_loader,
    #     epochs=20,
    #     learning_rate=0.1
    # )
    #
    # model_statistics(predictions=resnet_predictions, true_labels=resnet_labels)


if __name__ == "__main__":
    main()
