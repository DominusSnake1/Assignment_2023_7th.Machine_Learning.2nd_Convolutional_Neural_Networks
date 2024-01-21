from Classes.MLProjectToDataset import MLProject2Dataset, split_dataset
from Other.Utils import transformation, model_statistics
from Classes.NetTrainAndTest import train_model


def main():
    dataset = MLProject2Dataset(
        data_dir='Data/',
        transform=transformation(50, 62)
    )

    train_loader, val_loader, test_loader = split_dataset(dataset=dataset)

    # # Simple CNN
    # from Models.SimpleCNN import SimpleCNN
    # simplecnn_predictions, simplecnn_labels = train_model(
    #     model=SimpleCNN(),
    #     train_loader=train_loader,
    #     test_loader=test_loader,
    #     epochs=20,
    #     learning_rate=0.1
    # )
    # model_statistics(predictions=simplecnn_predictions, true_labels=simplecnn_labels)
    #
    # # Complex CNN
    # from Models.ComplexCNN import ComplexCNN
    # complexcnn_predictions, complexcnn_labels = train_model(
    #     model=ComplexCNN(),
    #     train_loader=train_loader,
    #     test_loader=test_loader,
    #     epochs=20,
    #     learning_rate=1e-3
    # )
    # model_statistics(predictions=complexcnn_predictions, true_labels=complexcnn_labels)

    # ResNet34
    from Models.ResNet import train_resnet
    resnet_predictions, resnet_labels = train_resnet(
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=5
    )
    model_statistics(predictions=resnet_predictions, true_labels=resnet_labels)


if __name__ == "__main__":
    main()
