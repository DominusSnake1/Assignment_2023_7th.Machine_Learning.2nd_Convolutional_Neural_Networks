from Classes.MLProjectToDataset import MLProject2Dataset, split_dataset
from Other.Utils import transformation, model_train_and_stats
from Models.ComplexCNN import ComplexCNN
from Models.SimpleCNN import SimpleCNN


def main():
    dataset = MLProject2Dataset(
        data_dir='Data/',
        transform=transformation(50, 62)
    )
    train_loader, val_loader, test_loader = split_dataset(dataset=dataset)

    # Simple CNN
    model_train_and_stats(model=SimpleCNN(),
                          train_loader=train_loader,
                          test_loader=test_loader,
                          epochs=20,
                          learning_rate=0.1
                          )

    # Complex CNN
    model_train_and_stats(model=ComplexCNN(),
                          train_loader=train_loader,
                          test_loader=test_loader,
                          epochs=20,
                          learning_rate=1e-3)


if __name__ == "__main__":
    main()
