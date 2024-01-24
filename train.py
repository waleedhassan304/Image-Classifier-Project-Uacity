# Basic usage: python train.py data_directory
# Prints out training loss, validation loss, and validation accuracy as the network trains
# Options:
#         Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
#         Set hyperparameters: python train.py data_dir --learning_rate 0.01 --epochs 5 --gpu

import argparse
import torch
from torch import nn, optim
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os


def create_model(arch):
    if arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        in_features = 1024
    elif arch == 'resnet50':
        model = models.resnet50(pretrained=True)
        in_features = 2048
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        in_features = 9216
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        in_features = 25088
    else:
        raise ValueError("Unsupported architecture. Choose from 'densenet121', 'resnet50', 'alexnet', or 'vgg16'.")

    return model, in_features

def data_transformation(args):
    # Define transformations, ImageFolder & DataLoader
    # Returns DataLoader objects for training and validation, and a class_to_idx dictionary
    train_dir = os.path.join(args.data_directory, "train")
    valid_dir = os.path.join(args.data_directory, "valid")

    # Validate paths before doing anything else
    if not os.path.exists(args.data_directory):
        print("Data Directory doesn't exist: {}".format(args.data_directory))
        raise FileNotFoundError

    if not os.path.exists(train_dir):
        print("Train folder doesn't exist: {}".format(train_dir))
        raise FileNotFoundError

    if not os.path.exists(valid_dir):
        print("Valid folder doesn't exist: {}".format(valid_dir))
        raise FileNotFoundError

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_data = ImageFolder(root=train_dir, transform=train_transforms)
    valid_data = ImageFolder(root=valid_dir, transform=valid_transforms)

    train_data_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    valid_data_loader = DataLoader(valid_data, batch_size=64, shuffle=True)

    return train_data_loader, valid_data_loader, train_data.class_to_idx

def train_model(args, train_data_loader, valid_data_loader, class_to_idx):
    # Train the model, save model to directory, return True if successful

    # Load a pre-trained VGG16 model
    model = models.vgg16(pretrained=True)

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # Modify the classifier to match your output size
    classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 102),  # Assuming 102 output classes
        nn.LogSoftmax(dim=1)
    )

    model.classifier = classifier

    # Define the criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    # Move the model to the appropriate device
    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
    model.to(device)

    # Training loop
    for epoch in range(args.epochs):
        model.train()

        for inputs, labels in train_data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        accuracy = 0
        valid_loss = 0

        with torch.no_grad():
            for inputs, labels in valid_data_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                valid_loss += criterion(outputs, labels).item()

                ps = torch.exp(outputs)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        print(f"Epoch {epoch + 1}/{args.epochs}.. "
              f"Training Loss: {loss:.3f}.. "
              f"Validation Loss: {valid_loss / len(valid_data_loader):.3f}.. "
              f"Validation Accuracy: {accuracy / len(valid_data_loader) * 100:.2f}%")

    # Save the model checkpoint
    model.class_to_idx = class_to_idx
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'optimizer_state_dict': optimizer.state_dict(),
        'epochs': args.epochs
    }
    torch.save(checkpoint, os.path.join(args.save_directory, 'checkpoint.pth'))

    print(f"Model successfully trained and saved to {os.path.join(args.save_directory, 'checkpoint.pth')}")
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_directory', help="Path to the data directory")
    parser.add_argument('--save_directory', dest='save_directory', default='./', help="Directory to save the model checkpoint")
    parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=0.001, help="Learning rate for training")
    parser.add_argument('--epochs', dest='epochs', type=int, default=5, help="Number of epochs for training")
    parser.add_argument('--gpu', dest='gpu', action='store_true', help="Use GPU for training if available")

    args = parser.parse_args()

    train_data_loader, valid_data_loader, class_to_idx = data_transformation(args)
    train_model(args, train_data_loader, valid_data_loader, class_to_idx)
