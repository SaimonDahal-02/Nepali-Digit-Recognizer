import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.optim as optim

from torch import nn
from loader.data_loader import dataloader
from ml.cnn.model import ConvNeuralNetwork
from utils.early_stopping import EarlyStopping

import mlflow

mlflow.set_tracking_uri("http://3.249.112.184:5000/")
mlflow.set_experiment(experiment_id="2")

def main():

    train_loader, val_loader, _ = dataloader()
    learning_rate = mlflow.log_param('learning rate', 0.001)
    # momentum = mlflow.log_param('momentum', 0.9)
    patience = mlflow.log_param('patience', 3)
    min_delta = mlflow.log_param('min_delta', 0.001)
    epochs = mlflow.log_param('epochs', 100)
    mlflow.log_param('weight_initialization', 'xavier_normal')
    mlflow.log_param('activation_function', 'GELU')
    mlflow.log_param('optimizer_name', 'SGD')

    early_stopping = EarlyStopping()

    model = ConvNeuralNetwork()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)

    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1)

    n_epochs = epochs

    train_loss = []
    train_accuracy = []
    val_loss = []
    val_accuracy = []

    current_lr = learning_rate

    for epoch in range(n_epochs):
        with tqdm(train_loader, unit='banana', colour='GREEN') as tepoch:
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for input, labels in tepoch:
                tepoch.set_description(f"Training Epoch {epoch}")
                y_pred = model(input)
                loss = loss_fn(y_pred, labels)
                optimizer.zero_grad()
                loss.backward() # kick off backpropagation
                optimizer.step() # initiate gradient descent
                running_loss += loss.item()

                # Calculate training accuracy
                _, predicted = torch.max(y_pred, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                epoch_train_accuracy = correct / total
                tepoch.set_postfix(Training_loss=loss.item(), Training_accuracy = epoch_train_accuracy)


            epoch_train_loss = running_loss / len(train_loader)
            epoch_train_accuracy = correct / total
            mlflow.log_metrics({
                'Training Loss': epoch_train_loss,
                'Training Accuracy': epoch_train_accuracy
            }, step=epoch)
            train_loss.append(epoch_train_loss)
            train_accuracy.append(epoch_train_accuracy)
            
            # validation
            model.eval()
            correct = 0
            total = 0
            val_running_loss = 0.0
            with torch.no_grad():
                with tqdm(val_loader, unit='banana', colour='BLUE') as vepoch:
                    for input, labels in vepoch:
                        vepoch.set_description(f"Validation Epoch {epoch}")
                        y_pred = model(input)
                        v_loss = loss_fn(y_pred, labels)
                        val_running_loss += v_loss.item()
                        # Calculate training accuracy
                        _, predicted = torch.max(y_pred, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        epoch_val_accuracy = correct / total
                        vepoch.set_postfix(Validation_loss=v_loss.item(), Validation_accuracy = epoch_val_accuracy)


            epoch_val_loss = val_running_loss / len(val_loader)
            epoch_val_accuracy = correct / total
            mlflow.log_metrics({
                'Validation Loss': epoch_val_loss,
                'Validation Accuracy': epoch_val_accuracy
            }, step=epoch)
            val_loss.append(epoch_val_loss)
            val_accuracy.append(epoch_val_accuracy)


            lr_scheduler.step(epoch_val_loss)
            early_stopping(epoch_val_loss)

            if early_stopping.counter >= 5:
                if early_stopping.early_stop:
                    print("We are at epoch: ", epoch)
                    break
                else:
                    current_lr *= 0.1
                    early_stopping.counter = 0
                    for param_group in optimizer.param_groups:
                            param_group['lr'] = current_lr
                    print(f"Learning rate decreased to {current_lr}")

        print(' ' * 220)

    evaluate_model(train_loss, val_loss, train_accuracy, val_accuracy)

    torch.save(model.state_dict(), "models/adagrad_model.pth")

def evaluate_model(train_loss, val_loss, train_accuracy, val_accuracy):
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracy, label='Training Accuracy')
    plt.plot(val_accuracy, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig("plots/adagrad_plot.png")

if __name__ == "__main__":
    with mlflow.start_run(run_name="CNN + Variable lr + Adagrad", nested=True) as run:
        main()