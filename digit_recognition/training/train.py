import torch
import os
import logging
from models import DigitEfficientNet
from digit_recognition.data.data_preparation import DigitDataset
from digit_recognition.data import utils

torch.manual_seed(340)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


max_val_accuracy = float('-inf')
model_folder = "trained_models"
model_name = "digit_model.pth"
torch_script_model_name = "digit_model_torchscript.pt"
weights_name = "digit_model_weights.pth"

dataset_path = "/Users/mo/Projects/Digit_Recognition_Old/Data/All Digits"
training_path = os.path.join(dataset_path, "training")
validation_path = os.path.join(dataset_path, "validation")

def validate(model: torch.nn.Module, validation_dataset):
    global max_val_accuracy
    model.eval()
    total_length = 0
    correct_predictions = 0
    with torch.no_grad():
        for data in validation_dataset:
            inputs, labels = data
            outputs = model(inputs)
            predictions = torch.argmax(outputs, axis=1)
            correct_predictions += torch.sum(predictions == labels).sum()
            total_length += len(labels)

    acc = correct_predictions/total_length
    print(f'Validation Accuracy: {acc}')
    if acc > max_val_accuracy:
        print("Saving Model")
        max_val_accuracy = acc
        torch.save(model, os.path.join(model_folder, model_name))
        torch.save(model.state_dict(), os.path.join(model_folder, weights_name))
        ts_model = torch.jit.script(model)
        ts_model.save(os.path.join(model_folder, torch_script_model_name))


def train(model: torch.nn.Module, train_dataloader: torch.utils.data.DataLoader, val_dataloader: torch.utils.data.DataLoader = None, epochs: int = 20):
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    running_loss = 0
    running_corrects = 0

    for epoch in range(epochs):
        model.train()
        for batch, data in enumerate(train_dataloader, 0):
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            correct_preds = torch.sum(torch.argmax(outputs, axis=1) == labels)
            running_corrects += correct_preds
            logger.info(f'Batch: {batch} Acc: {correct_preds/inputs.size(0)} loss: {loss.item()}')


        epoch_loss = running_loss / len(train_dataloader)
        epoch_acc = running_corrects.double() /len(train_dataloader)

        logger.info(f'Epoch:{epoch} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        running_loss = 0.0

        if val_dataloader:
            validate(model, val_dataloader)



if __name__ == "__main__":
    # training_path = "Data/All Digits/training"
    # validation_path = "Data/All Digits/vali dation"
    train_dataset= DigitDataset(training_path, transform=utils.data_transforms)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, num_workers=8, shuffle=True)
    val_dataset= DigitDataset(validation_path, transform=utils.val_data_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, num_workers=1)

    model = DigitEfficientNet()

    train(model, train_dataloader, val_dataloader, epochs = 40)



