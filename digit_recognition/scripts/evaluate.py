import torch
from digit_recognition.data.data_preparation import DigitDataset
import digit_recognition.data.utils as utils
import matplotlib.pyplot as plt
from digit_recognition.training.models import CNN

if __name__ == "__main__":
    testing_path = "Data/All Digits/testing"
    testing_dataset = DigitDataset(testing_path, transform=utils.val_data_transforms)
    testing_dataloader = torch.utils.data.DataLoader(testing_dataset, batch_size=16, num_workers=1, shuffle=True)
    total_length = 0
    correct_predictions = 0

    
    # model = torch.load("models/digit_model_v1.pth")
    model = CNN()
    model.load_state_dict(torch.load("models/digit_model_3_state_dict.pth"))
    # m.eval()
    model.eval()
    # torch.save(model.state_dict(), "models/digit_model_2_state_dict.pth")
    # ts_model = torch.jit.script(model)
    # ts_model.save("models/digit_model_2_torchscript.pt")
    with torch.no_grad():
        for data in testing_dataloader:
            inputs, labels = data
            outputs = model(inputs)
            max_scores, predictions = torch.max(outputs, axis=1)
            # max_scores = 
            correct_predictions += torch.sum(predictions == labels).sum()
            total_length += len(labels)
            for i in range(16):
                image = inputs[i]
                label = labels[i]
                image = image.permute((1, 2, 0)).numpy()
                image = image
                prediction = predictions[i]
                score = max_scores[i]
                ax = plt.subplot(4, 4, i + 1)
                plt.tight_layout()
                ax.set_title(f'{label} - {prediction}-{score:.2f}')
                ax.axis('off')

                plt.imshow(image, 'gray')
                plt.pause(0.001)
            acc = correct_predictions/total_length
            # print(model)
            print(max_scores)
            print(predictions)
            # print(torch.max(m(inputs), axis=1))
            print(f'Testing Accuracy: {acc}')
            plt.show()
            break


