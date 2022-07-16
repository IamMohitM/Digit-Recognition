import torch
from digit_recognition.data.data_preparation import DigitDataset
import digit_recognition.data.utils as utils
import matplotlib.pyplot as plt

import requests
from tensorflow import make_tensor_proto

torch.random.manual_seed(2034)

def test(model: torch.nn.Module, dataset):
    model.eval()
    total_length = 0
    correct_predictions = 0
    with torch.no_grad():
        for data in dataset:
            inputs, labels = data
            print(inputs.shape)
            print(inputs.dtype)
            outputs = model(inputs)
            predictions = torch.argmax(outputs, axis=1)
            correct_predictions += torch.sum(predictions == labels)
            total_length += len(labels)

    acc = correct_predictions/total_length
    print(f'Test Accuracy: {acc}')

def test_serving(dataset, host, port, model_name):
    total_length = 0
    correct_predictions = 0
    session = requests.Session()

    url = f"http://{host}:{port}/predictions/{model_name}"
    for data in dataset:
        inputs, labels = data
        for index, img in enumerate(inputs):
            print(img.shape)
            image_as_bytes = make_tensor_proto(img).tensor_content
            label = labels[index]
            shape_as_bytes = bytes(str(tuple(img.shape)), 'utf-8')
            input_data = {"data": image_as_bytes, "shape": shape_as_bytes}
            response = int(session.post(url, data=input_data).text)

            correct_predictions += response == label.item()
        
        total_length += len(labels)
        
        print(f'Correct Predictions - {correct_predictions}')
        print(f'Accuracy - {correct_predictions/total_length}')
    








if __name__ == "__main__":
    testing_path = "/Users/mo/Projects/Digit_Recognition_Old/Data/All Digits/testing"
    testing_dataset = DigitDataset(testing_path)
    testing_dataloader = torch.utils.data.DataLoader(testing_dataset, batch_size=16, num_workers=1, shuffle=True, drop_last=True)
    total_length = 0
    correct_predictions = 0

    
    
    # model = torch.load("models/digit_model_v1.pth")
    # model = CNN()
    # model.load_state_dict(torch.load("models/digit_model_3_state_dict.pth"))
    # model = torch.jit.load("trained_models/mobilenet/digit_model_mobilenet_torchscript.pt")
    # test(model, testing_dataloader)

    test_serving(testing_dataloader, "localhost", "8081", 'digitmodel')
    # torch.save(model.state_dict(), "models/digit_model_2_state_dict.pth")
    # ts_model = torch.jit.script(model)
    # ts_model.save("models/digit_model_2_torchscript.pt")
    # with torch.no_grad():
    #     for data in testing_dataloader:
    #         inputs, labels = data
    #         outputs = model(inputs)
    #         max_scores, predictions = torch.max(outputs, axis=1)
    #         # max_scores = 
    #         correct_predictions += torch.sum(predictions == labels).sum()
    #         total_length += len(labels)
    #         for i in range(16):
    #             image = inputs[i]
    #             label = labels[i]
    #             image = image.permute((1, 2, 0)).numpy()
    #             image = image
    #             prediction = predictions[i]
    #             score = max_scores[i]
    #             ax = plt.subplot(4, 4, i + 1)
    #             plt.tight_layout()
    #             ax.set_title(f'{label} - {prediction}-{score:.2f}')
    #             ax.axis('off')

    #             plt.imshow(image, 'gray')
    #             plt.pause(0.001)
    #         acc = correct_predictions/total_length
    #         # print(model)
    #         print(max_scores)
    #         print(predictions)
    #         # print(torch.max(m(inputs), axis=1))
    #         print(f'Testing Accuracy: {acc}')
    #         plt.show()
    #         break


