import matplotlib.pyplot as plt
import torchvision
import logging



data_transforms = torchvision.transforms.Compose(
    [
        # torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomResizedCrop(96, scale=(0.7, 1.0)),
        torchvision.transforms.RandomRotation(25),
        torchvision.transforms.GaussianBlur(3),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

val_data_transforms = torchvision.transforms.Compose(
    [
        # torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((96, 96)),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)


def visualize(dataset, rows=1, columns=4):
    for i in range(len(dataset)):
        image, label = dataset[i]
        image = image.numpy()
        image = image.squeeze()/255
        # print(i, sample['image'].shape, sample['landmarks'].shape)

        ax = plt.subplot(rows, columns, i + 1)
        plt.tight_layout()
        ax.set_title(f'label - {label}')
        ax.axis('off')

        plt.imshow(image, 'gray')
        plt.pause(0.001)

        if i == (rows * columns) - 1:
            break

    plt.show()


def visualize_dataloader(dataloader, rows=1, columns=4):
    data_iterator = iter(dataloader)
    images, labels = next(data_iterator)
    for i in range(rows*columns):
        image = images[i]
        label = labels[i]
        image = image.permute((1, 2, 0)).numpy()/255

        ax = plt.subplot(rows, columns, i + 1)
        plt.tight_layout()
        ax.set_title(f'label - {label}')
        ax.axis('off')

        plt.imshow(image, 'gray')
        plt.pause(0.001)

    plt.show()



def compute_accuracy(predicted, truth):
    return (predicted == truth)/len(predicted)