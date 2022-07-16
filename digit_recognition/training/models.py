import torch, torchvision


class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = self._model_prep()

    def _model_prep(self):
        model = torchvision.models.mobilenet_v3_small(pretrained=True, progress=True)
        classification_layer = torch.nn.Sequential(
            torch.nn.Linear(576, out_features=1024),
            torch.nn.Hardswish(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(1024, 10),
            # torch.nn.Softmax()
        )
        model.classifier = classification_layer

        for param in model.features[:13].parameters():
            param.requires_grad = False
        return model

    def forward(self, x):
        return self.model(x)

class DigitEfficientNet(torch.nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes =  num_classes
        self._model_prep()

    def _model_prep(self):
        eff_net = torchvision.models.efficientnet_v2_s(weights = torchvision.models.EfficientNet_V2_S_Weights.DEFAULT)
        for param in eff_net.parameters():
            param.requires_grad = False

        classification_layer = torch.nn.Sequential(
            torch.nn.Dropout(0.5, inplace=True),
            torch.nn.Linear(in_features=eff_net.classifier[1].in_features, out_features=self.num_classes)
        )
        eff_net.classifier = classification_layer

        self.model = eff_net

    def forward(self, x):
        return self.model(x)



if __name__ == "__main__":
    model = DigitEfficientNet()
    print(model)
    for param in model.parameters():
        print(param.requires_grad)
    test_data = torch.randn(size= (1, 3, 96, 96))
    print(model(test_data))