import torch
import torch.nn as nn

class SimpleLinearModel(nn.Module):

    def __init__(self, num_classes):
        super(SimpleLinearModel, self).__init__()
        self.fc = nn.Linear(3*128*128, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
if __name__ == '__main__':
    model = SimpleLinearModel(2)
    print(model)
    img = torch.rand(1, 3, 128, 128)
    out = model(img)
    print(out)