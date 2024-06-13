# __name__ = 'Ringp'

import torch
import torch.nn as nn

from dataloader import VesselImageDataset
from model import SimpleLinearModel

import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, precision_score

input_size = 128  
learning_rate = 0.01
batch_size = 1
num_epochs = 10

image_classes = ["OOCL_VESSEL_SHIPS", "CARGO_TRUCKS"]
num_calsses = len(image_classes)

dataset = VesselImageDataset(image_classes)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


model = SimpleLinearModel(num_calsses)
criterion = nn.CrossEntropyLoss() 
optimizer = optim.SGD(model.parameters(), lr=learning_rate)



for epoch in range(num_epochs):

    true_labels = []
    pred_labels = []

    for i, (inputs, labels) in enumerate(train_loader):

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        pred_labels += outputs.argmax(1).tolist()
        true_labels += labels.tolist()
    
    epoch_accuracy = accuracy_score(true_labels, pred_labels)
    epoch_precision = precision_score(true_labels, pred_labels)

    print(f"Accuracy: {epoch_accuracy:.4f}, Precision: {epoch_precision:.4f}")