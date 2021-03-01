import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from livelossplot import PlotLosses
from loss import MSEloss_with_Mask
from model import AutoEncoder
from TrainTestDataset import TrainTestDataset

print("Start!!")
print("--" * 20)
# DataLoader
transformations= transforms.Compose([transforms.ToTensor()])
train_dat= TrainTestDataset('./data/train.csv', transformations)
test_dat= TrainTestDataset('./data/test.csv', transformations)

train_dl= DataLoader(dataset=train_dat, batch_size=128, shuffle=True, num_workers=0)
test_dl= DataLoader(dataset=test_dat, batch_size=512, shuffle=False, num_workers=0)

print("DataLoader Finished")

# Model
layer_sizes= [9559, 512, 512, 1024]
model=AutoEncoder(layer_size=layer_sizes, nl_type='selu', is_constrained=True, dp_drop_prob=0.0, last_layer_activations=False)

print("Model constructed")

# Loss & Optimizer
criterion= MSEloss_with_Mask()
optimizer= optim.Adam(model.parameters(), lr= 0.001)

# Training

print("--"*20)
print("Training Start")
print("--"*20)
def train(model, criterion, optimizer, train_dl, test_dl, num_epochs= 40):
    liveloss= PlotLosses()
    lr2_tr_loss, lr2_val_loss= [], []
    for epoch in range(num_epochs):
        train_loss, valid_loss= [], []
        logs= {}
        prefix= ''

        model.train()
        for i, data in enumerate(train_dl, 0):
            inputs = labels= data
            inputs= inputs.float()
            labels= labels.float()

            optimizer.zero_grad()

            outputs= model(inputs)
            loss= criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            # Iterative Dense Output Re-Feeding
            for iter_ in range(3):
                optimizer.zero_grad()

                outputs= model(outputs.detach())
                loss= criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            train_loss.append(loss.item())
            logs[prefix+"MME loss"]= loss.item()

        for i, data in enumerate(test_dl):
            model.eval()
            inputs = labels= data
            inputs= inputs.float()
            labels= labels.float()

            with torch.no_grad():
                outputs= model(inputs)
                loss= criterion(outputs, labels)
                valid_loss.append(loss.item())
                prefix= 'val_'
                logs[prefix + "MMSE loss"]= loss.item()

        lr2_tr_loss.append(np.mean(train_loss))
        lr2_val_loss.append(np.mean(valid_loss))
        liveloss.update(logs)
        liveloss.draw()

        print("Epoch:", epoch+1, "Training Loss: ", np.mean(train_loss), "Valid Loss: ", np.mean(valid_loss))

        if epoch == num_epochs -1:
            return lr2_tr_loss, lr2_val_loss

tr, val= train(model, criterion, optimizer, train_dl, test_dl, 5)


