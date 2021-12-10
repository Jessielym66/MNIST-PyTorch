import os
import argparse
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from model import Classifier

parser = argparse.ArgumentParser(description = 'PyTorch MNIST classification')
parser.add_argument('--batchsize', type=int, default=64, help='input batch size for training')
parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.5, help='SGD momentum')
parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--save_dir', type=str, default='./logs', help='the path to save model and results')
parser.add_argument('--save_freq', type=int, default=5, help='save model every freq epochs')
parser.add_argument('--log_freq', type=int, default=100, help='print loss every log_freq*batchsize images')
args = parser.parse_args()


def train(args, model, device, train_loader, val_loader, optimizer):
    training_losses = []
    validate_losses = []
    accuracies = []
    loss_file = os.path.join(args.save_dir, 'loss.txt')
    
    with open(loss_file, mode='w', encoding='UTF-8') as f:    
        for epoch in range(1, args.epochs + 1):
            # train
            print("====== epoch {} =====".format(epoch))
            model.train()
            for batch_idx, (data, label) in enumerate(train_loader):
                data, label = data.to(device), label.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, label)
                loss.backward()
                optimizer.step()
                if batch_idx % args.log_freq == 0:
                    text = 'Train Epoch: {} [{}/{}  ({:.0f}%)]\t Loss: {:.3f}'.format(epoch, batch_idx*len(data), len(train_loader.dataset), 100.*batch_idx/len(train_loader), loss.item())
                    print(text)
                    f.write(text+'\n')
                    training_losses.append(loss.item())
            
            # val
            model.eval()
            val_loss = 0
            correct = 0
            with torch.no_grad():
                for data, label in val_loader:
                    data, label = data.to(device), label.to(device)
                    output = model(data)
                    val_loss += F.nll_loss(output, label, reduction='sum')
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(label.view_as(pred)).sum().item()
            val_loss /= len(val_loader.dataset)
            validate_losses.append(val_loss)
            accuracies.append(100.*correct/len(val_loader.dataset))
            text = "Average val loss: {:.4f},\t Accuracy: {}/{}  ({:.1f}%)".format(val_loss, correct, len(val_loader.dataset), 100.*correct/len(val_loader.dataset))
            print(text)
            f.write(text+'\n') 
            
            # save
            if epoch % args.save_freq == 0:
                torch.save(model.state_dict(), os.path.join(args.save_dir, 'model_'+str(epoch)+'.pkl'))
        
    # plot     
    x1 = range(0, len(training_losses))
    x2 = range(0, len(validate_losses))
    x3 = range(0, len(accuracies))
    if len(validate_losses) != len(accuracies):
        print("val loss and accu don't consistent.")
    
    plt.figure(1)
    plt.plot(x1, training_losses, color='blue')
    plt.title("training loss")
    plt.xlabel("per {} batches".format(args.log_freq))
    plt.ylabel("loss")
    plt.savefig(os.path.join(args.save_dir, "training.png"))
    
    fig = plt.figure(2)
    ax1 = fig.subplots()
    ax2 = ax1.twinx()
    ax1.plot(x2, validate_losses, 'b-')
    ax2.plot(x3, accuracies, 'g-')
    ax1.set_title("validate loss and accuracy")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel('val loss')
    ax2.set_ylabel('accuracy')
    fig.savefig(os.path.join(args.save_dir, "validate.png"))
    
    
 

def main():
    # the path to save loss file and model
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    t = time.localtime(time.time())
    filename = str(t.tm_year) + str(t.tm_mon) + str(t.tm_mday) + '_' + str(t.tm_hour) + str(t.tm_min) + str(t.tm_sec)
    args.save_dir = os.path.join(args.save_dir, filename)
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    # training data (6000)
    train_loader = torch.utils.data.DataLoader(
                      datasets.MNIST('./data', train=True, download=True, transform = transforms.Compose(
                          [transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])),
                          batch_size=args.batchsize, shuffle=True, **kwargs)
    
    # validate data (1000)             
    val_loader = torch.utils.data.DataLoader(
                      datasets.MNIST('./data', train=False, transform = transforms.Compose(
                          [transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])),
                          batch_size=args.batchsize, shuffle=True, **kwargs)

    print("====== build model =====")
    model = Classifier().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    print("====== begin to train =====")
    train(args, model, device, train_loader, val_loader, optimizer)


if __name__ == '__main__':
    main()



