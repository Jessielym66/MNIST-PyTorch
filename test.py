import os
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from model import Classifier

parser = argparse.ArgumentParser(description = 'PyTorch MNIST classification')
parser.add_argument('--batchsize', type=int, default=64, help='input batch size for training')
parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--checkpoints', type=str, default='./logs', help='the path to load model')
args = parser.parse_args()


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, label, reduction='sum')
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print("Average test loss: {:.4f},\t Accuracy: {}/{}  ({:.1f}%)".format(test_loss, correct, len(test_loader.dataset), 100.*correct/len(test_loader.dataset))) 
        


def main():
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    # testing data (1000)
    test_loader = torch.utils.data.DataLoader(
                      datasets.MNIST('./data', train=False, transform = transforms.Compose(
                          [transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])),
                          batch_size=args.batchsize, shuffle=True, **kwargs)
    
    print("====== build model =====")
    model = Classifier().to(device)

    print("====== begin to test =====")
    state_dict = torch.load(os.path.join(args.checkpoints, 'model_20.pkl'))
    model.load_state_dict(state_dict)
    test(args, model, device, test_loader)


if __name__ == '__main__':
    main()



