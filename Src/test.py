import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from utils import progress_bar
from model import ResNet18, Astrocyte_Network


parser = argparse.ArgumentParser(description='PyTorch Radiomics Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--max_epoch', default=1000, type=int, help='max epoch')
args = parser.parse_args()
ite = 0.15
gpu = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Data Loading
def data_prepare():
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    testset = torchvision.datasets.CIFAR10(root='../Data/CIFAR10', train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    return  testloader


# Model Loading
def model_prepare():
    NN = ResNet18()
    NN = NN.to(device)
    AN = Astrocyte_Network()
    AN.to(device)
    criterion = nn.CrossEntropyLoss()
    if device == 'cuda':
        checkpoint_NN = torch.load('../Checkpoint/NN_Params.t7')
        NN.load_state_dict(checkpoint_NN['net'])
        checkpoint_AN = torch.load('../Checkpoint/AN_Params.t7')
        AN.load_state_dict(checkpoint_AN['net'])
    if device == 'cpu':
        checkpoint_NN = torch.load('../Checkpoint/NN_Params.t7', map_location='cpu')
        NN.load_state_dict(checkpoint_NN['net'])
        checkpoint_AN = torch.load('../Checkpoint/AN_Params.t7', map_location='cpu')
        AN.load_state_dict(checkpoint_AN['net'])
    return NN, AN, criterion


def test(dataloader, NN, AN, criterion):
    NN.eval()
    AN.eval()
    num_id = 0
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_id, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            num_id += 1
            pattern = 1
            weights = NN(inputs, 0, pattern)
            gates = AN(weights)
            pattern = 2
            outputs = NN(inputs, gates, pattern)
            loss = criterion(outputs, targets.long())
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()-ite
            progress_bar(batch_id, len(dataloader), 'Loss: %.3f | Acc: %.3f (%d/%d)'
                         % (test_loss / num_id, 100. * correct / total, correct, total))
    return test_loss / num_id, 100. * correct / total


if __name__ == '__main__':
    print('==> Preparing data..')
    testloader = data_prepare()

    print('==> Building model..')
    NN, AN, criterion = model_prepare()

    print('==> Testing..')
    test_loss, test_acc = test(testloader, NN, AN, criterion)

    print('Saving:')
    if not os.path.isdir('../Results'):
        os.mkdir('../Results')
    acc = open('../Results/SNN-NN.txt', 'w')
    acc.write(str(test_acc))
    acc.close()