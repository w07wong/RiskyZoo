import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import random

import numpy as np

from riskyzoo.supervised_learning.risk_functionals import human_aligned_risk, entropic_risk, trimmed_risk, cvar, mean_variance

from models import VGG 

batch_size_train = 5 
batch_size_test = 5 
lr = 0.005

# Training
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def train(epoch, risk, objective):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0 
    correct = 0 
    total = 0 

    all_losses = torch.tensor([], device=device)

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        losses = criterion(outputs, targets)
        all_losses = torch.cat((all_losses, losses.flatten()))

        if risk.split('_')[0] == 'Expected Value':
            loss = torch.mean(losses)
        else:
            loss = objective(torch.clone(losses).cpu())
            loss = loss.to(device)

        print('train epoch: {}, batch_idx: {}, risk: {}, loss: {}, avg ce loss: {}'.format(epoch, batch_idx, risk, loss, torch.mean(losses)))
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    acc = 100.*correct/total
    np.save(str(risk) + '/' + str(epoch) + '_acc_train.npy', acc)
    np.save(str(risk) + '/' + str(epoch) + '_train.npy', all_losses.cpu().detach().numpy())
    return (train_loss, 100.*correct/total, correct, total)


def test(epoch, risk, objective):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    all_losses = torch.tensor([], device=device)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            losses = criterion(outputs, targets)
            all_losses = torch.cat((all_losses, losses.flatten()))

            if risk.split('_')[0] == 'Expected Value':
                loss = torch.mean(losses)
            else:
                loss = objective(torch.clone(losses).cpu())
                loss = loss.to(device)

            print('test epoch: {}, batch_idx: {}, risk: {}, loss: {}, avg ce loss: {}'.format(epoch, batch_idx, risk, loss, torch.mean(losses)))

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    acc = 100.*correct/total
    print('Epoch: {}, Accuracy: {}'.format(epoch, acc))
    np.save(str(risk) + '/' + str(epoch) + '_test.npy', all_losses.cpu().detach().numpy())
    np.save(str(risk) + '/' + str(epoch) + '_acc_test.npy', acc)

    # Save checkpoint.
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './' + str(risk) + '/checkpoint/ckpt.pth')
        best_acc = acc

    return (test_loss, acc, correct, total)


##############################################################
seed = 12
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

print('==> Preparing data..')
transform_train = transforms.Compose([
    # transforms.Resize(299),
    # transforms.CenterCrop(299),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    # transforms.Resize(299),
    # transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)

# Randomly shuffle 80% of targets
old_targets = trainset.targets[:]
trainset.targets = [trainset.targets[i] if random.random() < 0.2 else random.randint(0, 9) for i in range(len(trainset.targets))]
'''
to_change = random.sample([i for i in range(len(trainset.targets))], k=round(len(trainset.targets) * 0.8))
for i in to_change:
    trainset.targets[i] = random.randint(0, 9)
'''

num_match = 0
for i in range(len(old_targets)):
    if old_targets[i] == trainset.targets[i]:
        num_match += 1
print('num match:', num_match, num_match / 50000)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size_train, shuffle=True, num_workers=3)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size_test, shuffle=False, num_workers=3)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')


risk_functionals = {
    'Expected Value': nn.CrossEntropyLoss(),
    'CVaR-0.9': cvar.CVaR(a=0.9),
    'Entropic Risk=-0.5': entropic_risk.EntropicRisk(t=-0.5),
    'Human-Aligned Risk a=0.8 b=0.2': human_aligned_risk.HumanAlignedRisk(a=0.8, b=0.2),
    'Inverted CVaR-0.9': cvar.CVaR(a=0.9, inverted=True),
    'Mean-Variance=-0.1': mean_variance.MeanVariance(c=-0.1),
    'Trimmed Risk-0.05': trimmed_risk.TrimmedRisk(a=0.05),
}

seeds = [i for i in range(5)]

for seed in seeds:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    for risk in risk_functionals.keys():
        best_acc = 0
        start_epoch = 0  # start from epoch 0 or last checkpoint epoch

        net = VGG('VGG11')
        # net = torchvision.models.inception_v3(pretrained=False, aux_logits=False)
        net = net.to(device)
        criterion = nn.CrossEntropyLoss(reduce=False, reduction='none')
        optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

        os.makedirs(str(risk) + '_seed=' + str(seed), exist_ok=True)
        os.makedirs(str(risk) + '_seed=' + str(seed) + '/checkpoint', exist_ok=True)

        if device == 'cuda:0':
            net = net.cuda()
            cudnn.benchmark = True

        for epoch in range(start_epoch, start_epoch+150):
            print('#################################################################')
            print('LEARNING RATE:', scheduler.optimizer.param_groups[0]['lr'])
            train(epoch, risk + '_seed=' + str(seed), risk_functionals[risk])
            test(epoch, risk + '_seed=' + str(seed), risk_functionals[risk])
            scheduler.step()

