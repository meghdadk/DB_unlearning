import torch

def accuracy(model, loader, device):
    model.eval()
    with torch.no_grad():
        _accuracy = 0.0
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            _accuracy += (predicted == targets).sum().item()/len(targets)

        _accuracy /= len(loader)

        
    return _accuracy

def all_tests(model, retain_loader, forget_loader, test_loader, device):
    test_acc = accuracy(model, test_loader, device)
    retain_acc = accuracy(model, retain_loader, device)
    forget_acc = accuracy(model, forget_loader, device)

    return test_acc, retain_acc, forget_acc


