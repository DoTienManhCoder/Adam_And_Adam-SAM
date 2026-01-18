import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import os

# Thiết lập seed để tái tạo kết quả
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

# CNN Model cho CIFAR-10
class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SmallCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout_conv = nn.Dropout2d(0.25)
        self.dropout_fc = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Conv block 1
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)  # 32x32 -> 16x16
        
        # Conv block 2
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)  # 16x16 -> 8x8
        x = self.dropout_conv(x)
        
        # Conv block 3
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)  # 8x8 -> 4x4
        x = self.dropout_conv(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = self.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        
        return x

# SAM Optimizer
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "SAM requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)
        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

# Hàm huấn luyện
def train_epoch(model, train_loader, optimizer, criterion, device, use_sam=False):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        if use_sam:
            def closure():
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                return loss
            
            loss = optimizer.step(closure)
            
            with torch.no_grad():
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        else:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        
        total_loss += loss.item()
        total += target.size(0)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

# Hàm đánh giá
def evaluate(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / total
    return test_loss, accuracy

# Hàm chính
def main():
    # Thiết lập
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Sử dụng device: {device}')
    
    # Tham số
    batch_size = 128
    epochs = 100
    learning_rate = 0.001
    
    # Data augmentation cho CIFAR-10
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Tải dữ liệu CIFAR-10
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10('./data', train=False, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Classes trong CIFAR-10
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Dictionary để lưu kết quả
    results = {}
    
    # Thực nghiệm với Adam
    print('\n=== Huấn luyện với Adam ===')
    model_adam = SmallCNN().to(device)
    optimizer_adam = optim.Adam(model_adam.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    print(f'Số tham số mô hình: {sum(p.numel() for p in model_adam.parameters())}')
    
    train_losses_adam = []
    train_accs_adam = []
    test_losses_adam = []
    test_accs_adam = []
    times_adam = []
    
    start_time = time.time()
    best_test_acc_adam = 0.0
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_epoch(model_adam, train_loader, optimizer_adam, criterion, device, use_sam=False)
        test_loss, test_acc = evaluate(model_adam, test_loader, criterion, device)
        epoch_time = time.time() - epoch_start
        
        train_losses_adam.append(train_loss)
        train_accs_adam.append(train_acc)
        test_losses_adam.append(test_loss)
        test_accs_adam.append(test_acc)
        times_adam.append(epoch_time)
        
        if test_acc > best_test_acc_adam:
            best_test_acc_adam = test_acc
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, Time: {epoch_time:.2f}s')
    
    total_time_adam = time.time() - start_time
    print(f'Tổng thời gian huấn luyện Adam: {total_time_adam:.2f}s')
    print(f'Best Test Accuracy Adam: {best_test_acc_adam:.2f}%')
    
    results['adam'] = {
        'train_losses': train_losses_adam,
        'train_accs': train_accs_adam,
        'test_losses': test_losses_adam,
        'test_accs': test_accs_adam,
        'times': times_adam,
        'total_time': total_time_adam,
        'best_test_acc': best_test_acc_adam
    }
    
    # Thực nghiệm với Adam + SAM
    print('\n=== Huấn luyện với Adam + SAM ===')
    model_sam = SmallCNN().to(device)
    optimizer_sam = SAM(model_sam.parameters(), optim.Adam, lr=learning_rate, rho=0.05)
    
    train_losses_sam = []
    train_accs_sam = []
    test_losses_sam = []
    test_accs_sam = []
    times_sam = []
    
    start_time = time.time()
    best_test_acc_sam = 0.0
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_epoch(model_sam, train_loader, optimizer_sam, criterion, device, use_sam=True)
        test_loss, test_acc = evaluate(model_sam, test_loader, criterion, device)
        epoch_time = time.time() - epoch_start
        
        train_losses_sam.append(train_loss)
        train_accs_sam.append(train_acc)
        test_losses_sam.append(test_loss)
        test_accs_sam.append(test_acc)
        times_sam.append(epoch_time)
        
        if test_acc > best_test_acc_sam:
            best_test_acc_sam = test_acc
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, Time: {epoch_time:.2f}s')
    
    total_time_sam = time.time() - start_time
    print(f'Tổng thời gian huấn luyện Adam+SAM: {total_time_sam:.2f}s')
    print(f'Best Test Accuracy Adam+SAM: {best_test_acc_sam:.2f}%')
    
    results['sam'] = {
        'train_losses': train_losses_sam,
        'train_accs': train_accs_sam,
        'test_losses': test_losses_sam,
        'test_accs': test_accs_sam,
        'times': times_sam,
        'total_time': total_time_sam,
        'best_test_acc': best_test_acc_sam
    }
    
    # Vẽ biểu đồ so sánh
    plot_results(results, epochs)
    
    # In kết quả cuối cùng
    print('\n=== KẾT QUẢ CUỐI CÙNG ===')
    print(f'Adam - Test Accuracy: {test_accs_adam[-1]:.2f}%, Best: {best_test_acc_adam:.2f}%')
    print(f'Adam+SAM - Test Accuracy: {test_accs_sam[-1]:.2f}%, Best: {best_test_acc_sam:.2f}%')
    print(f'Cải thiện Accuracy (cuối): {test_accs_sam[-1] - test_accs_adam[-1]:.2f}%')
    print(f'Cải thiện Accuracy (best): {best_test_acc_sam - best_test_acc_adam:.2f}%')

def plot_results(results, epochs):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    epochs_range = range(1, epochs + 1)
    
    # Train Loss
    axes[0, 0].plot(epochs_range, results['adam']['train_losses'], label='Adam', linewidth=2)
    axes[0, 0].plot(epochs_range, results['sam']['train_losses'], label='Adam+SAM', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Test Loss
    axes[0, 1].plot(epochs_range, results['adam']['test_losses'], label='Adam', linewidth=2)
    axes[0, 1].plot(epochs_range, results['sam']['test_losses'], label='Adam+SAM', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Test Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Train Accuracy
    axes[1, 0].plot(epochs_range, results['adam']['train_accs'], label='Adam', linewidth=2)
    axes[1, 0].plot(epochs_range, results['sam']['train_accs'], label='Adam+SAM', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].set_title('Training Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Test Accuracy
    axes[1, 1].plot(epochs_range, results['adam']['test_accs'], label='Adam', linewidth=2)
    axes[1, 1].plot(epochs_range, results['sam']['test_accs'], label='Adam+SAM', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].set_title('Test Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('cnn_cifar10_comparison.png', dpi=300, bbox_inches='tight')
    print('\nĐã lưu biểu đồ: cnn_cifar10_comparison.png')
    plt.show()

if __name__ == '__main__':
    main()
