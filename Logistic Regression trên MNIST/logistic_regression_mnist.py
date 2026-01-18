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

# Logistic Regression Model
class LogisticRegression(nn.Module):
    def __init__(self, input_dim=784, num_classes=10):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        return self.linear(x)

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
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"
        self.base_optimizer.step()  # do the actual "sharpness-aware" update
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "SAM requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass
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
            # First forward-backward pass
            def closure():
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                return loss
            
            loss = optimizer.step(closure)
            
            # Calculate accuracy
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
    epochs = 50
    learning_rate = 0.001
    
    # Tải dữ liệu MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Dictionary để lưu kết quả
    results = {}
    
    # Thực nghiệm với Adam
    print('\n=== Huấn luyện với Adam ===')
    model_adam = LogisticRegression().to(device)
    optimizer_adam = optim.Adam(model_adam.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    train_losses_adam = []
    train_accs_adam = []
    test_losses_adam = []
    test_accs_adam = []
    times_adam = []
    
    start_time = time.time()
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
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, Time: {epoch_time:.2f}s')
    
    total_time_adam = time.time() - start_time
    print(f'Tổng thời gian huấn luyện Adam: {total_time_adam:.2f}s')
    
    results['adam'] = {
        'train_losses': train_losses_adam,
        'train_accs': train_accs_adam,
        'test_losses': test_losses_adam,
        'test_accs': test_accs_adam,
        'times': times_adam,
        'total_time': total_time_adam
    }
    
    # Thực nghiệm với Adam + SAM
    print('\n=== Huấn luyện với Adam + SAM ===')
    model_sam = LogisticRegression().to(device)
    optimizer_sam = SAM(model_sam.parameters(), optim.Adam, lr=learning_rate, rho=0.05)
    
    train_losses_sam = []
    train_accs_sam = []
    test_losses_sam = []
    test_accs_sam = []
    times_sam = []
    
    start_time = time.time()
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
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, Time: {epoch_time:.2f}s')
    
    total_time_sam = time.time() - start_time
    print(f'Tổng thời gian huấn luyện Adam+SAM: {total_time_sam:.2f}s')
    
    results['sam'] = {
        'train_losses': train_losses_sam,
        'train_accs': train_accs_sam,
        'test_losses': test_losses_sam,
        'test_accs': test_accs_sam,
        'times': times_sam,
        'total_time': total_time_sam
    }
    
    # Vẽ biểu đồ so sánh
    plot_results(results, epochs)
    
    # In kết quả cuối cùng
    print('\n=== KẾT QUẢ CUỐI CÙNG ===')
    print(f'Adam - Test Accuracy: {test_accs_adam[-1]:.2f}%, Test Loss: {test_losses_adam[-1]:.4f}')
    print(f'Adam+SAM - Test Accuracy: {test_accs_sam[-1]:.2f}%, Test Loss: {test_losses_sam[-1]:.4f}')
    print(f'Cải thiện Accuracy: {test_accs_sam[-1] - test_accs_adam[-1]:.2f}%')

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
    plt.savefig('logistic_regression_comparison.png', dpi=300, bbox_inches='tight')
    print('\nĐã lưu biểu đồ: logistic_regression_comparison.png')
    plt.show()

if __name__ == '__main__':
    main()
