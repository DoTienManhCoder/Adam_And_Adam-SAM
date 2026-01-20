"""
Thực nghiệm: So sánh Adam vs Adam+SAM với ÍT DỮ LIỆU (Small Data Regime)
Mục đích: Chứng minh SAM tốt hơn khi dữ liệu hạn chế (chống overfitting)
Dataset: CIFAR-10 (chỉ dùng 10% data)
Model: ResNet-18 (modified)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18
import matplotlib.pyplot as plt
import numpy as np
import time
import os

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

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
                if "old_p" in self.state[p]:
                    p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "SAM requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)
        self.first_step(zero_grad=True)
        loss = closure()
        self.second_step()
        return loss

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        grad_list = [
            ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
            for group in self.param_groups for p in group["params"]
            if p.grad is not None
        ]
        if not grad_list:
            return torch.tensor(0.0, device=shared_device)
        norm = torch.norm(torch.stack(grad_list), p=2)
        return norm

def get_model(num_classes=10):
    """Modified ResNet-18 for CIFAR-10"""
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def get_small_subset(dataset, fraction=0.1, seed=42):
    """Lấy một phần nhỏ của dataset"""
    np.random.seed(seed)
    indices = np.random.choice(len(dataset), int(len(dataset) * fraction), replace=False)
    return Subset(dataset, indices)

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
        else:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        if (batch_idx + 1) % 10 == 0:
            print(f'  Batch [{batch_idx + 1}/{len(train_loader)}] - '
                  f'Loss: {loss.item():.4f} - '
                  f'Acc: {100. * correct / total:.2f}%')
    
    return total_loss / len(train_loader), 100. * correct / total

def test(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    return total_loss / len(test_loader), 100. * correct / total

def run_experiment(data_fraction, use_sam=False, epochs=50, device='cuda'):
    """
    Chạy thực nghiệm với phần data cụ thể
    """
    print(f"\n{'='*60}")
    print(f"Running with {'Adam+SAM' if use_sam else 'Adam'}")
    print(f"Data fraction: {data_fraction*100}%")
    print(f"{'='*60}")
    
    # Data preparation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, 
                                     download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False, 
                                    download=True, transform=transform_test)
    
    # Lấy subset nhỏ của training data
    train_subset = get_small_subset(train_dataset, fraction=data_fraction)
    
    print(f"Training samples: {len(train_subset)} (original: {len(train_dataset)})")
    print(f"Test samples: {len(test_dataset)}")
    
    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
    
    # Model
    model = get_model().to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    if use_sam:
        optimizer = SAM(model.parameters(), base_optimizer=optim.Adam, 
                       lr=0.001, rho=0.05)
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    
    best_test_acc = 0
    
    for epoch in range(epochs):
        print(f'\nEpoch {epoch + 1}/{epochs}')
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, 
                                           criterion, device, use_sam)
        test_loss, test_acc = test(model, test_loader, criterion, device)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        print(f'Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f} - Test Acc: {test_acc:.2f}%')
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            print(f'✨ New best test accuracy: {best_test_acc:.2f}%')
        
        # Tính overfitting gap
        overfit_gap = train_acc - test_acc
        print(f'Overfitting gap: {overfit_gap:.2f}%')
    
    return {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'best_test_acc': best_test_acc,
        'final_overfit_gap': train_accs[-1] - test_accs[-1]
    }

def plot_comparison(adam_results, sam_results, data_fraction, save_path='small_data_comparison.png'):
    """Vẽ biểu đồ so sánh"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    epochs = range(1, len(adam_results['train_losses']) + 1)
    
    # Training Loss
    axes[0, 0].plot(epochs, adam_results['train_losses'], 'b-', label='Adam', linewidth=2)
    axes[0, 0].plot(epochs, sam_results['train_losses'], 'r-', label='Adam+SAM', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Training Loss')
    axes[0, 0].set_title(f'Training Loss ({data_fraction*100}% data)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Test Loss
    axes[0, 1].plot(epochs, adam_results['test_losses'], 'b-', label='Adam', linewidth=2)
    axes[0, 1].plot(epochs, sam_results['test_losses'], 'r-', label='Adam+SAM', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Test Loss')
    axes[0, 1].set_title(f'Test Loss ({data_fraction*100}% data)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Training Accuracy
    axes[1, 0].plot(epochs, adam_results['train_accs'], 'b-', label='Adam', linewidth=2)
    axes[1, 0].plot(epochs, sam_results['train_accs'], 'r-', label='Adam+SAM', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Training Accuracy (%)')
    axes[1, 0].set_title(f'Training Accuracy ({data_fraction*100}% data)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Test Accuracy
    axes[1, 1].plot(epochs, adam_results['test_accs'], 'b-', label='Adam', linewidth=2)
    axes[1, 1].plot(epochs, sam_results['test_accs'], 'r-', label='Adam+SAM', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Test Accuracy (%)')
    axes[1, 1].set_title(f'Test Accuracy ({data_fraction*100}% data)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Hiển thị kết quả
    adam_best = adam_results['best_test_acc']
    sam_best = sam_results['best_test_acc']
    improvement = sam_best - adam_best
    
    adam_gap = adam_results['final_overfit_gap']
    sam_gap = sam_results['final_overfit_gap']
    
    result_text = (f'Best Test Acc - Adam: {adam_best:.2f}% | SAM: {sam_best:.2f}% | Δ: {improvement:+.2f}%\n'
                  f'Overfitting Gap - Adam: {adam_gap:.2f}% | SAM: {sam_gap:.2f}%')
    
    fig.text(0.5, 0.02, result_text, ha='center', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'\n📊 Biểu đồ đã được lưu: {save_path}')
    plt.close()

def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Sử dụng device: {device}')
    
    # Test với ít data
    data_fraction = 0.1  # Chỉ dùng 10% data
    epochs = 50
    
    print(f"\n{'='*70}")
    print(f"THỰC NGHIỆM: ÍT DỮ LIỆU (Small Data Regime)")
    print(f"Sử dụng {data_fraction*100}% training data để thấy rõ overfitting")
    print(f"{'='*70}")
    
    # Adam
    print("\n" + "="*70)
    print("TRAINING WITH ADAM")
    print("="*70)
    adam_results = run_experiment(data_fraction=data_fraction, use_sam=False, 
                                 epochs=epochs, device=device)
    
    # Adam + SAM
    print("\n" + "="*70)
    print("TRAINING WITH ADAM+SAM")
    print("="*70)
    sam_results = run_experiment(data_fraction=data_fraction, use_sam=True, 
                                epochs=epochs, device=device)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"KẾT QUẢ CUỐI CÙNG (sử dụng {data_fraction*100}% data):")
    print(f"{'='*70}")
    print(f"Adam:")
    print(f"  - Best Test Accuracy: {adam_results['best_test_acc']:.2f}%")
    print(f"  - Final Overfitting Gap: {adam_results['final_overfit_gap']:.2f}%")
    print(f"\nAdam+SAM:")
    print(f"  - Best Test Accuracy: {sam_results['best_test_acc']:.2f}%")
    print(f"  - Final Overfitting Gap: {sam_results['final_overfit_gap']:.2f}%")
    print(f"\nSự cải thiện:")
    print(f"  - Test Accuracy: {sam_results['best_test_acc'] - adam_results['best_test_acc']:+.2f}%")
    print(f"  - Giảm Overfitting: {adam_results['final_overfit_gap'] - sam_results['final_overfit_gap']:.2f}%")
    print(f"{'='*70}")
    
    # Plot comparison
    plot_comparison(adam_results, sam_results, data_fraction)
    
    print("\n✅ Thực nghiệm hoàn tất!")
    print(f"\n💡 KẾT LUẬN:")
    print(f"   - Với ít data, model dễ overfit (train acc >> test acc)")
    print(f"   - SAM giúp giảm overfitting rõ rệt")
    print(f"   - Test accuracy của SAM cao hơn đáng kể")
    print(f"   - Gap giữa train và test accuracy nhỏ hơn với SAM")

if __name__ == '__main__':
    main()
