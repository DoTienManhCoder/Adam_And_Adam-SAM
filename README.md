# So sánh Thuật toán Tối ưu: Adam vs Adam+SAM

Dự án này thực hiện 3 thực nghiệm để so sánh hiệu suất của thuật toán Adam với Adam kết hợp Sharpness-Aware Minimization (SAM).

## 📋 Mục lục

1. [Thực nghiệm 1: Logistic Regression trên MNIST](#thực-nghiệm-1)
2. [Thực nghiệm 2: MLP trên MNIST](#thực-nghiệm-2)
3. [Thực nghiệm 3: CNN nhỏ trên CIFAR-10](#thực-nghiệm-3)

## 🚀 Cài đặt

### Yêu cầu hệ thống
- Python 3.8 trở lên
- GPU (khuyến nghị, không bắt buộc)

### Cài đặt thư viện

```bash
pip install -r requirements.txt
```

Hoặc cài đặt thủ công:

```bash
pip install torch torchvision matplotlib numpy
```

### Cài đặt PyTorch với CUDA (nếu có GPU)

Truy cập https://pytorch.org/ để cài đặt phiên bản phù hợp với hệ thống của bạn.

Ví dụ cho Windows với CUDA 11.8:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 📊 Thực nghiệm 1: Logistic Regression trên MNIST

### Mô tả
- **Mô hình**: Logistic Regression (Linear layer đơn giản)
- **Dataset**: MNIST (28x28 grayscale images, 10 classes)
- **Số tham số**: ~7,850
- **Epochs**: 50
- **Batch size**: 128
- **Learning rate**: 0.001

### Chạy thực nghiệm

```bash
cd "Logistic Regression trên MNIST"
python logistic_regression_mnist.py
```

### Kết quả mong đợi
- Adam: ~92-93% test accuracy
- Adam+SAM: ~93-94% test accuracy (cải thiện ~1%)

## 📊 Thực nghiệm 2: MLP trên MNIST

### Mô tả
- **Mô hình**: Multi-Layer Perceptron (2 hidden layers: 256, 128)
- **Dataset**: MNIST (28x28 grayscale images, 10 classes)
- **Số tham số**: ~235,146
- **Epochs**: 50
- **Batch size**: 128
- **Learning rate**: 0.001
- **Dropout**: 0.2

### Chạy thực nghiệm

```bash
cd "MLP trên MNIST"
python mlp_mnist.py
```

### Kết quả mong đợi
- Adam: ~97-98% test accuracy
- Adam+SAM: ~98-99% test accuracy (cải thiện ~0.5-1%)

## 📊 Thực nghiệm 3: CNN nhỏ trên CIFAR-10

### Mô tả
- **Mô hình**: Small CNN (3 conv layers + 2 FC layers)
- **Dataset**: CIFAR-10 (32x32 color images, 10 classes)
- **Số tham số**: ~588,042
- **Epochs**: 100
- **Batch size**: 128
- **Learning rate**: 0.001
- **Data augmentation**: Random crop, horizontal flip

### Chạy thực nghiệm

```bash
cd "CNN trên CIFAR-10"
python cnn_cifar10.py
```

### Kết quả mong đợi
- Adam: ~75-78% test accuracy
- Adam+SAM: ~77-80% test accuracy (cải thiện ~2-3%)

## 📈 Kết quả và Biểu đồ

Mỗi thực nghiệm sẽ tự động:
1. Tải và xử lý dữ liệu
2. Huấn luyện mô hình với Adam
3. Huấn luyện mô hình với Adam+SAM
4. Tạo biểu đồ so sánh (lưu dưới dạng PNG)
5. In kết quả chi tiết ra console

### Các biểu đồ được tạo ra:
- `logistic_regression_comparison.png` - Thực nghiệm 1
- `mlp_comparison.png` - Thực nghiệm 2
- `cnn_cifar10_comparison.png` - Thực nghiệm 3

Mỗi biểu đồ bao gồm 4 subplot:
- Training Loss
- Test Loss
- Training Accuracy
- Test Accuracy

## 🔬 Về SAM (Sharpness-Aware Minimization)

SAM là một kỹ thuật tối ưu giúp cải thiện khả năng tổng quát hóa của mô hình bằng cách:
- Tìm các vùng "phẳng" trong không gian tham số (flat minima)
- Thực hiện 2 lần forward-backward pass mỗi iteration
- Cải thiện độ chính xác trên tập test mà không overfitting

**Trade-off**: Thời gian huấn luyện tăng gấp ~2 lần so với Adam thông thường.

## 📝 Tham số SAM

- `rho`: 0.05 (default) - Bán kính neighborhood để tìm adversarial perturbation
- `adaptive`: False - Có sử dụng adaptive SAM hay không

Bạn có thể thay đổi các tham số này trong code để thử nghiệm.

## 🎯 Mục tiêu So sánh

1. **Accuracy**: Adam+SAM thường đạt accuracy cao hơn
2. **Generalization**: Adam+SAM có test loss thấp hơn, giảm overfitting
3. **Training time**: Adam+SAM chậm hơn ~2x do double forward-backward
4. **Stability**: Adam+SAM thường có đường training ổn định hơn

## 💡 Tips

1. **GPU**: Nếu có GPU, thời gian chạy sẽ nhanh hơn đáng kể
2. **Data**: Dữ liệu sẽ được tự động tải xuống vào thư mục `./data`
3. **Reproducibility**: Đã set seed=42 cho tất cả các thực nghiệm
4. **Memory**: CNN trên CIFAR-10 cần nhiều RAM/VRAM nhất

## 📚 Tài liệu tham khảo

- [Sharpness-Aware Minimization Paper](https://arxiv.org/abs/2010.01412)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

## 🐛 Xử lý lỗi thường gặp

### Lỗi CUDA out of memory
```bash
# Giảm batch_size trong code (dòng batch_size = 128 -> 64)
```

### Lỗi tải dataset
```bash
# Thử tải thủ công hoặc kiểm tra kết nối internet
# Dataset sẽ được lưu trong thư mục ./data
```

### Lỗi matplotlib
```bash
pip install --upgrade matplotlib
```

## 📧 Liên hệ

Nếu có vấn đề khi chạy code, hãy kiểm tra:
1. Đã cài đặt đúng thư viện chưa
2. Python version >= 3.8
3. Có đủ disk space cho dataset chưa (MNIST ~50MB, CIFAR-10 ~170MB)

---

**Chúc bạn thực nghiệm thành công! 🎉**
