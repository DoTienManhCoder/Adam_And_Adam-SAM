# So sánh Thuật toán Tối ưu: Adam vs Adam+SAM

Dự án này thực hiện 3 thực nghiệm để so sánh hiệu suất của thuật toán Adam với Adam kết hợp Sharpness-Aware Minimization (SAM).

## 📋 Mục lục

1. [Thực nghiệm 1: Logistic Regression trên MNIST](#thực-nghiệm-1)
2. [Thực nghiệm 2: MLP trên MNIST](#thực-nghiệm-2)
3. [Thực nghiệm 3: CNN nhỏ trên CIFAR-10](#thực-nghiệm-3)

## 🚀 Cài đặt

### Yêu cầu hệ thống
- Python 3.8 trở lên
- GPU NVIDIA với CUDA support (khuyến nghị để tăng tốc đáng kể)
- 4GB+ RAM (8GB+ khuyến nghị cho CNN)

### ⚠️ LƯU Ý QUAN TRỌNG VỀ GPU

**Vấn đề**: Nếu bạn có GPU NVIDIA nhưng code vẫn chạy trên CPU, nguyên nhân là bạn đã cài đặt **PyTorch phiên bản CPU** thay vì phiên bản CUDA.

**Kiểm tra GPU**:
```bash
# Kiểm tra xem PyTorch có nhận GPU không
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

Nếu hiển thị `CUDA available: False`, bạn cần cài đặt lại PyTorch với CUDA support.

### Cài đặt thư viện

#### Bước 1: Xác định phiên bản CUDA của GPU
```bash
nvidia-smi
```
Lệnh này sẽ hiển thị phiên bản CUDA (ví dụ: CUDA 12.8, 12.4, 11.8...)

#### Bước 2: Gỡ cài đặt PyTorch CPU (nếu đã cài)
```bash
pip uninstall torch torchvision torchaudio -y
```

#### Bước 3: Cài đặt PyTorch với CUDA support

**Cho CUDA 12.x** (RTX 30xx, 40xx, A100...):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**Cho CUDA 11.8** (GTX 16xx, RTX 20xx...):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Cài đặt các thư viện còn lại**:
```bash
pip install matplotlib numpy
```

**Hoặc dùng requirements.txt** (sau khi đã cài PyTorch CUDA):
```bash
pip install -r requirements.txt
```

### Kiểm tra cài đặt thành công
Sau khi cài đặt, chạy lệnh này để xác nhận GPU hoạt động:
```bash
python check_gpu.py
```

Kết quả mong đợi:
```
CUDA available: True
GPU name: NVIDIA GeForce RTX xxxx
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

**Nếu sử dụng Virtual Environment** (.venv):
```bash
# Kích hoạt virtual environment trước
.\.venv\Scripts\Activate.ps1  # Windows PowerShell
# hoặc
.venv\Scripts\activate.bat     # Windows CMD

# Sau đó chạy
cd "Logistic Regression trên MNIST"
python logistic_regression_mnist.py
```

**Hoặc dùng đường dẫn đầy đủ**:
```bash
cd "Logistic Regression trên MNIST"
C:/Users/<YourUsername>/Documents/GitHub/Adam_And_Adam-SAM/.venv/Scripts/python.exe logistic_regression_mnist.py
```

**Nếu không dùng Virtual Environment**:
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

**Nếu sử dụng Virtual Environment** (.venv):
```bash
# Kích hoạt virtual environment trước
.\.venv\Scripts\Activate.ps1  # Windows PowerShell

# Sau đó chạy
cd "MLP trên MNIST"
python mlp_mnist.py
```

**Hoặc dùng đường dẫn đầy đủ đến Python trong venv**:
```bash
cd "MLP trên MNIST"
C:/Users/<YourUsername>/Documents/GitHub/Adam_And_Adam-SAM/.venv/Scripts/python.exe mlp_mnist.py
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

**Nếu sử dụng Virtual Environment** (.venv):
```bash
# Kích hoạt virtual environment trước
.\.venv\Scripts\Activate.ps1  # Windows PowerShell

# Sau đó chạy
cd "CNN trên CIFAR-10"
python cnn_cifar10.py
```

**Hoặc dùng đường dẫn đầy đủ đến Python trong venv**:
```bash
cd "CNN trên CIFAR-10"
C:/Users/<YourUsername>/Documents/GitHub/Adam_And_Adam-SAM/.venv/Scripts/python.exe cnn_cifar10.py
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

1. **GPU**: 
   - **BẮT BUỘC** cài đặt PyTorch với CUDA support nếu có GPU NVIDIA
   - Kiểm tra bằng `nvidia-smi` và `python check_gpu.py`
   - Thời gian chạy nhanh hơn 10-50x so với CPU
   - Console phải hiển thị `Sử dụng device: cuda` khi chạy
2. **Virtual Environment**: 
   - Nếu dùng venv, nhớ kích hoạt bằng `.\.venv\Scripts\Activate.ps1`
   - Hoặc dùng đường dẫn đầy đủ: `.venv/Scripts/python.exe script.py`
3. **Data**: Dữ liệu sẽ được tự động tải xuống vào thư mục `./data`
4. **Reproducibility**: Đã set seed=42 cho tất cả các thực nghiệm
5. **Memory**: CNN trên CIFAR-10 cần nhiều RAM/VRAM nhất (~2-4GB VRAM)

## 📚 Tài liệu tham khảo

- [Sharpness-Aware Minimization Paper](https://arxiv.org/abs/2010.01412)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

## 🐛 Xử lý lỗi thường gặp

### ❌ Lỗi: Code chạy trên CPU thay vì GPU

**Triệu chứng**: Console hiển thị `Sử dụng device: cpu` thay vì `cuda`

**Nguyên nhân**: Đã cài đặt PyTorch phiên bản CPU (ví dụ: `2.9.1+cpu`) thay vì CUDA.

**Giải pháp**:
```bash
# 1. Kiểm tra xem GPU có được nhận diện không
nvidia-smi

# 2. Gỡ PyTorch CPU
pip uninstall torch torchvision torchaudio -y

# 3. Cài đặt PyTorch CUDA (phù hợp với phiên bản CUDA của bạn)
# Cho CUDA 12.x:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 4. Kiểm tra lại
python check_gpu.py
```

### ❌ Lỗi: Không chạy được bằng lệnh `python script.py`

**Triệu chứng**: Lỗi ModuleNotFoundError hoặc chạy sai Python version

**Nguyên nhân**: Đang dùng Virtual Environment nhưng chưa kích hoạt hoặc lệnh `python` toàn cục trỏ sai.

**Giải pháp**:

**Cách 1 - Kích hoạt Virtual Environment**:
```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Sau đó chạy bình thường
python logistic_regression_mnist.py
```

**Cách 2 - Dùng đường dẫn đầy đủ**:
```powershell
# Không cần kích hoạt venv
C:/Users/dotie/Documents/GitHub/Adam_And_Adam-SAM/.venv/Scripts/python.exe script.py
```

### Lỗi CUDA out of memory
```bash
# Giảm batch_size trong code (dòng batch_size = 128 -> 64 hoặc 32)
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
1. **Đã cài đúng PyTorch CUDA** (không phải CPU version) - Quan trọng nhất!
2. PyTorch version tương thích với CUDA driver của GPU
3. Đã kích hoạt virtual environment (nếu dùng venv)
4. Python version >= 3.8
5. Có đủ disk space cho dataset (MNIST ~50MB, CIFAR-10 ~170MB)
6. Có đủ VRAM trên GPU (tối thiểu 2GB cho CNN)

**Checklist nhanh trước khi chạy**:
```bash
# 1. Kiểm tra GPU
nvidia-smi
python check_gpu.py

# 2. Kích hoạt venv
.\.venv\Scripts\Activate.ps1

# 3. Chạy code
cd "Logistic Regression trên MNIST"
python logistic_regression_mnist.py
```

---

**Chúc bạn thực nghiệm thành công! 🎉**
