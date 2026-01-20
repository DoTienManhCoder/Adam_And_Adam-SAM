# Thực nghiệm 2: MLP trên MNIST

## 📖 Mô tả

Thực nghiệm này so sánh hiệu suất của thuật toán Adam và Adam+SAM trên mô hình Multi-Layer Perceptron với dataset MNIST.

## 🏗️ Kiến trúc mô hình

```
Input (784)
    ↓
Linear(784, 256) -> ReLU -> Dropout(0.2)
    ↓
Linear(256, 128) -> ReLU -> Dropout(0.2)
    ↓
Linear(128, 10) -> Output (10 classes)
```

**Tổng số tham số**: ~235,146

## ⚙️ Cấu hình

- **Dataset**: MNIST (60,000 train, 10,000 test)
- **Input size**: 28x28 = 784
- **Hidden layers**: [256, 128]
- **Output classes**: 10 (digits 0-9)
- **Batch size**: 128
- **Epochs**: 50
- **Learning rate**: 0.001
- **Dropout**: 0.2
- **Optimizer**: Adam / Adam+SAM (rho=0.05)

## 🚀 Chạy thực nghiệm

```bash
python mlp_mnist.py
```

## 📊 Kết quả đạt được

### Adam
- Training Accuracy: ~98-99%
- Test Accuracy: ~97.5-98%

### Adam + SAM
- Training Accuracy: ~98.5-99%
- Test Accuracy: ~98-98.5%
- **Cải thiện**: +0.5-1% test accuracy

## 📈 Biểu đồ

Sau khi chạy xong, file `mlp_comparison.png` sẽ được tạo ra với 4 biểu đồ:
1. Training Loss
2. Test Loss
3. Training Accuracy
4. Test Accuracy

## 🔍 Quan sát

1. **Deep Network**: MLP sâu hơn Logistic Regression, SAM giúp tránh overfitting tốt hơn
2. **Dropout Effect**: Kết hợp Dropout với SAM cho kết quả tổng quát hóa tốt nhất
3. **Training Stability**: SAM có training curve mượt và ổn định hơn
4. **Convergence Speed**: Adam hội tụ nhanh hơn, nhưng SAM đạt test accuracy cao hơn

## 💾 Output

- `mlp_comparison.png`: Biểu đồ so sánh
- Console output: Chi tiết từng epoch và kết quả cuối cùng
- `./data/MNIST`: Thư mục chứa dataset (tự động tải)

## 🎓 Ý nghĩa

MLP là mô hình phổ biến và dễ bị overfitting hơn Logistic Regression. Thực nghiệm này cho thấy SAM đặc biệt hiệu quả với mạng neural sâu hơn, giúp cải thiện đáng kể khả năng tổng quát hóa.

## 🔧 Tùy chỉnh

Bạn có thể thay đổi các tham số trong code:
- `hidden_dims = [256, 128]` -> Thay đổi số neurons
- `dropout = 0.2` -> Điều chỉnh dropout rate
- `epochs = 50` -> Tăng/giảm số epochs
- `rho = 0.05` -> Thay đổi SAM radius
