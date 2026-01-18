# Thực nghiệm 1: Logistic Regression trên MNIST

## 📖 Mô tả

Thực nghiệm này so sánh hiệu suất của thuật toán Adam và Adam+SAM trên mô hình Logistic Regression đơn giản với dataset MNIST.

## 🏗️ Kiến trúc mô hình

```
Input (784) -> Linear(784, 10) -> Output (10 classes)
```

**Tổng số tham số**: ~7,850

## ⚙️ Cấu hình

- **Dataset**: MNIST (60,000 train, 10,000 test)
- **Input size**: 28x28 = 784
- **Output classes**: 10 (digits 0-9)
- **Batch size**: 128
- **Epochs**: 50
- **Learning rate**: 0.001
- **Optimizer**: Adam / Adam+SAM (rho=0.05)

## 🚀 Chạy thực nghiệm

```bash
python logistic_regression_mnist.py
```

## 📊 Kết quả mong đợi

### Adam
- Training Accuracy: ~93-94%
- Test Accuracy: ~92-93%
- Training time: ~2-3 phút (CPU) / ~30s (GPU)

### Adam + SAM
- Training Accuracy: ~94-95%
- Test Accuracy: ~93-94%
- Training time: ~4-6 phút (CPU) / ~1 phút (GPU)
- **Cải thiện**: +1-1.5% test accuracy

## 📈 Biểu đồ

Sau khi chạy xong, file `logistic_regression_comparison.png` sẽ được tạo ra với 4 biểu đồ:
1. Training Loss
2. Test Loss
3. Training Accuracy
4. Test Accuracy

## 🔍 Quan sát

1. **Convergence**: Adam+SAM hội tụ chậm hơn nhưng ổn định hơn
2. **Generalization**: Test accuracy của SAM cao hơn, cho thấy khả năng tổng quát tốt hơn
3. **Overfitting**: SAM giảm overfitting so với Adam
4. **Trade-off**: Thời gian training tăng ~2x

## 💾 Output

- `logistic_regression_comparison.png`: Biểu đồ so sánh
- Console output: Chi tiết từng epoch và kết quả cuối cùng
- `./data/MNIST`: Thư mục chứa dataset (tự động tải)

## 🎓 Ý nghĩa

Thực nghiệm này cho thấy ngay cả với mô hình đơn giản như Logistic Regression, SAM vẫn có thể cải thiện khả năng tổng quát hóa. Điều này đặc biệt hữu ích khi làm việc với dữ liệu hạn chế hoặc cần độ chính xác cao.
