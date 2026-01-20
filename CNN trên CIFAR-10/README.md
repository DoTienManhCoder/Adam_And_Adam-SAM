# Thực nghiệm 3: CNN nhỏ trên CIFAR-10

## 📖 Mô tả

Thực nghiệm này so sánh hiệu suất của thuật toán Adam và Adam+SAM trên mô hình Convolutional Neural Network với dataset CIFAR-10.

## 🏗️ Kiến trúc mô hình

```
Input (3x32x32)
    ↓
Conv2d(3, 32, 3x3) -> BatchNorm -> ReLU -> MaxPool(2x2)    [32x16x16]
    ↓
Conv2d(32, 64, 3x3) -> BatchNorm -> ReLU -> MaxPool(2x2) -> Dropout2d(0.25)    [64x8x8]
    ↓
Conv2d(64, 128, 3x3) -> BatchNorm -> ReLU -> MaxPool(2x2) -> Dropout2d(0.25)   [128x4x4]
    ↓
Flatten [2048]
    ↓
Linear(2048, 256) -> ReLU -> Dropout(0.5)
    ↓
Linear(256, 10) -> Output (10 classes)
```

**Tổng số tham số**: ~588,042

## ⚙️ Cấu hình

- **Dataset**: CIFAR-10 (50,000 train, 10,000 test)
- **Input size**: 32x32x3 (color images)
- **Output classes**: 10 (plane, car, bird, cat, deer, dog, frog, horse, ship, truck)
- **Batch size**: 128
- **Epochs**: 100
- **Learning rate**: 0.001
- **Data Augmentation**: 
  - Random Crop (32x32 with padding=4)
  - Random Horizontal Flip
- **Normalization**: Mean=(0.4914, 0.4822, 0.4465), Std=(0.2023, 0.1994, 0.2010)
- **Optimizer**: Adam / Adam+SAM (rho=0.05)

## 🚀 Chạy thực nghiệm

```bash
python cnn_cifar10.py
```

⚠️ **Lưu ý**: Thực nghiệm này mất nhiều thời gian hơn (100 epochs)

## 📊 Kết quả đạt được

### Adam
- Training Accuracy: ~85-90%
- Test Accuracy: ~75-78%
- Best Test Accuracy: ~76-78%

### Adam + SAM
- Training Accuracy: ~82-87%
- Test Accuracy: ~77-80%
- Best Test Accuracy: ~78-81%
- Training time: ~20-30 phút (GPU) / ~4-6 giờ (CPU)
- **Cải thiện**: +2-3% test accuracy

## 📈 Biểu đồ

Sau khi chạy xong, file `cnn_cifar10_comparison.png` sẽ được tạo ra với 4 biểu đồ:
1. Training Loss
2. Test Loss
3. Training Accuracy
4. Test Accuracy

## 🔍 Quan sát

1. **Complex Dataset**: CIFAR-10 khó hơn MNIST, SAM cho thấy lợi ích rõ rệt hơn
2. **Overfitting**: Adam thường overfit hơn (train acc cao nhưng test acc thấp hơn)
3. **SAM Effect**: SAM giảm overfitting đáng kể, train acc thấp hơn nhưng test acc cao hơn
4. **Data Augmentation**: Kết hợp data augmentation với SAM cho kết quả tốt nhất
5. **Best Accuracy**: SAM thường đạt best test accuracy cao hơn 2-3%

## 💾 Output

- `cnn_cifar10_comparison.png`: Biểu đồ so sánh
- Console output: Chi tiết từng 10 epoch và kết quả cuối cùng
- `./data/cifar-10-batches-py`: Thư mục chứa dataset (tự động tải, ~170MB)

## 🎓 Ý nghĩa

CIFAR-10 là benchmark quan trọng trong computer vision. Thực nghiệm này cho thấy:
- SAM đặc biệt hiệu quả với CNN và dữ liệu phức tạp
- Trade-off giữa training accuracy và test accuracy
- Flat minima (do SAM tìm được) tổng quát hóa tốt hơn sharp minima

## 🔧 Tùy chỉnh

Bạn có thể thay đổi các tham số trong code:
- `epochs = 100` -> Tăng lên 150-200 để kết quả tốt hơn
- `learning_rate = 0.001` -> Thử learning rate decay
- `rho = 0.05` -> Thử rho = 0.1 hoặc 0.02
- Thêm conv layers để mô hình mạnh hơn

## 💡 Tips

1. **GPU recommended**: CNN huấn luyện rất chậm trên CPU
2. **Patience**: 100 epochs mất thời gian, có thể giảm xuống 50 để test nhanh
3. **Memory**: Cần ~2-3GB RAM/VRAM
4. **num_workers**: Đã set num_workers=2 cho DataLoader, có thể tăng nếu CPU mạnh

## 🏆 Benchmark

State-of-the-art trên CIFAR-10:
- Simple CNN: ~75-80%
- ResNet: ~90-95%
- Vision Transformer: ~95-98%

Mô hình này đạt ~78-80% với Adam+SAM là kết quả tốt cho small CNN!
