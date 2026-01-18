import torch
import pytest
from logistic_regression_mnist import LogisticRegressionModel, train_epoch

def test_logistic_regression_training():
    model = LogisticRegressionModel()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    
    # Mock data
    inputs = torch.randn(10, 784)  # 10 samples, 784 features (28x28 images flattened)
    labels = torch.randint(0, 10, (10,))  # 10 samples, 10 classes

    # Training step
    model.train()
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    assert loss.item() >= 0  # Loss should be non-negative

def test_optimizer_gradient_handling():
    model = LogisticRegressionModel()
    optimizer = torch.optim.Adam(model.parameters())
    
    # Mock data
    inputs = torch.randn(10, 784)
    labels = torch.randint(0, 10, (10,))

    model.train()
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = torch.nn.CrossEntropyLoss()(outputs, labels)
    loss.backward()

    for param in model.parameters():
        assert param.grad is not None  # Ensure gradients are computed

def test_optimizer_behavior():
    model = LogisticRegressionModel()
    optimizer = torch.optim.Adam(model.parameters())
    
    initial_params = [p.clone() for p in model.parameters()]
    
    # Mock data
    inputs = torch.randn(10, 784)
    labels = torch.randint(0, 10, (10,))

    model.train()
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = torch.nn.CrossEntropyLoss()(outputs, labels)
    loss.backward()
    optimizer.step()

    for initial_param, param in zip(initial_params, model.parameters()):
        assert not torch.equal(initial_param, param)  # Parameters should change after optimization