def test_logistic_regression_training():
    import torch
    from logistic_regression_mnist import LogisticRegressionModel, train_epoch

    model = LogisticRegressionModel()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    device = 'cpu'
    
    # Dummy data
    train_loader = [(torch.randn(10, 28*28), torch.randint(0, 10, (10,)))]
    
    model.train()
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, use_sam=False)
    
    assert train_loss >= 0
    assert 0 <= train_acc <= 1