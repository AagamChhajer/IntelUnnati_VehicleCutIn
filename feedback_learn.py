import torch.optim as optim

# Define loss function and optimizer
loss_function = nn.CrossEntropyLoss()
adam_optimizer = optim.Adam(lane_change_model.parameters(), lr=0.001)

# Training loop
total_epochs = 20
for epoch in range(total_epochs):
    lane_change_model.train()
    for batch_images, batch_labels in train_loader:
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)
        
        # Forward pass
        predictions = lane_change_model(batch_images)
        loss = loss_function(predictions, batch_labels)
        
        # Backward pass and optimization
        adam_optimizer.zero_grad()
        loss.backward()
        adam_optimizer.step()
    
    print(f"Epoch [{epoch+1}/{total_epochs}], Loss: {loss.item():.4f}")
