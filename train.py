from config import *
from data import testloader, trainloader

model = Net()
model.to(device)  # Sending the Model to device for training

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Define a learning rate scheduler
scheduler = ReduceLROnPlateau(
    optimizer, mode="min", factor=0.1, patience=3, verbose=True
)

for epoch in range(NUM_EPOCHS):
    # Training
    model.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % LOG_INTERVAL == LOG_INTERVAL - 1:
            print(
                f"[{epoch + 1}, {i + 1}/{len(trainloader)}] loss: {running_loss / LOG_INTERVAL:.5f}"
            )
            running_loss = 0.0

    if ((epoch + 1) % VALIDATION_INTERVAL) == 0:
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"+++++" * 10)
        print("Finished Validation")
        print(f"Validation accuracy: {accuracy}%")
        print(f"+++++" * 10)

        # Learning rate scheduler step
        scheduler.step(running_loss)
print(f"+++++" * 10)
print("Finished Training")
print(f"+++++" * 10)
if os.path.isdir("./models"):
    pass
else:
    os.mkdir("./models")
torch.save(model.state_dict(), EXPORT_PATH)
