import torch
import torch.nn as nn
import torch.optim as optim
from model_arch import HousePriceNet 


# 1. Setup Data
# Features: [Area/1000, Toilets, TypeCode, DistCenter/10, HasGarden]
# TypeCode: 1: apartment, 2: villa, 3: condo
X = torch.tensor([
    [0.6, 1.0, 1.0, 0.2, 0.0], [1.2, 2.0, 1.0, 1.0, 0.0], [1.8, 3.0, 2.0, 1.5, 1.0],
    [2.5, 4.0, 2.0, 0.5, 1.0], [0.8, 1.0, 3.0, 0.3, 0.0], [1.5, 2.0, 3.0, 0.8, 1.0],
    [1.1, 1.0, 3.0, 1.6, 0.0], [1.7, 5.0, 2.0, 0.4, 1.0], [2.2, 2.0, 3.0, 0.1, 1.0],
    [1.0, 1.0, 1.0, 1.3, 0.0], [0.8, 2.0, 1.0, 1.1, 0.0], [1.4, 2.0, 1.0, 0.1, 0.0],
    [1.2, 3.0, 3.0, 0.3, 1.0], [1.4, 2.0, 3.0, 1.7, 1.0], [1.8, 2.0, 2.0, 0.7, 1.0],
    [1.1, 1.0, 1.0, 0.3, 0.0], [2.1, 3.0, 3.0, 1.9, 0.0], [1.8, 2.0, 3.0, 0.7, 0.0],
    [2.3, 3.0, 2.0, 1.6, 1.0], [3.1, 4.0, 2.0, 0.6, 1.0], [1.0, 2.0, 1.0, 0.6, 0.0],
    [2.5, 5.0, 2.0, 1.3, 1.0], [1.2, 2.0, 1.0, 0.2, 0.0], [3.7, 4.0, 2.0, 1.9, 1.0],
    [0.9, 1.0, 3.0, 1.0, 0.0], [2.1, 2.0, 3.0, 0.2, 0.0], [0.6, 2.0, 1.0, 0.4, 0.0],
    [2.6, 3.0, 2.0, 1.3, 1.0], [0.7, 2.0, 1.0, 0.2, 0.0], [3.3, 5.0, 2.0, 1.5, 1.0],
    [1.7, 3.0, 3.0, 0.8, 0.0], [1.9, 4.0, 2.0, 0.3, 1.0], [3.7, 2.0, 2.0, 1.3, 1.0],
    [0.8, 1.0, 1.0, 1.2, 0.0], [1.3, 2.0, 1.0, 0.2, 0.0], [1.2, 2.0, 1.0, 1.0, 0.0],
    [2.7, 5.0, 2.0, 0.4, 1.0], [1.0, 1.0, 1.0, 0.7, 1.0], [3.4, 5.0, 2.0, 0.3, 1.0],
    [1.0, 2.0, 3.0, 1.8, 0.0], [0.5, 2.0, 1.0, 1.0, 0.0], [2.4, 2.0, 2.0, 1.0, 1.0],
    [1.8, 2.0, 3.0, 1.1, 0.0], [2.1, 3.0, 3.0, 0.4, 0.0], [1.0, 2.0, 1.0, 1.1, 0.0],
    [2.3, 2.0, 3.0, 1.7, 0.0], [1.3, 2.0, 1.0, 0.2, 0.0], [1.9, 2.0, 3.0, 1.6, 0.0],
    [1.4, 3.0, 3.0, 1.9, 1.0], [1.3, 1.0, 3.0, 1.6, 0.0], [2.8, 2.0, 2.0, 0.2, 1.0],
    [3.1, 4.0, 2.0, 1.2, 1.0], [1.4, 1.0, 3.0, 1.1, 0.0], [3.4, 4.0, 2.0, 1.1, 1.0],
    [0.6, 1.0, 1.0, 0.4, 0.0], [1.5, 3.0, 3.0, 0.4, 0.0]
], dtype=torch.float32)

print(f"New Database Shape: {X.shape}")
# Prices in thousands (Matches the 1000x multiplier in your API)

# 2. Setup Labels (y) - Prices in thousands
# Shape: (56, 1)
y = torch.tensor([
    [60.0], [110.0], [210.0], [350.0], [90.0], [160.0], [115.5], [350.5], [317.5], 
    [95.5], [99.5], [171.5], [239.5], [217.5], [295.5], [120.0], [246.0], [215.5], 
    [349.5], [460.5], [126.0], [413.0], [151.0], [498.0], [105.5], [251.5], [91.0], 
    [382.5], [103.5], [486.0], [224.5], [351.0], [467.0], [78.0], [160.5], [139.0], 
    [445.5], [159.5], [513.5], [123.0], [72.5], [348.0], [209.5], [268.5], [118.5], 
    [248.0], [160.5], [211.5], [234.5], [134.5], [398.0], [451.5], [151.5], [481.5], 
    [71.0], [211.5]
], dtype=torch.float32)

y_scaled = y / 100.0 # scale down so that our model can learn better
model = HousePriceNet(input_size=5) 
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_function = nn.MSELoss()

# 3. Training Loop
print("Training started...")
epochs=2000
for epoch in range(epochs):
# 0. Put model in train mode
    model.train()
    
    # 1. Forward pass
    y_pred=model(X)

    # 2. Calculate loss
    loss =loss_function(y_pred,y_scaled)
   
    # 3. Zero gradients
    optimizer.zero_grad()
    
    # 4.Loss backward ( Backpropagation)
    loss.backward()
    
    # 5. Step the optimizer
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/2000], Loss: {loss.item():.4f}")

# 4. Save to the models folder
torch.save(model.state_dict(), "models/house_model_pytorch.pth")
print("\nSaved PyTorch model weights (5 features) to models/ folder!")

model.eval()
with torch.inference_mode():
    sample_pred = model(X[0:5])*100.0 # Predict first 5 houses
    print(f"Actual Prices: {y[0:5].flatten()}")
    print(f"Predicted Prices: {sample_pred.flatten()}")