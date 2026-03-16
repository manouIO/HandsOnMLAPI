import torch
import torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

# 1. Define model architecture
class MoonModelV0(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers=nn.Sequential(nn.Linear(in_features=2,out_features=10),
                                  nn.ReLU(),
                                  nn.Linear(in_features=10,out_features=10),
                                  nn.ReLU(),
                                  nn.Linear(in_features=10,out_features=2) )

        
    def forward(self, x:torch.Tensor)->torch.Tensor:
       return self.layers(x)
    
# 2. Initialize FastAPI and load the model
app = FastAPI()
device = "cuda" if torch.cuda.is_available() else "cpu"

model = MoonModelV0().to(device)
# Ensure you have exported your weights from the notebook using torch.save(model.state_dict(), "model.pth")
model.load_state_dict(torch.load("models/model_0_classification.pth", map_location=device))
model.eval()

# 3. Define the data format for incoming requests
class DataInput(BaseModel):
    x1: float
    x2: float

@app.post("/predict")
async def predict(data: DataInput):
    # Convert input to tensor
    input_tensor = torch.tensor([[data.x1, data.x2]], dtype=torch.float).to(device)
    
    # Run inference
    with torch.inference_mode():
        logits = model(input_tensor)
        # Apply softmax to get probabilities and then argmax to get predicted class
        probs = torch.softmax(logits,dim=1)
        prediction = probs.argmax(dim=1)
    
    return {
        "prediction": int(prediction.item()),
        "probability": float(probs.max().item())
    }

