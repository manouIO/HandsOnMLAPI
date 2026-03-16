from fastapi import FastAPI, HTTPException
import torch
from .schemas import HousePredictionInput, PredictionOutput
from .model_arch import HousePriceNet
import os



app = FastAPI(title="HandsOn PyTorch API")

# 1. Initialize and Load Model
MODEL_PATH = "models/house_model_pytorch.pth"
model = HousePriceNet(input_size=5)  # We expect 5 features based on our schema

# Only load if the file exists to prevent startup crashes
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()
else:
    print(f"Warning: {MODEL_PATH} not found. Please run training first.")

# 2. Category Mapping
TYPE_MAPPING = {"apartment": 1, "villa": 2, "condo": 3}



@app.post("/predict",response_model=PredictionOutput)
async def predict(payload: HousePredictionInput):
    try:
        print(f"Received input: {payload}")
        # 1. Map Categorical data (House Type)
        type_idx = TYPE_MAPPING.get(payload.house_type.lower(), 0)
        print("type_idx:", type_idx)
        # 2. Map Boolean data (Garden)
        # Neurons only understand numbers, so True -> 1.0, False -> 0.0
        garden_idx = 1.0 if payload.has_garden else 0.0
        
        # 3. Scaling
        # We divide area by 1000 and distance by 10 to keep values small (best for NN)
        area_scaled = payload.area_sqft / 1000.0
        dist_scaled = payload.dist_to_center_km / 10.0
        
        # 4. The 5-Feature Tensor [Batch, Features]
        # Order matters! It must match your train_pytorch.py order exactly.
        input_data = [
            area_scaled, 
            float(payload.num_toilets), 
            float(type_idx), 
            dist_scaled, 
            garden_idx
        ]
        
        input_tensor = torch.tensor([input_data], dtype=torch.float32)
        
        # 5. Inference
        with torch.no_grad():
            prediction = model(input_tensor)
        
        estimated_price = float(prediction.item()) * 100000
        
        return {
            "house_type": payload.house_type,
            "estimated_price": f"{round(estimated_price, 2)} USD"

        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))