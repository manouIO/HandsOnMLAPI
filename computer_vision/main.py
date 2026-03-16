import torch
import io
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from PIL import Image, ImageOps
from torchvision import transforms
from computer_vision.model_arch import FashionMNISTModelV2 , class_names# Import your specific architecture
from torch import Tensor
from pathlib import Path
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/", response_class=HTMLResponse)
async def main():
    content = """
    <html>
        <head>
            <style>
                body { font-family: sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; background: #f0f2f5; }
                .card { background: white; padding: 2rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
                h2 { color: #1a73e8; }
                input { margin: 10px 0; }
                button { background: #1a73e8; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; }
            </style>
        </head>
        <body>
            <div class="card">
                <h2>FashionMNIST Predictor</h2>
                <p>Upload a photo of a clothing item (max 3MB)</p>
                <form action="/predict_clothing" enctype="multipart/form-data" method="post">
                    <input name="file" type="file" accept="image/png, image/jpeg, image/webp">
                    <br>
                    <button type="submit">Predict Item</button>
                </form>
            </div>
        </body>
    </html>
    """
    return content

# 1. Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = FashionMNISTModelV2(input_shape=1, hidden_units=10, output_shape=len(class_names)).to(device)
model.load_state_dict(torch.load("/home/maryse/Documents/Training/HandsOnMLAPI/models/03_pytorch_computer_vision_model_2.pth", map_location=device))
model.to(device)
model.eval()

# 2. Define the Preprocessing Transform
# Model expects: Grayscale, 28x28 pixels, and a Normalized Tensor
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])



# Define your limit (3MB)
MIN_FILE_SIZE = 1 * 500 # 0.5 KB
MAX_FILE_SIZE = 3 * 1024 * 1024 #5 mb*1024kb*1024bytes

@app.post("/predict_clothing")
async def predict_clothing(file: UploadFile = File(...)):
    # 1. Check File Size first
    file_size = file.size if file.size is not None else 0
    if file_size < MIN_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File is empty or too small to be a valid image.")
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE / (1024*1024)} MB."
        )
    # 2. Validate file extension
    allowed_extensions = ["jpg", "jpeg", "png", "webp"]
    
    filename = file.filename or ""  # Fallback to empty string if None
    file_ext = Path(filename).suffix.lower().replace(".", "")

    if file_ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Unsupported file extension")
    
    # 3. Try to open the image (handles corrupted files)
    start_time = time.perf_counter()
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not decode image. The file might be corrupted."
        )

    # Inside your predict function:
    image = Image.open(io.BytesIO(contents)).convert('L')
    image = ImageOps.invert(image) # Flip so background is black

    # --- NEW: Print size before transforming ---
    # PIL returns (Width, Height)
    print(f"DEBUG: Original image size (PIL): {image.size}") 

    # 2. Preprocess the image
    img_tensor: Tensor = transform(image)  # type: ignore
    
    # --- NEW: Print size after transforming ---
    # PyTorch returns [Channels, Height, Width]
    print(f"DEBUG: Tensor size after transform: {img_tensor.shape}")

    img_tensor = img_tensor.unsqueeze(0) #type:ignore
    
    # Print size after adding batch dimension
    print(f"DEBUG: Final tensor size (with batch): {img_tensor.shape}")
    # 3. Run Inference
    with torch.inference_mode():
        logits = model(img_tensor)
        # Use Softmax to get probabilities (optional but helpful)
        probs = torch.softmax(logits, dim=1)
        # Use Argmax to get the class index
        pred_label = torch.argmax(probs, dim=1).item()

    print( probs, pred_label)
    end_time = time.perf_counter()
    duration = end_time - start_time #result in seconds
    
    return {
        "prediction": class_names[int(pred_label)],
        "confidence": round(float(probs[0][int(pred_label)].item()), 4),
        "inference_time_seconds": f"{round(duration*1000, 4)} ms"
    }