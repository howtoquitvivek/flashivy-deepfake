import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from src.model import DeepfakeModel

# --------------------
# Config
# --------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "weights/deepfake_v1.pth"
CLASS_NAMES = ["REAL", "FAKE"]

# --------------------
# Image preprocessing
# --------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------
# Load model
# --------------------
def load_model():
    model = DeepfakeModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

# --------------------
# Predict
# --------------------
def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    model = load_model()

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)

    label = CLASS_NAMES[pred.item()]
    confidence = conf.item()

    return label, confidence

# --------------------
# CLI
# --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deepfake image inference")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input face image"
    )

    args = parser.parse_args()

    label, confidence = predict(args.image)

    print(f"Prediction : {label}")
    print(f"Confidence : {confidence:.4f}")
