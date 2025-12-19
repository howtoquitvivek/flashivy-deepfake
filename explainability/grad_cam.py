import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from src.model import DeepfakeModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------
# Load model
# --------------------
model = DeepfakeModel().to(device)
model.load_state_dict(torch.load("weights/deepfake_v1.pth", map_location=device))
model.eval()

# --------------------
# Hook variables
# --------------------
gradients = None
activations = None

def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]

def forward_hook(module, input, output):
    global activations
    activations = output

# --------------------
# Register hooks
# --------------------
target_layer = model.model.conv_head
target_layer.register_forward_hook(forward_hook)
target_layer.register_backward_hook(backward_hook)

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
# Generate Grad-CAM for ONE image
# --------------------
def get_gradcam(image_path):
    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)

    output = model(input_tensor)
    class_idx = output.argmax(dim=1).item()

    model.zero_grad()
    output[0, class_idx].backward()

    grads = gradients.cpu().data.numpy()[0]
    acts = activations.cpu().data.numpy()[0]

    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    cam = cv2.resize(cam, (224, 224))

    img_np = np.array(img.resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

    return img_np, overlay

# --------------------
# Plot comparison
# --------------------
def plot_real_vs_fake(
    real_img_path,
    fake_img_path,
    output_dir="outputs/gradcam",
    filename="gradcam_comparison.png"
):
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)

    real_img, real_cam = get_gradcam(real_img_path)
    fake_img, fake_cam = get_gradcam(fake_img_path)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(real_img)
    plt.title("Real Image")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(real_cam)
    plt.title("Grad-CAM (Real)")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(fake_img)
    plt.title("Fake Image")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(fake_cam)
    plt.title("Grad-CAM (Fake)")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"Comparison saved to {save_path}")