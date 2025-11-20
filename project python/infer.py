import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
from models import get_model
from torchvision.models import resnet18, ResNet18_Weights

def predict(image_path, model_path="model_epoch_5.pth"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    model = get_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Image transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # If folder is given, predict for all images
    if os.path.isdir(image_path):
        image_files = [os.path.join(image_path, f) for f in os.listdir(image_path) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for img_file in image_files:
            classify_image(img_file, model, transform, device)
    else:
        classify_image(image_path, model, transform, device)

def classify_image(img_path, model, transform, device):
    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)
        confidence = confidence.item() * 100

        real_conf = probs[0][0].item() * 100
        fake_conf = probs[0][1].item() * 100

    label = "REAL" if pred.item() == 0 else "FAKE"
    print(f"\n🖼️ Image: {os.path.basename(img_path)}")
    print(f"Prediction: {label}")
    print(f"Confidence → Real: {real_conf:.2f}% | Fake: {fake_conf:.2f}%")

if __name__ == "__main__":
    path = input("Enter image or folder path: ").strip()
    predict(path)
