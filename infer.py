import torch
from PIL import Image
from torchvision import transforms
from tkinter import Tk, filedialog
from models import get_model  # Your EfficientNet-B0 model

# -------------------------------------------
# Load the model
# -------------------------------------------
def load_model(model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

# -------------------------------------------
# Load and preprocess image
# -------------------------------------------
def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)

# -------------------------------------------
# Predict with Confidence
# -------------------------------------------
def predict(image_path):
    print(f"\n🖼️ Selected Image: {image_path}")

    model, device = load_model("efficientnet_epoch_7.pth")  # Change to your best model file
    img = load_image(image_path).to(device)

    with torch.no_grad():
        output = model(img)
        probabilities = torch.softmax(output, dim=1)[0]
        confidence_real = probabilities[0].item() * 100
        confidence_fake = probabilities[1].item() * 100
        predicted_class = torch.argmax(probabilities).item()

    print("\n===============================")
    print("   🔥 DEEPFAKE PREDICTION 🔥")
    print("===============================\n")

    if predicted_class == 0:
        print(f"Result: REAL IMAGE ✅")
        print(f"Confidence: {confidence_real:.2f}%")
    else:
        print(f"Result: FAKE IMAGE ❌")
        print(f"Confidence: {confidence_fake:.2f}%")

    print("\n===============================\n")

# -------------------------------------------
# File Picker (GUI Dialog)
# -------------------------------------------
def ask_user_for_image():
    Tk().withdraw()  # Hide GUI window
    image_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    return image_path

# -------------------------------------------
# MAIN (GUI ONLY)
# -------------------------------------------
if __name__ == "__main__":
    print("\n📌 Opening image chooser...")

    image = ask_user_for_image()

    if image:
        predict(image)
    else:
        print("⚠️ No image selected. Exiting.\n")
