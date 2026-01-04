# =========================
# Step 1: Create Streamlit app & configure page
# =========================
import streamlit as st

st.set_page_config(
    page_title="Image Classification with ResNet18",
    layout="centered"
)

st.title("Image Classification with ResNet18")
st.write("Upload an image to classify it using a pre-trained ResNet18 model.")

# =========================
# Step 2: Import required libraries
# =========================
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
import pandas as pd

# =========================
# Step 3: Configure CPU-only execution
# =========================
device = torch.device("cpu")
torch.set_num_threads(1)

# =========================
# Step 4: Load pre-trained ResNet18 model
# =========================
weights = ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
model.eval()
model.to(device)

# =========================
# Step 5: Image preprocessing transforms
# =========================
preprocess = weights.transforms()

# Load ImageNet class labels
class_labels = weights.meta["categories"]

# =========================
# Step 6: User interface for image upload
# =========================
uploaded_file = st.file_uploader(
    "Upload an image (JPG or PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # =========================
    # Step 7: Convert image to tensor & run inference
    # =========================
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)

    # =========================
    # Step 8: Softmax & Top-5 predictions
    # =========================
    probabilities = F.softmax(outputs, dim=1)[0]
    top5_probs, top5_indices = torch.topk(probabilities, 5)

    results = []
    for prob, idx in zip(top5_probs, top5_indices):
        results.append({
            "Class": class_labels[idx],
            "Probability": float(prob)
        })

    df = pd.DataFrame(results)
    df["Probability"] = df["Probability"] * 100  # convert to %

    st.subheader("🔍 Top-5 Predictions")
    st.dataframe(df, use_container_width=True)

    # =========================
    # Step 9: Bar chart visualization
    # =========================
    st.subheader("📊 Prediction Probabilities")
    st.bar_chart(
        df.set_index("Class")["Probability"]
    )
