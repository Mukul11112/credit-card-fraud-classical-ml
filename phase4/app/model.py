import os

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLASSES = 10

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")

elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")

else:
    DEVICE = torch.device("cpu")


# ============================================================
# MODEL PATH
# ============================================================

# Project structure:
#
# phase1-data-engineering/
#
# ├── phase3/
# │   ├── data/
# │   ├── notebooks/
# │   └── results/
# │       └── day13_14_resnet18_best.pth
# │
# └── phase4/
#     └── app/
#         └── model.py
#
#
# When running locally:
# model.py -> phase4/app/
# ../../phase3/results/...
#
# When running inside Docker:
# /app/phase4/app/model.py
# /app/phase3/results/...
#
# We automatically detect which environment is being used.


PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../../"
    )
)


LOCAL_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "phase3",
    "results",
    "day13_14_resnet18_best.pth"
)


DOCKER_MODEL_PATH = (
    "/app/phase3/results/day13_14_resnet18_best.pth"
)


if os.path.exists(DOCKER_MODEL_PATH):
    MODEL_PATH = DOCKER_MODEL_PATH

else:
    MODEL_PATH = LOCAL_MODEL_PATH


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

# IMPORTANT:
#
# These preprocessing steps match the preprocessing
# used during your ResNet18 training.
#
# Original CIFAR-10:
# 32 x 32
#
# ResNet18 input:
# 224 x 224
#
# ImageNet normalization:
# mean = [0.485, 0.456, 0.406]
# std  = [0.229, 0.224, 0.225]

IMAGE_TRANSFORM = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# CREATE RESNET18
# ============================================================

def create_model():
    """
    Create the exact ResNet18 architecture used
    during the transfer-learning experiment.
    """

    model = models.resnet18(
        weights=None
    )

    # Original ResNet18:
    #
    # fc -> 1000 ImageNet classes
    #
    # Your model:
    #
    # fc -> 10 CIFAR-10 classes

    model.fc = nn.Linear(
        model.fc.in_features,
        NUM_CLASSES
    )

    return model


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

def load_model():

    print(
        f"Loading model from:\n{MODEL_PATH}"
    )

    # --------------------------------------------------------
    # Check model file
    # --------------------------------------------------------

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            "\n"
            "==================================================\n"
            "MODEL FILE NOT FOUND\n"
            "==================================================\n"
            f"Expected model at:\n{MODEL_PATH}\n"
            "\n"
            "Make sure this file exists:\n"
            "phase3/results/day13_14_resnet18_best.pth\n"
            "==================================================\n"
        )

    # --------------------------------------------------------
    # Create architecture
    # --------------------------------------------------------

    model = create_model()

    # --------------------------------------------------------
    # Load trained weights
    # --------------------------------------------------------

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    # Your training code saved:
    #
    # torch.save(
    #     model.state_dict(),
    #     MODEL_PATH
    # )
    #
    # Therefore checkpoint should be a state_dict.

    if isinstance(checkpoint, dict):

        # Normal state_dict
        model.load_state_dict(
            checkpoint
        )

    else:

        raise RuntimeError(
            "The model file does not contain "
            "a valid PyTorch state_dict."
        )

    # --------------------------------------------------------
    # Move model to device
    # --------------------------------------------------------

    model = model.to(
        DEVICE
    )

    # --------------------------------------------------------
    # Evaluation mode
    # --------------------------------------------------------

    model.eval()

    print(
        "Model loaded successfully."
    )

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Classes: {NUM_CLASSES}"
    )

    return model


# ============================================================
# LOAD MODEL ONCE
# ============================================================

model = load_model()


# ============================================================
# PREDICTION
# ============================================================

def predict(image):
    """
    Predict the class of a CIFAR-10 image.

    Parameters
    ----------
    image : PIL.Image.Image
        Input image.

    Returns
    -------
    dict
        {
            "class_name": str,
            "confidence": float
        }
    """

    # --------------------------------------------------------
    # Validate image
    # --------------------------------------------------------

    if image is None:

        raise ValueError(
            "Image cannot be None."
        )

    # --------------------------------------------------------
    # Convert to RGB
    # --------------------------------------------------------

    image = image.convert(
        "RGB"
    )

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    input_tensor = IMAGE_TRANSFORM(
        image
    )

    # --------------------------------------------------------
    # Add batch dimension
    #
    # [3, 224, 224]
    #
    # becomes
    #
    # [1, 3, 224, 224]
    # --------------------------------------------------------

    input_tensor = input_tensor.unsqueeze(
        0
    )

    # --------------------------------------------------------
    # Move input to same device as model
    # --------------------------------------------------------

    input_tensor = input_tensor.to(
        DEVICE
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(
            input_tensor
        )

        # Convert logits to probabilities

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        # Get highest probability

        confidence, predicted_index = torch.max(
            probabilities,
            dim=1
        )

    # --------------------------------------------------------
    # Convert prediction to class
    # --------------------------------------------------------

    predicted_index = (
        predicted_index.item()
    )

    confidence = (
        confidence.item()
    )

    class_name = CLASS_NAMES[
        predicted_index
    ]

    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {
        "class_name": class_name,

        "confidence": round(
            confidence * 100,
            2
        )
    }


# ============================================================
# OPTIONAL TEST
# ============================================================

if __name__ == "__main__":

    print()
    print(
        "=============================================="
    )
    print(
        "ResNet18 Model Test"
    )
    print(
        "=============================================="
    )

    print(
        f"Model path: {MODEL_PATH}"
    )

    print(
        f"Device: {DEVICE}"
    )

    print(
        "Model is ready."
    )

    print(
        "=============================================="
    )