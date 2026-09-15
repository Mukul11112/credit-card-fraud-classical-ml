from io import BytesIO

from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image

from .model import predict
from .schemas import PredictionResponse


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="CIFAR-10 ResNet18 API",
    description="Image classification API using a ResNet18 transfer learning model.",
    version="1.0.0",
)


# ============================================================
# Root Endpoint
# ============================================================

@app.get("/")
def root():
    return {
        "message": "CIFAR-10 ResNet18 API is running",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


# ============================================================
# Health Check
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# ============================================================
# Prediction Endpoint
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict Image",
    description="Upload a CIFAR-10 image and get the predicted class and confidence.",
)
async def predict_image(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # Validate file
    # --------------------------------------------------------

    if file is None:
        raise HTTPException(
            status_code=400,
            detail="No image file provided"
        )

    # --------------------------------------------------------
    # Validate image type
    # --------------------------------------------------------

    allowed_types = {
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp",
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid file type. "
                "Please upload a JPEG, JPG, PNG, or WEBP image."
            )
        )

    # --------------------------------------------------------
    # Read uploaded file
    # --------------------------------------------------------

    try:
        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty"
            )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read uploaded file: {str(e)}"
        )

    # --------------------------------------------------------
    # Convert uploaded bytes to PIL Image
    # --------------------------------------------------------

    try:
        image = Image.open(
            BytesIO(contents)
        ).convert("RGB")

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file"
        )

    # --------------------------------------------------------
    # Run model prediction
    # --------------------------------------------------------

    try:

        result = predict(image)

        # ----------------------------------------------------
        # Validate model output
        # ----------------------------------------------------

        if not isinstance(result, dict):
            raise RuntimeError(
                "Model prediction must return a dictionary."
            )

        if "class_name" not in result:
            raise RuntimeError(
                "Model response does not contain 'class_name'."
            )

        if "confidence" not in result:
            raise RuntimeError(
                "Model response does not contain 'confidence'."
            )

        # ----------------------------------------------------
        # Get prediction values
        # ----------------------------------------------------

        class_name = result["class_name"]
        confidence = result["confidence"]

        # ----------------------------------------------------
        # Convert confidence to float
        # ----------------------------------------------------

        try:
            confidence = float(confidence)

        except (TypeError, ValueError):
            raise RuntimeError(
                "Model returned an invalid confidence value."
            )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Pydantic schema expects confidence between 0 and 1.
        #
        # If model returns:
        #
        #     66.37
        #
        # that means:
        #
        #     66.37%
        #
        # so convert it to:
        #
        #     0.6637
        # ----------------------------------------------------

        if confidence > 1.0:
            confidence = confidence / 100.0

        # ----------------------------------------------------
        # Protect against invalid values
        # ----------------------------------------------------

        confidence = max(0.0, min(1.0, confidence))

        # ----------------------------------------------------
        # Return successful prediction
        # ----------------------------------------------------

        return PredictionResponse(
            class_name=str(class_name),
            confidence=confidence
        )

    # --------------------------------------------------------
    # Preserve HTTP exceptions
    # --------------------------------------------------------

    except HTTPException:
        raise

    # --------------------------------------------------------
    # Catch model/prediction errors
    # --------------------------------------------------------

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )