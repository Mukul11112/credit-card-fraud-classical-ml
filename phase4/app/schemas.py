from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    class_name: str = Field(
        ...,
        description="Predicted CIFAR-10 class"
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Prediction confidence between 0 and 1"
    )
