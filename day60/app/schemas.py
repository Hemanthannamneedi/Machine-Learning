
from pydantic import BaseModel, Field


class CustomerInput(BaseModel):
    tenure: float = Field(ge=0, le=1000)
    monthly_charges: float = Field(ge=0, le=100000)
    support_calls: float = Field(ge=0, le=10000)
    contract: str = Field(min_length=1, max_length=50)


class BatchPredictionInput(BaseModel):
    customers: list[CustomerInput] = Field(
        min_length=1,
        max_length=100
    )
