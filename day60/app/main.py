
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from app.schemas import CustomerInput, BatchPredictionInput
from app.model_service import (
    model,
    metadata,
    predict_customer,
    get_model_info,
    get_cache_info,
    clear_cache
)

app = FastAPI(
    title="AI-Powered Customer Churn Prediction",
    description=(
        "A machine learning application for predicting "
        "customer churn using a Random Forest model."
    ),
    version="1.0.0"
)


@app.get("/")
def home():
    return {
        "message": "AI-Powered Customer Churn API is running",
        "model_version": metadata["version"],
        "documentation": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_version": metadata["version"]
    }


@app.get("/model-info")
def model_info():
    return get_model_info()


@app.post("/predict")
def predict(customer: CustomerInput):
    try:
        result = predict_customer(customer)

        return {
            "success": True,
            "result": result
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(error)}"
        )


@app.post("/predict/batch")
def batch_predict(request: BatchPredictionInput):
    try:
        results = [
            predict_customer(customer)
            for customer in request.customers
        ]

        return {
            "success": True,
            "total_predictions": len(results),
            "model_version": metadata["version"],
            "results": results
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction error: {str(error)}"
        )


@app.get("/cache-info")
def cache_info():
    return get_cache_info()


@app.post("/cache/clear")
def cache_clear():
    clear_cache()

    return {
        "success": True,
        "message": "Prediction cache cleared."
    }


@app.exception_handler(Exception)
async def unexpected_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "message": "An unexpected error occurred."
        }
    )
