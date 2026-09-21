from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from ml.model_store import load_churn_model
from fastapi.responses import JSONResponse
from core.errors import ChurnServiceError
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import logging
from core.state import model_state
from api.routes import router


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("churn_service")


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_state["bundle"] = load_churn_model()
    yield

app = FastAPI(lifespan=lifespan)
app.include_router(router)


@app.exception_handler(StarletteHTTPException)
def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "code": "HTTP_ERROR",
            "message": exc.detail,
            "details": {},
        },
    )


@app.exception_handler(ChurnServiceError)
def churn_error_handler(request: Request, exc: ChurnServiceError):
    logger.warning("ChurnServiceError: code=%s status=%s", exc.code, exc.status_code)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "code": exc.code,
            "message": exc.message,
            "details": exc.details,
        },
    )


@app.exception_handler(RequestValidationError)
def validation_error_handler(request: Request, exc: RequestValidationError):
    logger.warning("Validation error")
    return JSONResponse(
        status_code=422,
        content={
            "code": "VALIDATION_ERROR",
            "message": "Request validation failed.",
            "details": {"errors": exc.errors()},
        },
    )


@app.exception_handler(Exception)
def unhandled_error_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s", exc, exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={
            "code": "INTERNAL_ERROR",
            "message": "Internal server error.",
            "details": {},
        },
    )
