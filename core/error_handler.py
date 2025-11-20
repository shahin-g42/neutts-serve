"""
Global error handler for FastAPI application.
"""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from core.exceptions import TTSServiceError
from core.logging import app_logger


def register_error_handlers(app: FastAPI) -> None:
    """Register global error handlers for the FastAPI application."""
    
    @app.exception_handler(TTSServiceError)
    async def tts_service_error_handler(request: Request, exc: TTSServiceError):
        """Handle TTSServiceError exceptions."""
        app_logger.error(f"TTSServiceError: {exc.error_code.value} - {exc.message}")
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_dict()
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions."""
        app_logger.exception(f"Unhandled exception: {str(exc)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": "internal_error",
                    "message": "An internal error occurred",
                    "detail": str(exc) if app.debug else "Internal server error"
                }
            }
        )
