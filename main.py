#!/usr/bin/env python3
"""Main application file for the NeuTTS-Air TTS service."""

import os
import uvicorn
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from core.logging import app_logger
from core.error_handler import register_error_handlers
from api.tts import router as tts_router, get_tts_model
from api.voice_profiles import router as voice_profiles_router
from core.settings import settings

# Set phonemizer environment variables if specified in settings
if settings.phonemizer_espeak_path:
    os.environ['PHONEMIZER_ESPEAK_PATH'] = settings.phonemizer_espeak_path
if settings.phonemizer_espeak_library:
    os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = settings.phonemizer_espeak_library


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager for the FastAPI application."""
    app_logger.info("Starting NeuTTS-Air TTS service initialization...")
    
    # Initialize the TTS model at startup
    try:
        app_logger.info("Loading TTS model...")
        model_loading_start = asyncio.get_event_loop().time()
        tts_model = get_tts_model()
        model_loading_time = asyncio.get_event_loop().time() - model_loading_start
        
        if tts_model:
            app_logger.info(f"TTS model loaded successfully in {model_loading_time:.2f} seconds")
            app_logger.info(f"Backend: {'vLLM' if settings.use_vllm else 'GGUF' if settings.use_gguf else 'Transformers'}")
        else:
            app_logger.error("Failed to load TTS model")
    except Exception as e:
        app_logger.error(f"Error during TTS model loading: {e}")
    
    yield
    
    app_logger.info("Shutting down NeuTTS-Air TTS service...")


# Create FastAPI app with lifespan
app = FastAPI(
    title="NeuTTS-Air Text-to-Speech Service",
    description="A production-ready text-to-speech service using NeuTTS-Air and NeuCodec",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Register error handlers
register_error_handlers(app)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(tts_router)
app.include_router(voice_profiles_router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "NeuTTS-Air Text-to-Speech Service is running",
        "version": "1.0.0",
        "backend": "vLLM" if settings.use_vllm else "GGUF" if settings.use_gguf else "Transformers"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    tts_model = get_tts_model()
    return {
        "status": "healthy" if tts_model is not None else "degraded",
        "model_loaded": tts_model is not None
    }


if __name__ == "__main__":
    app_logger.info(f"Starting NeuTTS-Air TTS service on {settings.host}:{settings.port}")
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    )
