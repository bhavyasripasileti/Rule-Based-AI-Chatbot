"""
main.py — FastAPI application entry point.

This file:
  1. Creates the FastAPI app instance
  2. Configures CORS (Cross-Origin Resource Sharing)
  3. Sets up logging
  4. Mounts all routers
  5. Adds a health check endpoint
  6. Configures startup/shutdown events

WHY IS STARTUP CONFIGURATION IMPORTANT?
  In production (Render), the app starts once and handles many requests.
  Startup events let us validate configuration early (e.g. check GROQ_API_KEY
  exists) rather than failing on the first real request.
"""

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.extract import router as extract_router

# ── Load environment variables from .env ──────────────────────────────────
# This must happen BEFORE anything that reads os.getenv()
# In production (Render), variables are set in the dashboard — load_dotenv
# is a no-op when the variable is already in the environment.
load_dotenv()

# ── Logging configuration ─────────────────────────────────────────────────
# Using basicConfig here means all loggers in the app use this format.
# In production, consider structured JSON logging (e.g. python-json-logger).
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Lifespan (startup + shutdown) ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs startup logic before the app starts accepting requests,
    and shutdown logic when the app is stopping.

    WHY USE LIFESPAN INSTEAD OF @app.on_event("startup")?
      @app.on_event is deprecated in newer FastAPI versions.
      Lifespan context managers are the current recommended approach.
    """
    # ── STARTUP ────────────────────────────────────────────────────────
    logger.info("=== LLM Extraction API starting up ===")

    # Validate critical environment variables at startup
    # Better to fail loudly here than silently on first request
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        logger.warning(
            "⚠️  GROQ_API_KEY is not set. "
            "The /extract endpoint will return 503 until it is configured."
        )
    else:
        logger.info("✓ GROQ_API_KEY found")

    logger.info("=== Startup complete. Ready to accept requests. ===")

    yield  # ← application runs here

    # ── SHUTDOWN ────────────────────────────────────────────────────────
    logger.info("=== LLM Extraction API shutting down ===")


# ── FastAPI app instance ──────────────────────────────────────────────────
app = FastAPI(
    title="LLM Field Extraction API",
    description=(
        "A production-quality REST API that extracts structured fields "
        "from unstructured text using an LLM (Groq / llama-3.3-70b-versatile). "
        "Returns per-field confidence scores and flags low-confidence fields "
        "for human review."
    ),
    version="1.0.0",
    lifespan=lifespan,
    # These control the OpenAPI docs available at /docs and /redoc
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS middleware ────────────────────────────────────────────────────────
# Allows the API to be called from browsers (e.g. a frontend demo).
# In production, restrict allow_origins to known frontend domains.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Permissive for assessment demo; lock down in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount routers ─────────────────────────────────────────────────────────
# All routes from extract.py are now available under /
# You could add a prefix here: prefix="/api/v1"
app.include_router(extract_router)


# ── Health check ──────────────────────────────────────────────────────────
@app.get(
    "/health",
    summary="Health check",
    description="Returns 200 if the API is running. Used by Render for uptime monitoring.",
    tags=["System"],
)
async def health_check() -> dict:
    """
    Simple health check endpoint.

    Render and other platforms ping this to verify the app is live.
    We keep it deliberately simple — just confirm the server is running.
    A more advanced version could also check Groq connectivity.
    """
    return {
        "status": "ok",
        "service": "llm-extraction-api",
        "version": "1.0.0",
    }


@app.get(
    "/",
    summary="Root",
    description="API root — redirects to documentation.",
    tags=["System"],
)
async def root() -> dict:
    """
    Root endpoint. Useful for confirming the deployment is live.
    """
    return {
        "message": "LLM Field Extraction API is running.",
        "docs": "/docs",
        "health": "/health",
        "extract": "POST /extract",
    }
