"""
Web UI Router serving the dashboard HTML.
"""

from pathlib import Path
from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from src.config.settings import get_settings

router = APIRouter()
settings = get_settings()


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
@router.get("/ui", response_class=HTMLResponse, include_in_schema=False)
async def serve_dashboard():
    """Serve the static HTML dashboard."""
    template_path = settings.project_root / "src" / "web" / "templates" / "index.html"
    if not template_path.exists():
        return HTMLResponse(
            "<h3>Dashboard template not found. Please verify project files.</h3>",
            status_code=404,
        )
    with open(template_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)
