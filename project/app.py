"""
SIH26084 Convective Scale Nowcasting - Root Application Entry Point
Unified FastAPI microservice delegating to modular backend architecture.
Domain: Atmospheric Sciences & Convective Nowcasting (0-6 hr)
Sponsoring Organization: Ministry of Earth Sciences (MoES) / NCMRWF
"""

import os
import sys

# Ensure backend directory is in path
backend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from backend.app import app, model_manager

if __name__ == "__main__":
    import uvicorn
    import socket
    port = int(os.environ.get("PORT", 0))
    if not port:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            port = 8000 if s.connect_ex(('127.0.0.1', 8000)) != 0 else 8080
    print(f"Starting SIH26084 server on port {port}...")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
