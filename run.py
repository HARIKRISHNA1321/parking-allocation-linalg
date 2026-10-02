"""
Single-Command Entrypoint for Parking Allocation Web Application.
Runs the FastAPI server with Uvicorn, serving API routes and the built React frontend.
"""

import os
import sys
import uvicorn

if __name__ == "__main__":
    print("=" * 70)
    print("  Parking Allocation Using Linear Algebra - University Mini-Project")
    print("  Data: Stolfi (2017), UCI Parking Birmingham, CC BY 4.0")
    print("  FastAPI Server + React SPA Serving on http://127.0.0.1:8000")
    print("=" * 70)
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)
