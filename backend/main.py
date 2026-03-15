# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="EdgeFinder API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/setups")
async def get_setups():
    # This will call run_scan and return results
    return {"setups": []}

@app.get("/api/health")
async def health():
    return {"status": "ok"}