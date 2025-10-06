from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import os

app = FastAPI(title="Feature Store")

@app.get("/feature/{detector}/{image_name}")
def get_feature(detector: str, image_name: str):
    path = f"/data/features/{detector}/{image_name}.npz"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="feature not found")
    # Option 1: return path (lightweight) - recommended
    return JSONResponse({"feature_path": path})
    # Option 2: stream binary (if you want)
    # return FileResponse(path, media_type="application/octet-stream")
