from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np, json

app = FastAPI(title="Shift Transform")

class ShiftRequest(BaseModel):
    detector: str
    image: str
    params: dict  # e.g. {"dx":1.2, "dy":-0.5, "theta":0.1, "scale":1.0}

@app.post("/shift")
def shift_feats(req: ShiftRequest):
    from save_feature import load_feature_npz, save_feature_npz
    path = f"/data/features/{req.detector}/{req.image}.npz"
    kpts, desc, meta = load_feature_npz(path)
    dx = req.params.get("dx",0.0); dy = req.params.get("dy",0.0)
    theta = req.params.get("theta",0.0); scale = req.params.get("scale",1.0)
    # apply rotation+scale+translate about origin or center
    # for simplicity apply: [x,y] -> scale*R(theta) @ [x,y] + [dx,dy]
    c = np.cos(theta); s = np.sin(theta)
    R = np.array([[scale*c, -scale*s],[scale*s, scale*c]])
    new_kpts = (kpts @ R.T) + np.array([dx,dy])
    # save as a new feature file for this run
    new_meta = meta.copy(); new_meta.update({"shift":req.params})
    out_path = f"/data/features_shifted/{req.detector}/{req.image}.npz"
    save_feature_npz(out_path, new_kpts, desc, new_meta)
    return {"shifted_path": out_path}
