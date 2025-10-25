import os, requests
from fastapi import FastAPI, HTTPException

AUDIT_API = os.getenv("AUDIT_API_URL", "http://audit_api:8080")

app = FastAPI(title="AIEO MCP Tools", version="0.1.0")

def _proxy(path: str, **params):
    try:
        r = requests.get(f"{AUDIT_API}{path}", params=params, timeout=8)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"upstream error: {e}")

@app.get("/health")
def health():
    return {"ok": True, "audit_api": AUDIT_API}

# --- Tools your LLM can call ---
@app.get("/tools/get_latest_routing_summary")
def get_latest_routing_summary(minutes: int = 60):
    """Wraps audit_api:/routing/summary"""
    return _proxy("/routing/summary", minutes=minutes)

@app.get("/tools/get_top_shap_features")
def get_top_shap_features(limit: int = 8, minutes: int = 120):
    """Wraps audit_api:/routing/top-features"""
    return _proxy("/routing/top-features", limit=limit, minutes=minutes)

@app.get("/tools/get_event_stats")
def get_event_stats(minutes: int = 60):
    """Returns counts + recent sample from audit_api:/events/recent + summary"""
    summary = _proxy("/routing/summary", minutes=minutes)
    recent  = _proxy("/events/recent",  limit=200)
    return {"window_min": minutes, "summary": summary, "recent": recent}

@app.get("/tools/get_system_health")
def get_system_health():
    """Wraps audit_api:/system/health"""
    return _proxy("/system/health")
