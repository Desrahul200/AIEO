import os, json, requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Bring your own LLM: OpenAI-compatible endpoint
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE    = os.getenv("GROQ_BASE", "https://api.groq.com/openai/v1")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

MCP_URL = os.getenv("MCP_URL", "http://mcp_server:8080")

app = FastAPI(title="AIEO LLM Summarizer", version="0.1.0")

class SummIn(BaseModel):
    minutes: int = 60
    shap_minutes: int | None = None
    shap_limit: int = 8

def _get(path, **params):
    r = requests.get(f"{MCP_URL}{path}", params=params, timeout=8)
    if not r.ok:
        raise HTTPException(status_code=502, detail=f"{path} upstream {r.status_code}: {r.text}")
    return r.json()

def _chat(messages):
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    body = {"model": GROQ_MODEL, "messages": messages, "temperature": 0.2}
    r = requests.post(f"{GROQ_BASE}/chat/completions", headers=headers, json=body, timeout=20)
    if not r.ok:
        raise HTTPException(status_code=502, detail=f"LLM error {r.status_code}: {r.text}")
    return r.json()["choices"][0]["message"]["content"]

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/summarize")
def summarize(inp: SummIn):
    sm = _get("/tools/get_latest_routing_summary", minutes=inp.minutes)
    tf = _get("/tools/get_top_shap_features", minutes=inp.shap_minutes or inp.minutes, limit=inp.shap_limit)
    st = _get("/tools/get_system_health")

    prompt = f"""
You are the on-call assistant for the AIEO pipeline. Summarize the last {inp.minutes} minutes.
Provide: high-level trend, low-latency vs batch split, top SHAP drivers, anomalies, and any health risks.

Routing summary JSON:
{json.dumps(sm, indent=2)}

Top SHAP features JSON:
{json.dumps(tf, indent=2)}

System health JSON:
{json.dumps(st, indent=2)}

Write 5-8 bullet points plus a one-line exec summary. Be specific with numbers and time windows.
"""
    out = _chat([{"role":"system","content":"You are a precise SRE+MLE assistant."},
                 {"role":"user","content":prompt}])
    return {"minutes": inp.minutes, "summary": out}
