import os, json, time, math
from datetime import datetime, timezone
import requests
import pandas as pd
import streamlit as st

AUDIT_API = os.getenv("AUDIT_API_URL", "http://audit_api:8090")
PG_ENABLED = os.getenv("PG_DSN") is not None  # optional per-event lookup
REFRESH_SEC = int(os.getenv("DASH_REFRESH_SEC", "5"))
# --- add near top ---
LLM_URL = os.getenv("LLM_SUMM_URL", "http://llm_summarizer:8080")
if "llm_summary" not in st.session_state:
    st.session_state.llm_summary = None
    st.session_state.llm_summary_ts = None
    st.session_state.llm_hold_refresh = False


st.set_page_config(page_title="AIEO – Live Ops", layout="wide")

@st.cache_data(ttl=5)
def get_summary(minutes: int = 60):
    r = requests.get(f"{AUDIT_API}/routing/summary", params={"minutes": minutes}, timeout=5)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=5)
def get_top_features(limit: int = 8, minutes: int = 120):
    r = requests.get(f"{AUDIT_API}/routing/top-features", params={"limit": limit, "minutes": minutes}, timeout=5)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=5)
def get_recent(limit: int = 200):
    r = requests.get(f"{AUDIT_API}/events/recent", params={"limit": limit}, timeout=5)
    r.raise_for_status()
    return r.json()

def df_routing_counts(by_decision):
    # accept either the full response or just the list
    rows = by_decision if isinstance(by_decision, list) else by_decision.get("by_decision", [])
    if not rows:
        return pd.DataFrame(columns=["decision", "count", "avg_score"])
    # normalize key names: API uses "n", dashboard used "count"
    recs = [
        {
            "decision": r.get("decision"),
            "count": r.get("n", r.get("count", 0)),
            "avg_score": r.get("avg_score"),
        }
        for r in rows
    ]
    return pd.DataFrame(recs).sort_values("count", ascending=False)


def df_top_features(obj):
    # obj can be a list of rows or a dict {feature: value}
    if isinstance(obj, list):
        df = pd.DataFrame(obj)
        if df.empty:
            return pd.DataFrame(columns=["feature", "mean_abs_shap"])
        # normalize columns
        if "mean_abs_shap" not in df.columns:
            if "mean_shap" in df.columns:
                df["mean_abs_shap"] = df["mean_shap"].abs()
            elif "value" in df.columns:
                df = df.rename(columns={"value": "mean_abs_shap"})
            else:
                df["mean_abs_shap"] = None
        # keep only needed
        keep = [c for c in ["feature", "mean_abs_shap"] if c in df.columns]
        return df[keep].sort_values("mean_abs_shap", ascending=False)
    elif isinstance(obj, dict):
        recs = [{"feature": k, "mean_abs_shap": v} for k, v in (obj or {}).items()]
        return pd.DataFrame(recs).sort_values("mean_abs_shap", ascending=False)
    else:
        return pd.DataFrame(columns=["feature", "mean_abs_shap"])


def df_recent_events(lst):
    # each: {"event_id": "...", "event": <json or empty>, "created_at": "..."}
    if not lst: return pd.DataFrame(columns=["event_id","event_type","created_at"])
    rows = []
    for e in lst:
        ev = e.get("event") or {}
        if isinstance(ev, str):
            try: ev = json.loads(ev) if ev else {}
            except: ev = {}
        rows.append({
            "event_id": e.get("event_id"),
            "event_type": ev.get("event_type"),
            "created_at": e.get("created_at"),
        })
    df = pd.DataFrame(rows)
    # local throughput per minute (approx)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    return df

def kfmt(x):
    if x is None: return "-"
    if x >= 1_000_000: return f"{x/1_000_000:.1f}M"
    if x >= 1_000: return f"{x/1_000:.1f}k"
    return str(int(x))

st.title("🛰️ AIEO — Live Observability")

# --- Controls ---
colA, colB, colC, colD = st.columns([1,1,1,1])
with colA:
    window_min = st.selectbox("Routing window (min)", [5, 15, 30, 60, 120], index=3)
with colB:
    shap_min = st.selectbox("SHAP window (min)", [15, 30, 60, 120, 240], index=2)
with colC:
    feat_limit = st.slider("Top features", 3, 20, 8, 1)
with colD:
    st.write(" ") ; auto = st.toggle(f"Auto-refresh {REFRESH_SEC}s", value=True)

# --- Fetch ---
summary = get_summary(window_min)
topf    = get_top_features(feat_limit, shap_min)
recent  = get_recent(300)

df_counts = df_routing_counts(summary.get("by_decision", {}))
df_features = df_top_features(topf.get("features", []))
df_recent = df_recent_events(recent.get("events", []))

# --- KPIs ---
total_recent = len(df_recent)
ll_count = int(df_counts.loc[df_counts["decision"]=="low_latency", "count"].sum()) if not df_counts.empty else 0
batch_count = int(df_counts.loc[df_counts["decision"]=="batch", "count"].sum()) if not df_counts.empty else 0
ll_ratio = (ll_count / max(1, (ll_count+batch_count))) * 100.0

c1, c2, c3 = st.columns(3)
c1.metric("Events (recent fetch)", kfmt(total_recent))
c2.metric("Low-latency / Batch", f"{kfmt(ll_count)} / {kfmt(batch_count)}", f"{ll_ratio:.1f}% LL")
c3.metric("Window (routing/SHAP)", f"{window_min}m / {shap_min}m")

st.divider()

# --- Charts ---
col1, col2 = st.columns([1,1])

with col1:
    st.subheader("Routing decisions (window)")
    if df_counts.empty:
        st.info("No routing data in this window.")
    else:
        st.bar_chart(df_counts.set_index("decision"))

with col2:
    st.subheader("Top SHAP features (mean |value|)")
    if df_features.empty:
        st.info("No SHAP data in this window.")
    else:
        st.bar_chart(df_features.set_index("feature"))

st.divider()

# --- Recent events table + quick filter ---
st.subheader("Recent events")
q = st.text_input("Filter by event_id or event_type (client-side)", "")
if q:
    mask = df_recent["event_id"].astype(str).str.contains(q) | df_recent["event_type"].astype(str).str.contains(q)
    st.dataframe(df_recent.loc[mask].sort_values("created_at", ascending=False), use_container_width=True, height=320)
else:
    st.dataframe(df_recent.sort_values("created_at", ascending=False), use_container_width=True, height=320)

# --- LLM Summary section ---
st.divider()
st.subheader("LLM Summary (Groq)")

col_left, col_right = st.columns([1,3])
with col_left:
    summ_minutes = 60
    if st.button("Summarize last 60 minutes"):
        try:
            r = requests.post(f"{LLM_URL}/summarize",
                              json={"minutes": summ_minutes, "shap_limit": 8},
                              timeout=30)
            r.raise_for_status()
            data = r.json()
            st.session_state.llm_summary = data.get("summary", "")
            st.session_state.llm_summary_ts = datetime.now(timezone.utc).isoformat()
            st.session_state.llm_hold_refresh = True
            st.success("Summary generated.")
        except Exception as e:
            st.error(f"Failed to call summarizer: {e}")

    if st.button("Clear summary"):
        st.session_state.llm_summary = None
        st.session_state.llm_summary_ts = None
        st.session_state.llm_hold_refresh = False

with col_right:
    box = st.container(border=True)
    if st.session_state.llm_summary:
        ts = st.session_state.llm_summary_ts or "-"
        box.markdown(f"**Generated:** {ts} UTC")
        box.write(st.session_state.llm_summary)
    else:
        box.info("Click **Summarize last 60 minutes** to generate a Groq-based summary of routing and SHAP.")

# ---- Auto-refresh AFTER render ----
if auto and not st.session_state.llm_hold_refresh:
    time.sleep(REFRESH_SEC)
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()


# --- Optional per-event panel (requires PG_DSN and audit view) ---
if PG_ENABLED:
    st.divider()
    st.subheader("Per-event details (from audit view)")
    event_id = st.text_input("Enter event_id to inspect", "")
    if event_id:
        try:
            # call a lightweight endpoint if you added it; else do a direct query:
            # We’ll try /routing/by-id first (if you created it in 6a+), fallback to /events/recent scan.
            r = requests.get(f"{AUDIT_API}/routing/by-id", params={"event_id": event_id}, timeout=5)
            if r.ok:
                data = r.json()
                st.json(data)
            else:
                # Fallback: best-effort search in recent list
                row = df_recent.loc[df_recent["event_id"]==event_id]
                if row.empty:
                    st.warning("Not found in recent window.")
                else:
                    st.write(row.iloc[0].to_dict())
        except Exception as e:
            st.warning(f"Lookup failed: {e}")
else:
    st.caption("Tip: set PG_DSN env + add /routing/by-id in the audit API to enable per-event view.")
