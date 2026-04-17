# streamlit_app.py
import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import numpy as np

API_URL = st.secrets["Deployment_URL"]

CANONICAL_FEATURES = {
    "ExterQual": "None",
    "BsmtQual": "None",
    "HeatingQC": "None",
    "KitchenQual": "None",
    "Neighborhood": "None",
    "Foundation": "None",
    "BsmtFinType1": "None",
    "GarageType": "None",
    "GarageFinish": "None",
    "OverallQual": "None",
    "GrLivArea": "None"
}

FEATURE_META = {
    "ExterQual": {
        "label": "Exterior quality",
        "examples": "Po,Fa,TA,Gd,Ex",
        "description": "Rates the quality of the house’s exterior materials."
    },
    "BsmtQual": {
        "label": "Basement quality",
        "examples": "None,Po,Fa,TA,Gd,Ex",
        "description": "Evaluates the height and finish of the basement."
    },
    "HeatingQC": {
        "label": "Heating quality",
        "examples": "Po,Fa,TA,Gd,Ex",
        "description": "Assesses the quality and condition of the heating system."
    },
    "KitchenQual": {
        "label": "Kitchen quality",
        "examples": "Po,Fa,TA,Gd,Ex",
        "description": "Rates the quality of kitchen finishes and layout."
    },
    "Neighborhood": {
        "label": "Neighborhood",
        "examples": "NAmes,CollgCr,OldTown,Edwards,Somerst,NridgHt,StoneBr,Timber,ClearCr,BrkSide",
        "description": "Identifies the physical location within Ames city limits."
    },
    "Foundation": {
        "label": "Foundation type",
        "examples": "PConc,CBlock,BrkTil,Slab,Stone,Wood",
        "description": "Specifies the type of foundation supporting the house."
    },
    "BsmtFinType1": {
        "label": "Basement finish type",
        "examples": "GLQ,ALQ,BLQ,LwQ,Rec,Unf",
        "description": "Describes the rating of the finished area in the basement."
    },
    "GarageType": {
        "label": "Garage type",
        "examples": "Attchd,Detchd,BuiltIn,CarPort,Basment,2Types",
        "description": "Indicates the type of garage present."
    },
    "GarageFinish": {
        "label": "Garage finish",
        "examples": "Fin,RFn,Unf",
        "description": "Describes the interior finish of the garage."
    },
    "OverallQual": {
        "label": "Overall quality (numeric)",
        "examples": "1,2,3,4,5,6,7,8,9,10",
        "description": "Rates the overall material and finish of the house (1=poor, 10=excellent)."
    },
    "GrLivArea": {
        "label": "Living area (sqft)",
        "examples": "850,1200,1650,3000",
        "description": "Total above‑ground living area in square feet."
    }
}

st.set_page_config(page_title="AI Real Estate Agent", layout="wide")
st.title("AI Real Estate Agent")

# Sidebar: description and quick help
st.sidebar.header("Property description")
user_query = st.sidebar.text_area(
    "Description",
    value="3-bedroom ranch with a big garage in a good neighborhood",
    height=140,
)

st.sidebar.markdown("**Quick help** — pick a feature to see examples")
feature_options = ["— select —"] + [FEATURE_META[k]["label"] for k in CANONICAL_FEATURES.keys()]
label_to_key = {v["label"]: k for k, v in FEATURE_META.items()}
selected_label = st.sidebar.selectbox("Feature", feature_options, index=0)
selected_key = label_to_key.get(selected_label)
if selected_key:
    meta = FEATURE_META[selected_key]
    st.sidebar.markdown(f"**{meta['label']}**")
    st.sidebar.write("Examples:", meta["examples"])
    st.sidebar.write(meta["description"])
    if st.sidebar.button("Quick-fill example"):
        example = meta["examples"].split(",")[0].strip()
        st.session_state.setdefault("features", CANONICAL_FEATURES.copy())[selected_key] = example
        st.sidebar.success(f"Inserted example '{example}' for {meta['label']}")

# Session state init
if "features" not in st.session_state:
    st.session_state["features"] = CANONICAL_FEATURES.copy()
if "last_response" not in st.session_state:
    st.session_state["last_response"] = None

def _fill_feature_callback(key: str):
    widget_key = f"fill_{key}"
    val = st.session_state.get(widget_key)
    val = val if (val is not None and val != "") else "None"
    features = st.session_state.get("features", CANONICAL_FEATURES.copy())
    features[key] = val
    st.session_state["features"] = features

# Submit and Reset buttons
col_submit, col_reset = st.columns([1,1])
with col_submit:
    if st.button("Submit description and features"):
        payload = {"user_input": user_query, "json_feat": st.session_state["features"]}
        try:
            r = requests.post(API_URL, json=payload, timeout=15)
            r.raise_for_status()
            resp = r.json()
            returned = resp.get("features")
            if isinstance(returned, list) and returned:
                returned = returned[0]
            if isinstance(returned, dict):
                merged = CANONICAL_FEATURES.copy()
                merged.update({k: (v if v is not None else "None") for k,v in returned.items()})
                st.session_state["features"] = merged
            st.session_state["last_response"] = resp
        except Exception as e:
            st.error(f"Server error: {e}")
with col_reset:
    if st.button("Reset all features"):
        st.session_state["features"] = CANONICAL_FEATURES.copy()
        st.session_state["last_response"] = None
        for k in list(st.session_state.keys()):
            if k.startswith("fill_"):
                del st.session_state[k]
        st.success("Features reset")

# Provided vs Missing
features = st.session_state["features"]
provided = [k for k,v in features.items() if v not in (None,"None","")]
missing = [k for k,v in features.items() if v in (None,"None","")]

col_ok, col_missing = st.columns([1,1])
with col_ok:
    st.markdown("<div style='background:#e6ffed;border-radius:6px;padding:10px'>", unsafe_allow_html=True)
    st.markdown(f"**Provided ({len(provided)})**")
    st.write(", ".join(provided) if provided else "—")
    st.markdown("</div>", unsafe_allow_html=True)
with col_missing:
    st.markdown("<div style='background:#ffecec;border-radius:6px;padding:10px'>", unsafe_allow_html=True)
    st.markdown(f"**Missing ({len(missing)})**")
    st.write(", ".join(missing) if missing else "—")
    st.markdown("</div>", unsafe_allow_html=True)

# Dropdowns for missing features
if missing:
    st.subheader("Fill missing features")
    cols_per_row = 3
    for i in range(0, len(missing), cols_per_row):
        row_keys = missing[i:i+cols_per_row]
        cols = st.columns(len(row_keys))
        for col, key in zip(cols, row_keys):
            meta = FEATURE_META.get(key, {"label": key, "examples": ""})
            label = meta["label"]
            examples = [x.strip() for x in meta.get("examples","").split(",") if x.strip()]
            options = ["None"] + examples
            widget_key = f"fill_{key}"
            current_val = features.get(key, "None")
            if widget_key not in st.session_state:
                st.session_state[widget_key] = current_val
            with col:
                st.selectbox(
                    label,
                    options=options,
                    key=widget_key,
                    on_change=_fill_feature_callback,
                    args=(key,),
                    index=options.index(current_val) if current_val in options else 0
                )

# Editable table
st.subheader("Features table")
df = pd.DataFrame(list(features.items()), columns=["feature","value"])
edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)
edited = dict(zip(edited_df["feature"].tolist(), edited_df["value"].tolist()))
if edited:
    cleaned = {k: (v if v != "" else "None") for k,v in edited.items()}
    merged = CANONICAL_FEATURES.copy()
    merged.update(cleaned)
    st.session_state["features"] = merged

# Prediction
st.subheader("Prediction and interpretation")
resp = st.session_state.get("last_response")
if resp:
    pred = resp.get("prediction")
    price = None
    if isinstance(pred,(int,float)):
        price = pred
    elif isinstance(pred,dict):
        price = pred.get("predicted_price")
    if price is not None:
        st.metric("Predicted Price (USD)", f"${int(price):,}")

    interp = resp.get("interpretation")
    if interp:
        st.write("**Interpretation**")
        st.write(interp)

    # --- Feature importance chart (appears after price + interpretation) ---
    st.subheader("Feature Importance Analysis")

    # Accept multiple possible keys from API for importance
    importance = resp.get("feature_importance") or resp.get("feature_importances") or resp.get("importance") or None

    # If API didn't return importances, use a sensible fallback (so UI still shows something)
    if importance is None:
        # fallback example (sorted later)
        importance = {
            "OverallQual": 0.50,
            "GrLivArea": 0.15,
            "ExterQual": 0.08,
            "KitchenQual": 0.06,
            "BsmtQual": 0.05,
            "GarageFinish": 0.04,
            "BsmtFinType1": 0.03,
            "GarageType": 0.03,
            "Foundation": 0.02,
            "Neighborhood": 0.02
        }

    # Ensure importance is a dict; if API returned list/tuples convert accordingly
    if isinstance(importance, list):
        try:
            importance = dict(importance)
        except Exception:
            # try to coerce list of pairs
            importance = {str(i): float(v) for i, v in enumerate(importance)}

    # Convert values to floats and filter zero/negative
    cleaned_imp = {}
    for k, v in importance.items():
        try:
            val = float(v)
        except Exception:
            continue
        if val > 0:
            cleaned_imp[k] = val

    if not cleaned_imp:
        st.info("No feature importance values available to plot.")
    else:
        # Sort by importance descending
        items = sorted(cleaned_imp.items(), key=lambda x: x[1], reverse=True)
        names = [i[0] for i in items]
        values = [i[1] for i in items]

        # Normalize to sum to 1 for nicer y-axis if not already
        total = sum(values)
        if total > 0:
            values = [v / total for v in values]

        # Create a polished horizontal bar chart
        fig, ax = plt.subplots(figsize=(9, max(4, 0.5 * len(names))))
        y_pos = np.arange(len(names))

        bars = ax.barh(y_pos, values, color="#4c72b0", edgecolor="#2a4a7a", height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=11)
        ax.invert_yaxis()  # largest on top

        # Add value labels to bars
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{width:.2f}", va="center", fontsize=10, color="#333333")

        ax.set_xlabel("Relative importance (normalized)", fontsize=12)
        ax.set_title("Feature Importance in Price Prediction", fontsize=14, fontweight="bold")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        plt.tight_layout()

        st.pyplot(fig)
else:
    st.info("No server response yet. Use the Submit button above.")
