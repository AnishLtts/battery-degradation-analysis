import os
import glob
import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                             # the "My Project" folder
OUT_DIR = r"C:\Users\99017994\battery_out"               # Model 1 output

MECH = {"LLI": "LLI (lost lithium)", "LAM_PE": "LAM-PE (positive side)",
        "LAM_NE": "LAM-NE (negative side)", "RES": "Resistance growth"}

st.set_page_config(page_title="Battery Degradation", layout="wide")
st.title("Battery Degradation Analysis")

tab1, tab2 = st.tabs(["Model 1 - which mechanism", "Model 2 - how much each"])

# ================================================================ MODEL 1
with tab1:
    df = pd.read_csv(os.path.join(OUT_DIR, "real_diagnosis.csv"))
    cell = st.selectbox("Choose a battery", sorted(df["cell"].unique()), key="m1")
    d = df[df["cell"] == cell].sort_values("cycle")

    last = d.iloc[-1]
    c1, c2, c3 = st.columns(3)
    c1.metric("Cycles", len(d))
    c2.metric("Latest diagnosis", last["diagnosis"])
    c3.metric("Confidence", f'{last["confidence_pct"]:.0f} %', last["trust"] + " trust")

    st.subheader("Capacity over cycles")
    st.line_chart(d.set_index("cycle")["measured_capacity_Ah"])

    st.subheader("What was diagnosed, and how often")
    st.bar_chart(d["diagnosis"].value_counts())

    with st.expander("All cycles"):
        st.dataframe(d, use_container_width=True)

# ================================================================ MODEL 2
with tab2:
    # look everywhere under the project folder, and under battery_out too
    files = (glob.glob(os.path.join(ROOT, "**", "*_contributions.csv"), recursive=True)
             + glob.glob(os.path.join(OUT_DIR, "**", "*_contributions.csv"), recursive=True))
    if not files:
        st.error("No *_contributions.csv found. Run Model 2 step 6 first.")
        st.write("Searched inside:", ROOT, "and", OUT_DIR)
        st.stop()

    names = {os.path.basename(f).split("_")[0].upper(): f for f in files}
    cell2 = st.selectbox("Choose a battery", sorted(names), key="m2")
    m = pd.read_csv(names[cell2]).sort_values("cycle")
    st.caption("file: " + names[cell2])

    last = m.iloc[-1]
    st.subheader(f"Cycle {int(last['cycle'])}: what caused the "
                 f"{last['capacity_loss_pct_smooth']:.1f} % capacity loss")

    share = pd.DataFrame({
        "mechanism": [MECH[k] for k in MECH],
        "share_pct": [last[f"{k}_pct_smooth"] for k in MECH],
        "plus_minus": [last[f"{k}_pm"] for k in MECH],
    }).set_index("mechanism")

    left, right = st.columns([2, 1])
    left.bar_chart(share["share_pct"])
    right.metric("Dominant mechanism", last["dominant_mechanism"])
    right.metric("Label confidence", last["label_confidence"])
    right.dataframe(share.round(1), use_container_width=True)

    st.subheader("How the shares change as the battery ages")
    trend = m.set_index("cycle")[[f"{k}_pct_smooth" for k in MECH]]
    trend.columns = [MECH[k] for k in MECH]
    st.line_chart(trend)

    st.caption("Shares are 10-cycle averages. Early cycles say 'Too early' because the "
               "capacity loss is still under 2 %, too small to split up.")

    with st.expander("All cycles"):
        st.dataframe(m, use_container_width=True)