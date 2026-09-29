"""
Corrected results dashboard for the BLDC Hall-sensor project.

Run from the project folder:
    streamlit run app_corrected.py

All numbers are the ones reported in the paper (leak-free protocol), stored
directly in this file, so it does not depend on any CSV path.
"""
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="BLDC Hall-Sensor Fault Results", layout="wide")

# ------------------------------------------------------------------ data
LEAKFREE = pd.DataFrame(
    [
        ("EfficientNetB0", 58.8, 60.0, 60.4, 59.7, 4.05, 132.6),
        ("ConvNeXt-Tiny*", 54.2, 57.1, 62.1, 57.8, 27.82, 429.9),
        ("EfficientNetV2-B0*", 55.8, 58.3, 58.3, 57.5, 5.92, 109.9),
        ("ResNet50", 52.5, 52.9, 54.6, 53.3, 23.59, 197.0),
        ("MobileNetV3-Small*", 49.6, 55.0, 52.5, 52.4, 0.94, 29.4),
        ("MobileNetV3-Large*", 52.5, 52.9, 51.3, 52.2, 3.00, 59.7),
        ("VGG16", 52.1, 51.7, 52.1, 51.9, 14.71, 385.0),
        ("DenseNet121", 49.2, 53.3, 52.5, 51.7, 7.04, 256.1),
        ("NASNetMobile*", 48.8, 50.4, 52.5, 50.6, 4.27, 120.6),
        ("InceptionV3", 47.5, 52.5, 49.6, 49.9, 21.80, 154.1),
        ("Xception", 49.2, 51.3, 47.5, 49.3, 20.86, 216.0),
        ("MobileNetV2", 47.9, 49.2, 50.0, 49.0, 2.26, 59.2),
        ("MobileNetV1", 47.1, 48.3, 50.4, 48.6, 3.23, 128.0),
    ],
    columns=["Model", "0.0001 s", "0.005 s", "0.01 s", "Average", "Params (M)", "Latency (ms)"],
)

INITIAL = pd.DataFrame(
    [
        ("MobileNetV1", 72.9, 79.2, 70.8, 74.3),
        ("InceptionV3", 64.6, 68.8, 66.7, 66.7),
        ("Xception", 60.4, 66.7, 68.8, 65.3),
        ("DenseNet121", 66.7, 62.5, 56.2, 61.8),
        ("VGG16", 60.4, 50.0, 68.8, 59.7),
        ("MobileNetV2", 60.4, 54.2, 64.6, 59.7),
        ("ResNet50", 50.0, 50.0, 70.8, 56.9),
        ("EfficientNetB0", 50.0, 50.0, 50.0, 50.0),
    ],
    columns=["Model", "0.0001 s", "0.005 s", "0.01 s", "Average"],
)

FEATURES = pd.DataFrame(
    [
        ("Healthy", 0.056, 0.053, 0.998, 0.999, 0.005, 7.252),
        ("0.0001 s", 0.236, 0.234, 0.972, 0.972, 0.051, 7.251),
        ("0.005 s", 0.800, 0.761, 0.743, 0.742, 0.335, 7.198),
        ("0.01 s", 3.275, 3.122, 0.291, 0.289, 0.740, 7.907),
    ],
    columns=["Level", "e_b", "e_c", "rho_b", "rho_c", "h3/h1", "T (ms)"],
)

CLASSIFIERS = pd.DataFrame(
    [
        ("0.0001 s", 100, 100, 100, 79.2, 100, 100),
        ("0.005 s", 100, 100, 100, 83.3, 100, 100),
        ("0.01 s", 100, 100, 100, 83.3, 100, 100),
    ],
    columns=["Level", "A: LR", "A: SVM", "A: RF", "B: LR", "B: SVM", "B: RF"],
)

STARTUP = pd.DataFrame(
    [
        ("Healthy", 59.2, 6.6, 5.5),
        ("0.0001 s", 59.6, 17.7, 26.5),
        ("0.005 s", 54.7, 51.0, 76.0),
        ("0.01 s", 40.7, 114.0, 740.0),
    ],
    columns=["Recording", "Stage 2 onset (ms)", "i_b error (%)", "i_c error (%)"],
)

# ------------------------------------------------------------------ page
st.title("Hall-Sensor Fault Identification in a BLDC Drive")
st.caption(
    "Leak-free results. Test data was not used to select the training epoch. "
    "Chance level for the CNN task is 50%."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Best CNN (leak-free)", "59.7 %", "EfficientNetB0")
c2.metric("First (leaky) headline", "74.3 %", "MobileNetV1, not reproduced")
c3.metric("Current features (SVM/RF)", "100 %", "needs wider validation")
c4.metric("Start-up block, healthy", "5-7 % error", "after about 59 ms")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "CNN results (corrected)",
        "Before vs after the fix",
        "Current features",
        "Single-current start-up",
        "Limitations",
    ]
)

with tab1:
    st.subheader("13 CNN backbones, mean test accuracy over 5 random splits")
    d = LEAKFREE.sort_values("Average")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(d["Model"], d["Average"], color="#4a7bb7")
    ax.axvline(50, color="red", linestyle="--", label="chance (50%)")
    ax.set_xlim(40, 65)
    ax.set_xlabel("Average test accuracy (%)")
    ax.legend()
    st.pyplot(fig)
    st.dataframe(LEAKFREE, use_container_width=True)
    st.caption("* newer model added in this work. Every model is within about 10 points of chance.")

with tab2:
    st.subheader("Same 8 reference models: initial protocol vs corrected protocol")
    ref = INITIAL[["Model", "Average"]].rename(columns={"Average": "Initial (test set used for epoch selection)"})
    fixed = LEAKFREE[["Model", "Average"]].rename(columns={"Average": "Corrected (leak-free)"})
    cmp_df = ref.merge(fixed, on="Model")
    fig2, ax2 = plt.subplots(figsize=(9, 4.5))
    x = range(len(cmp_df))
    w = 0.38
    ax2.bar([i - w / 2 for i in x], cmp_df.iloc[:, 1], w, label="Initial", color="#d98c5f")
    ax2.bar([i + w / 2 for i in x], cmp_df.iloc[:, 2], w, label="Corrected", color="#4a7bb7")
    ax2.axhline(50, color="red", linestyle="--", label="chance (50%)")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(cmp_df["Model"], rotation=30, ha="right")
    ax2.set_ylabel("Average test accuracy (%)")
    ax2.legend()
    st.pyplot(fig2)
    st.dataframe(cmp_df, use_container_width=True)
    st.info(
        "Selecting the best epoch on the 48 test images inflated a near-chance classifier. "
        "MobileNetV1 drops from 74.3% to 48.6% once the test set is kept untouched."
    )

with tab3:
    st.subheader("Classifiers on 23 features from the raw phase currents")
    st.dataframe(CLASSIFIERS, use_container_width=True)
    st.caption("Balanced accuracy (%). A = time-split, B = unseen fault file.")
    st.subheader("How the reconstruction features change with displacement")
    fig3, ax3 = plt.subplots(figsize=(6, 3.5))
    ax3.plot(FEATURES["Level"], FEATURES["rho_b"], marker="o", label="rho_b")
    ax3.plot(FEATURES["Level"], FEATURES["rho_c"], marker="s", label="rho_c")
    ax3.set_ylabel("Correlation with delayed i_a")
    ax3.legend()
    st.pyplot(fig3)
    st.dataframe(FEATURES, use_container_width=True)
    st.warning(
        "Only one healthy recording and short 300 ms recordings, so part of the score may reflect "
        "recording identity rather than the fault."
    )

with tab4:
    st.subheader("Two-stage single-current block (measures i_a only)")
    st.write(
        "Stage 1: no output while the start-up surge is present (open-loop start). "
        "Stage 2: once two consecutive periods agree, i_b and i_c are generated from delayed i_a."
    )
    st.dataframe(STARTUP, use_container_width=True)
    fig4, ax4 = plt.subplots(figsize=(6, 3.5))
    ax4.bar(STARTUP["Recording"], STARTUP["i_b error (%)"], color="#4a7bb7")
    ax4.set_yscale("log")
    ax4.set_ylabel("i_b steady-state error (%), log scale")
    st.pyplot(fig4)
    st.caption("Error grows with fault size, so the reconstruction error can act as a health indicator.")

with tab5:
    st.subheader("Limitations")
    st.markdown(
        "- Recordings are only 300 ms long and there is a single healthy recording.\n"
        "- Test sets are small (48 images, or about 3-4 independent windows).\n"
        "- CNN backbones were frozen, not fine-tuned.\n"
        "- The dataset may be simulated, so results should be confirmed on measured currents.\n"
        "- The single-current block cannot work during the first inrush surge."
    )
    st.subheader("Next step")
    st.write("Record real healthy and faulty currents in the lab and repeat the leak-free evaluation.")
