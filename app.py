import streamlit as st
import plotly.graph_objects as go
import nibabel as nib
import numpy as np
import tempfile
import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess

# ----------------------------
# Prediction Configs
# ----------------------------
IMG_SIZE = 300
label_mapping = {
    'AD'  : "Alzheimer's Disease",
    'CN'  : "Cognitive Normal",
    'EMCI': "Early Mild Cognitive Impairment",
    'LMCI': "Late Mild Cognitive Impairment"
}
idx_to_label = dict(enumerate(sorted(label_mapping)))

st.set_page_config(page_title="Alzheimer's MRI Viewer", layout="wide")
st.title("🧠 Alzheimer's Detection and Brain MRI Explorer")

tabs = st.tabs(["🏠 Home", "🧠 MRI Viewer", "🧪 2D MRI Prediction", "🧬 Biomarker Prediction"])

with tabs[0]:
    st.markdown("""
    ## 🧠 Understanding Alzheimer's Disease
    Alzheimer's disease (AD) is a progressive neurological disorder that leads to memory loss, confusion, and behavioral changes. It is the most common cause of dementia, affecting millions worldwide.

    ### 🧩 Symptoms of Alzheimer's:
    - Memory loss that disrupts daily life
    - Challenges in problem-solving or planning
    - Difficulty completing familiar tasks
    - Confusion with time or place
    - Trouble understanding visual images and spatial relationships
    - Withdrawal from work or social activities

    ### 🧠 Brain Comparison
    **Normal Brain**:
    - Full volume, well-defined regions, minimal shrinkage

    **Alzheimer's Brain**:
    - Shrinkage (atrophy), especially in the hippocampus and cortex
    - Enlarged ventricles
    - Loss of synapses and neurons
    """)

    st.image("AugmentedAlzheimerDataset/normalVsAD.jpg", caption="Comparison of Healthy Brain and Alzheimer's Disease Brain",width=500)

    st.markdown("""
    ### 🧪 What This App Offers
    - 🧬 **MRI Visualization**: Interactively browse 3D brain scans using the .nii file.
    - 📊 **2D MRI Prediction**: Classify stages of Alzheimer’s from single MRI slice
    - 💡 **Biomarker Analysis** *in progress*

    ### 📚 Research Insight
    [Biomarker Focus & Brain Changes in Alzheimer's](https://adni.bitbucket.io/reference/docs/UPENNBIOMK9/ADNI%20METHODS%20doc%20for%20Roche%20Elecsys%20CSF%20immunoassays%20vfinal.pdf)

    [Glial Fibrillary Acidic Protein (GFAP) in Plasma vs. CSF](https://pmc.ncbi.nlm.nih.gov/articles/PMC8524356/)

    ---
    💡 *This is a research-driven project combining deep learning with interactive MRI exploration to support early detection and awareness of Alzheimer’s disease.*
    """)

with tabs[1]:
    st.header("📤 Upload Brain MRI File (.nii)")
    file = st.file_uploader("Upload a NIfTI file (.nii or .nii.gz)", type=["nii", "gz"])

    if file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as tmp_file:
            tmp_file.write(file.read())
            nii_path = tmp_file.name

        img = nib.load(nii_path)
        data = img.get_fdata()
        nb_slices = data.shape[2]
        r, c = data.shape[0], data.shape[1]

        fig = go.Figure(frames=[
            go.Frame(
                data=go.Surface(
                    z=k * np.ones((r, c)),
                    surfacecolor=np.flipud(data[:, :, k].T),
                    cmin=0,
                    cmax=data.max()
                ),
                name=str(k)
            ) for k in range(nb_slices)
        ])

        fig.add_trace(go.Surface(
            z=0 * np.ones((r, c)),
            surfacecolor=np.flipud(data[:, :, 0].T),
            colorscale='Gray',
            cmin=0,
            cmax=data.max(),
            colorbar=dict(thickness=20, ticklen=4)
        ))

        def frame_args(duration):
            return {
                "frame": {"duration": duration},
                "mode": "immediate",
                "fromcurrent": True,
                "transition": {"duration": duration, "easing": "linear"},
            }

        sliders = [{
            "pad": {"b": 10, "t": 60},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {"args": [[f.name], frame_args(0)], "label": str(k), "method": "animate"}
                for k, f in enumerate(fig.frames)
            ],
        }]

        fig.update_layout(
            title='🧠 Axial Brain MRI View (Interactive)',
            width=700,
            height=700,
            scene=dict(
                zaxis=dict(range=[-1, nb_slices + 1], autorange=False),
                aspectratio=dict(x=1, y=1, z=0.5),
            ),
            updatemenus=[{
                "buttons": [
                    {"args": [None, frame_args(70)], "label": "▶", "method": "animate"},
                    {"args": [[None], frame_args(0)], "label": "❚❚", "method": "animate"},
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 70},
                "type": "buttons",
                "x": 0.1,
                "y": 0,
            }],
            sliders=sliders
        )

        st.plotly_chart(fig)

# MRI
with tabs[2]:
    st.header("🧪 Predict Alzheimer's Stage from MRI Image")
    st.markdown("""
    Upload a single **2D brain MRI slice** (JPG/PNG) to predict its classification into one of the following:
    - 🟢 CN: Cognitive Normal
    - 🟡 EMCI: Early Mild Cognitive Impairment
    - 🟠 LMCI: Late Mild Cognitive Impairment
    - 🔴 AD: Alzheimer’s Disease
    
    The model is trained on clinical datasets and uses deep neural networks (EfficientNet, DenseNet) to provide predictions with confidence levels.
    
    You can download the test images in the zip file present in the [data](https://drive.google.com/drive/folders/1Hv4hVk4WX5ZrVINIYvdiDY_Q9R_n4TeV?usp=sharing)
    """)

    img_file = st.file_uploader("Upload a 2D MRI Image", type=["png", "jpg", "jpeg"])
    model_choice = st.selectbox("Choose a Model", ["EfficientNetB3", "DenseNet169", "Ensemble"])

    if img_file and st.button("Predict"):
        temp_path = os.path.join(tempfile.gettempdir(), img_file.name)
        with open(temp_path, "wb") as f:
            f.write(img_file.read())

        def load_and_preprocess(path):
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = eff_preprocess(img.astype(np.float32))
            return np.expand_dims(img, 0)

        x = load_and_preprocess(temp_path)

        models, names = [], []
        if model_choice == "EfficientNetB3":
            models = [load_model('./AugmentedAlzheimerDataset/EfficientNetB3_best.keras', compile=False)]
            names = ["EfficientNetB3"]

        probs_stack = [m.predict(x, verbose=0)[0] for m in models]
        probs = np.mean(probs_stack, axis=0)
        pred_idx = int(np.argmax(probs))
        pred_label = idx_to_label[pred_idx]
        human_readable = label_mapping[pred_label]

        st.success(f"Predicted: **{human_readable}** ({pred_label})")

        st.markdown("### 🔬 Class Probabilities")
        for i, p in enumerate(probs):
            st.write(f"{idx_to_label[i]} ({label_mapping[idx_to_label[i]]}): {p*100:.2f}%")
            
with tabs[3]:
    st.header("🧬 Biomarker Analysis (In Progress)")

    st.markdown("""
    I am currently analyzing various **CSF and plasma biomarkers** from the ADNI dataset to predict **amyloid positivity**, a key indicator of Alzheimer's pathology.

    ### 📂 Datasets Used:
    - `ADNIMERGE`: Demographics, baseline diagnosis, cognitive scores
    - `UPENNBIOMK Roche Elecsys`: CSF markers like Aβ42, Aβ40, tau, p-tau181
    - `UPENN_PLASMA FUJIREBIO_QUANTERIX`: Plasma markers like p-tau217, GFAP, NfL, Aβ42/40 ratios

    | Biomarker           | Source   | Description                                                                 | Cutoff / Note                    |
    |---------------------|----------|-----------------------------------------------------------------------------|----------------------------------|
    | **AB42_AB40_F**     | Plasma   | Ratio of Aβ42 to Aβ40 in blood. Reflects amyloid burden non-invasively.    | No fixed cutoff; higher is better|
    | **ABETA42**         | CSF      | Aβ42 peptide concentration; lower = more amyloid plaques in brain.         | **< 192 pg/mL** → amyloid+       |
    | **ABETA40**         | CSF      | Aβ40 peptide; used to normalize Aβ42 to reduce variability.                | Used with Aβ42 to compute ratio  |
    | **ABETA42/40 ratio**| CSF      | Normalized Aβ42/Aβ40 ratio improves sensitivity over Aβ42 alone.           | **< 0.061** → amyloid+ (Elecsys) |
    | **pT217_F**         | Plasma   | Phosphorylated tau at threonine 217 — early marker of tau pathology.       | Higher → more risk               |
    | **NfL_Q**           | Plasma   | Neurofilament light — marker of axonal damage.                             | Elevated in neurodegeneration    |
    | **GFAP_Q**          | Plasma   | Glial fibrillary acidic protein — indicates astrocyte activation.          | Higher in AD and inflammation    |
    | **LOG_PTAU**, **LOG_TAU** | CSF | Reflect tau accumulation, a downstream marker in AD cascade.              | Exploratory                      |

    ---


    The visualizations below show how these features distribute with respect to **Amyloid status (positive = 1 / negative = 0)**.
    """)

    st.image("AugmentedAlzheimerDataset/adni/violin_AB42_AB40_F.png", caption="A lower Aβ42/Aβ40 ratio is typically associated with amyloid positivity, indicating abnormal amyloid accumulation. This plot shows that positive individuals tend to have reduced ratios", width=600)
    st.image("AugmentedAlzheimerDataset/adni/violin_ABETA40.png", caption="ABETA40 levels remain relatively stable across groups, making it a normalization factor. However, minor spread differences may reflect biological variability.", width=600)
    st.image("AugmentedAlzheimerDataset/adni/violin_ABETA42.png", caption="Aβ42 levels are often significantly lower in amyloid-positive individuals due to deposition of this peptide in plaques, making it a key diagnostic indicator.", width=600)
    st.image("AugmentedAlzheimerDataset/adni/violin_ABETA42_40.png", caption="This ratio is a more reliable predictor than Aβ42 alone. A clear decrease is observable in amyloid-positive individuals, supporting its use in classification.", width=600)
    st.image("AugmentedAlzheimerDataset/adni/violin_GFAP_Q.png", caption="Elevated GFAP is associated with astrocytic activation, often seen in early Alzheimer's pathology. The spread here suggests higher levels in amyloid-positive cases.", width=600)
    st.image("AugmentedAlzheimerDataset/adni/violin_LOG_PTAU.png", caption="Higher phosphorylated tau levels are strongly linked with amyloid pathology and neurodegeneration. Amyloid-positive subjects show slightly elevated levels.", width=600)
    st.image("AugmentedAlzheimerDataset/adni/violin_LOG_TAU.png", caption="Total tau is a general marker of neuronal injury. While not specific to amyloid, elevated values are more common in positive cases.", width=600)
    st.image("AugmentedAlzheimerDataset/adni/violin_NfL_Q.png", caption="A marker of neurodegeneration. This plot shows a rightward tail in amyloid-positive individuals, suggesting higher neuronal injury.", width=600)
    st.image("AugmentedAlzheimerDataset/adni/violin_pT217_F.png", caption="This highly specific biomarker for Alzheimer's shows increased levels in amyloid-positive individuals, supporting its diagnostic value.", width=600)

    st.info("🧪 This section is under development. Further statistical tests and modeling will follow.")
