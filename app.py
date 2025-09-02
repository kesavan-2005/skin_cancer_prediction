import os
import datetime
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
from fpdf import FPDF
from streamlit_lottie import st_lottie
import requests
import tempfile

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "fmodel")
MODEL_PATH = os.path.join(MODEL_DIR, "trained_multimodal_model.h5")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "age_scaler.pkl")
DX_TYPE_ENCODER_PATH = os.path.join(MODEL_DIR, "dx_type_encoder.pkl")
SEX_ENCODER_PATH = os.path.join(MODEL_DIR, "sex_encoder.pkl")
LOC_ENCODER_PATH = os.path.join(MODEL_DIR, "localization_encoder.pkl")

IMG_SIZE = (224, 224)

# ----------- Utility: Lottie Animations -----------
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

loading_anim = load_lottie_url("https://assets10.lottiefiles.com/packages/lf20_tll0j4bb.json")
success_anim = load_lottie_url("https://assets10.lottiefiles.com/private_files/lf30_e3pteeho.json")

# ---------------- STREAMLIT PAGE CONFIG ----------------
st.set_page_config(page_title="Skin Cancer Multimodal Diagnosis", page_icon="🧠", layout="wide")

# -------------- Modern Style CSS -----------------
st.markdown("""
    <style>
        .main { background-color: #f6faff; }
        .stButton>button { border-radius: 8px; font-weight: bold; }
        .stFileUploader>div>div { border-radius: 12px !important; }
        .result-card {
            background: linear-gradient(90deg, #f9dcdc 40%, #ffe8e8);
            border-radius: 16px; 
            box-shadow: 0 4px 12px #efd5ce69;
            padding: 1.5em 2em; margin-bottom: 1.6em;
        }
        .recommend-card {
            background: #e7f6ef;
            border-radius: 14px;
            box-shadow: 0 2px 6px #bccfd869;
            padding: 1.2em 1.5em; margin-bottom: 1em;
        }
        .lottie { height: 110px; margin-bottom: 0.5em }
    </style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL + ENCODERS ----------------
@st.cache_resource(show_spinner="Loading model and assets...")
def load_assets():
    with open(LABEL_ENCODER_PATH, "rb") as f:
        le = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(DX_TYPE_ENCODER_PATH, "rb") as f:
        dx_type_le = pickle.load(f)
    with open(SEX_ENCODER_PATH, "rb") as f:
        sex_le = pickle.load(f)
    with open(LOC_ENCODER_PATH, "rb") as f:
        loc_le = pickle.load(f)

    model = load_model(
        MODEL_PATH,
        custom_objects={"focal_loss_fixed": None},
        compile=False
    )
    return model, le, scaler, dx_type_le, sex_le, loc_le

model, le, scaler, dx_type_le, sex_le, loc_le = load_assets()

# ----------- Recommendation Mapping -----------
rec_map = {
    "nv": {
        "rec": "Usually benign; monitor with ABCDE rule. If growth, color change, bleeding, or itching occurs, consult dermatologist.",
        "prev": "Use sunscreen SPF≥30, avoid excessive sun, perform monthly skin checks."
    },
    "bkl": {
        "rec": "Benign lesion; treatment usually not needed unless irritated or cosmetic. Confirm diagnosis if atypical.",
        "prev": "UV protection and routine skin checks."
    },
    "vasc": {
        "rec": "Usually harmless (angiomas, etc). Remove only if bleeding, painful, or cosmetically concerning.",
        "prev": "No specific prevention; avoid trauma and protect from sun."
    },
    "akiec": {
        "rec": "Precancerous; must be treated to avoid squamous cell carcinoma. Options: cryotherapy, topical creams, photodynamic therapy.",
        "prev": "Strict UV protection, avoid tanning, regular skin screening."
    },
    "bcc": {
        "rec": "Basal cell carcinoma; early treatment is very effective. Options: excision, Mohs surgery, topical/cryotherapy.",
        "prev": "Protect against UV exposure, avoid tanning beds, periodic skin exams."
    },
    "mel": {
        "rec": "Melanoma detected: urgent dermatologist visit required. May need surgery, immunotherapy, or targeted therapy.",
        "prev": "Strong UV protection (SPF≥50), monthly self-checks, yearly dermatologist check."
    },
    "df": {
        "rec": "Dermatofibroma; benign, no treatment unless painful or suspicious. Can be excised if needed.",
        "prev": "No specific prevention; monitor for changes."
    }
}

# ---------------- HELPERS ----------------
def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize(IMG_SIZE, Image.BILINEAR)
    arr = np.asarray(image).astype(np.float32)
    arr = preprocess_input(arr)  # [-1,1]
    return arr

def get_tab_tensor(age, sex, dx_type, loc):
    age_scaled = scaler.transform([[age]])[0]
    feats = np.array([
        float(dx_type_le.transform([dx_type])),
        float(age_scaled),
        float(sex_le.transform([sex])),
        float(loc_le.transform([loc])),
    ], dtype=np.float32)
    return np.expand_dims(feats, axis=0)

def get_model_inputs(model):
    img_input, tab_input = None, None
    for inp in model.inputs:
        if len(inp.shape) == 4:  # (None, H, W, C)
            img_input = inp
        elif len(inp.shape) == 2:  # (None, features)
            tab_input = inp
    return img_input, tab_input

def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
    raise ValueError("No Conv2D layer found in model.")

def make_gradcam_heatmap(img_array, tab_array, model, class_index=None):
    image_input, tab_input = get_model_inputs(model)
    last_conv_layer = get_last_conv_layer(model)
    grad_model = tf.keras.models.Model(
        [image_input, tab_input],
        [last_conv_layer.output, model.output],
    )
    img_batch = np.expand_dims(img_array, axis=0)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model([img_batch, tab_array], training=False)
        if class_index is None:
            class_index = tf.argmax(predictions[0])
        class_channel = predictions[:, class_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), predictions.numpy()

def render_gradcam_views(img_array, heatmap, alpha=0.35):
    orig = ((img_array + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    heatmap_resized = tf.image.resize(heatmap[..., np.newaxis], IMG_SIZE).numpy().squeeze()
    import matplotlib.cm as cm
    hm_255 = (heatmap_resized * 255.0).astype(np.uint8)
    jet = cm.get_cmap("jet")
    heatmap_rgb = (jet(hm_255)[:, :, :3] * 255).astype(np.uint8)
    overlay = (alpha * heatmap_rgb + (1 - alpha) * orig).astype(np.uint8)
    return orig, heatmap_rgb, overlay

def softmax_safe(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-12)

# ---------------- MAIN UI: NAVIGATION ---------------
st.sidebar.markdown("## 🔎 Navigation")
app_section = st.sidebar.radio("Jump to",
                               ["🏠 Home", "📤 Upload & Diagnose", "📈 Progress & History", "📑 Report", "ℹ About"])

# ------ Home ------
if app_section == "🏠 Home":
    st.markdown("<h1 style='color:#FF4B4B;font-size:2.5em;'>🧠 Skin Cancer Multimodal Diagnosis</h1>", unsafe_allow_html=True)
    st.caption("AI-powered tool for educational skin lesion assessment. No patient data leaves your device.")
    st_lottie(loading_anim, key="intro-lottie", height=180)
    st.markdown("""
    <div style="padding:18px 28px;background:#fff4e7;border-radius:18px;max-width:700px;">
    <b>How to use?</b><br>
    - Upload a clear skin lesion photo.<br>
    - Enter patient details.<br>
    - Instantly get diagnosis, Grad-CAM visuals, PDF reports, and advice.<br>
    <br>
    <i><b>Disclaimer:</b> This is NOT a substitute for real medical care. Always consult a dermatologist for any concern.</i>
    </div>
    """, unsafe_allow_html=True)

# ------ Upload & Diagnose ------
if app_section == "📤 Upload & Diagnose":
    st.warning("⚠ Please upload only skin lesion images. Other types of medical images (e.g., X-rays, MRI) are not supported.", icon="🚨")
    st.markdown("""
    #### Supported Skin Lesion Types
    <div style="padding:9px 18px;background:#e8f1fb;border-radius:14px;margin-bottom:1.2em;">
    1. nv (<i>Melanocytic Nevi – common mole</i>)<br>
    2. bcc (<i>Basal Cell Carcinoma</i>)<br>
    3. bkl (<i>Benign Keratosis-like Lesions</i>)<br>
    4. vasc (<i>Vascular Lesions</i>)<br>
    5. akiec (<i>Actinic Keratoses / Intraepithelial Carcinoma</i>)<br>
    6. mel (<i>Melanoma</i>)<br>
    7. df (<i>Dermatofibroma</i>)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<h2 style='color:#FA6A6A;'>📤 Upload Patient Data</h2>", unsafe_allow_html=True)
    with st.form("upload_form", clear_on_submit=False):
        c1, c2 = st.columns(2)
        with c1:
            uploaded_file = st.file_uploader("Skin Lesion Image", type=["jpg", "jpeg", "png"],
                                             help="Clear, close-up preferred.", accept_multiple_files=False)
            if uploaded_file:
                st.image(uploaded_file, use_column_width=True, caption="Uploaded Image Preview", output_format="PNG")
        with c2:
            patient_name = st.text_input("Patient Name", max_chars=100)
            patient_mobile = st.text_input("Patient Mobile Number", max_chars=20, help="Include country code if applicable")
            age = st.number_input("Age", min_value=0, max_value=120, value=45)
            sex = st.selectbox("Sex", sex_le.classes_, help="Biological sex as listed in patient record.")
            dx_type = st.selectbox("Dx Type", dx_type_le.classes_, help="Clinical assessment label (e.g. BCC).")
            loc = st.selectbox("Localization", loc_le.classes_, help="Body site.")
            symptoms = st.multiselect("Symptoms (optional)", ["Growth", "Color change", "Bleeding", "Itching"])

        submitted = st.form_submit_button("🔍 Diagnose")

    if submitted and uploaded_file:
        st_lottie(loading_anim, loop=True, height=90, key="spinner")
        try:
            img_array = preprocess_image(uploaded_file)
            tab_array = get_tab_tensor(age, sex, dx_type, loc)
            heatmap, preds = make_gradcam_heatmap(img_array, tab_array, model)
            preds = np.asarray(preds).reshape(-1)
            if preds.shape[0] != len(le.classes_):
                preds = softmax_safe(preds)
            pred_idx = int(np.argmax(preds))
            pred_label = le.classes_[pred_idx]
            confidence = float(preds[pred_idx])
        except Exception as e:
            st.error("⚠ There was an error processing this image. Try another or contact admin.")
            st.stop()

        st_lottie(success_anim, height=80, key="done-suc")
        st.markdown(
            f"""<div class='result-card'>
            <h2>🧬 <span style='color:#eb354c'>{pred_label}</span></h2>
            <h5>Confidence: <span style='color:#247c49'>{confidence:.1%}</span></h5>
            <b>Patient Name:</b> {patient_name}<br>
            <b>Patient Mobile:</b> {patient_mobile}<br>
            <b>Age:</b> {age} | <b>Sex:</b> {sex}<br>
            <span style='font-size:1.1em;font-weight:bold;color:#f1835b;'>{'High risk ⛔' if pred_label.lower() == 'mel' or confidence < 0.3 else 'Moderate/Low risk ✅'}</span>
            </div>""", unsafe_allow_html=True
        )
        fig_conf = px.bar(
            x=le.classes_, y=preds,
            labels={"x": "Class", "y": "Confidence"},
            title="AI Confidence in Each Diagnosis",
            color=le.classes_,
            color_discrete_sequence=px.colors.sequential.RdPu
        )
        fig_conf.update_traces(text=[f"{p:.1%}" for p in preds], textposition="outside")
        fig_conf.update_yaxes(range=[0, 1])
        st.plotly_chart(fig_conf, use_container_width=True)

        # --- GradCAM visuals ---
        orig, hm_rgb, overlay = render_gradcam_views(img_array, heatmap)
        grad_cols = st.columns(3)
        grad_cols[0].image(orig, caption="Original", use_column_width=True)
        grad_cols[1].image(hm_rgb, caption="Grad-CAM Heatmap", use_column_width=True)
        grad_cols[2].image(overlay, caption="Overlay", use_column_width=True)

        # --- Recommendation ---
        info = rec_map.get(pred_label.lower(), {
            "rec": "Unrecognized lesion type. Consult dermatologist urgently.",
            "prev": "General sun safety and monthly skin checks."
        })
        st.markdown(f"""<div class='recommend-card'>
        <b>Doctor's Recommendation:</b><br> {info['rec']}<br>
        <b>Prevention:</b> {info['prev']}
        </div>""", unsafe_allow_html=True)

        # --- Benign / Normal Skin Alert ---
        benign_labels = ["nv", "bkl", "vasc", "df"]
        if pred_label.lower() in benign_labels and confidence > 0.6:
            st.info("✅ You are likely NOT affected by skin cancer based on this image. Continue to monitor your skin and consult a doctor if you notice changes.")

        # --- Success/Warning Feedback ---
        if confidence < 0.5:
            st.warning("🟡 Low confidence: Please consult a real doctor and do not rely solely on this prediction.")
        else:
            st.success("✅ Diagnosis successful. See results above.")

        # --- Save to session state ---
        if "history" not in st.session_state:
            st.session_state["history"] = []
        current_result = {
            "date": datetime.date.today().isoformat(),
            "patient_name": patient_name,
            "patient_mobile": patient_mobile,
            "prediction": pred_label,
            "confidence": confidence,
            "age": int(age),
            "sex": sex,
            "dx_type": dx_type,
            "loc": loc,
            "symptoms": ", ".join(symptoms)
        }
        if not st.session_state["history"] or st.session_state["history"][-1] != current_result:
            st.session_state["history"].append(current_result)
            st.success("Result saved to Progress!")
    elif submitted and not uploaded_file:
        st.error("Please upload a skin lesion image to proceed.")

# ------- PROGRESS & HISTORY -------
if app_section == "📈 Progress & History":
    st.markdown("<h2 style='color:#407be7;'>📈 Your Prediction Progress</h2>", unsafe_allow_html=True)
    if "history" in st.session_state and st.session_state["history"]:
        df_hist = pd.DataFrame(st.session_state["history"])
        fig_progress = px.line(df_hist, x="date", y="confidence", markers=True, title="Saved Results Over Time")
        fig_progress.update_yaxes(range=[0, 1])
        st.plotly_chart(fig_progress, use_container_width=True)
        st.dataframe(df_hist.sort_values("date", ascending=False), use_container_width=True)
        sideb = st.columns(2)
        with sideb[0]:
            st.download_button("⬇ Download as CSV", df_hist.to_csv(index=False), file_name="results_history.csv")
        with sideb[1]:
            if st.button("🗑 Clear History", type="secondary"):
                st.session_state["history"] = []
                st.info("History cleared.")
    else:
        st.info("No saved results yet! Diagnose a patient to see your history here.")

# ------- REPORT PDF EXPORT -------
if app_section == "📑 Report":
    st.markdown("<h2 style='color:#9b50a2;'>📑 Doctor PDF Report</h2>", unsafe_allow_html=True)
    if "history" in st.session_state and st.session_state["history"]:
        last = st.session_state["history"][-1]
        st.write("Latest Result in Report: ", last)
        if st.button("📑 Export PDF", use_container_width=True):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=13)
            pdf.set_text_color(255, 81, 72)
            pdf.cell(195, 10, "Skin Cancer AI Prediction Report", ln=True, align="C")
            pdf.ln(10)
            pdf.set_text_color(0)
            for key, val in last.items():
                if key != "symptoms":
                    pdf.multi_cell(pdf.epw, 10, f"{key.replace('_', ' ').capitalize()}: {val}")
            pdf.ln(5)
            pdf.multi_cell(pdf.epw, 10, f"Symptoms: {last['symptoms']}")
            # Add GradCAM and Charts (recompute for temp files)
            try:
                if 'uploaded_file' in locals():
                    img_array = preprocess_image(uploaded_file)
                    tab_array = get_tab_tensor(last['age'], last['sex'], last['dx_type'], last['loc'])
                    heatmap, preds = make_gradcam_heatmap(img_array, tab_array, model)
                    orig, hm_rgb, overlay = render_gradcam_views(img_array, heatmap)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_overlay:
                        Image.fromarray(overlay).save(tmp_overlay.name)
                        pdf.ln(10)
                        pdf.multi_cell(pdf.epw, 10, "Grad-CAM Overlay:")
                        pdf.image(tmp_overlay.name, w=100)
            except Exception:
                pass
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                pdf.output(tmp_pdf.name)
                with open(tmp_pdf.name, "rb") as f:
                    st.download_button(
                        "⬇ Download Doctor Report (PDF)",
                        f,
                        file_name="skin_cancer_report.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
    else:
        st.info("You need to save at least one result before exporting a report.")

# ------- ABOUT ---------
if app_section == "ℹ About":
    st.markdown("""
        <h2 style='color:#465e8a;'>ℹ About the Skin Cancer AI App</h2>
        <div style='padding:1.2em;background:#f8f9fe;border-radius:14px;max-width:700px;'>
        <b>Tech stack:</b> Streamlit, TensorFlow, Plotly, FPDF, Python.<br>
        <b>Features:</b> Image + metadata multimodal diagnosis, Grad-CAM, rich history, PDF, and CSV export.<br>
        <b>Feedback:</b> This demo is for research/education. Please report bugs or give feedback via GitHub/email.<br>
        </div>
    """, unsafe_allow_html=True)

st.caption("🛡 All processing is local. No user info leaves your device. | Created by Kesavan | vNext-2025")
