import os, re, json, time, warnings, subprocess, signal
warnings.filterwarnings("ignore")

import numpy as np
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download

print("APP STARTED")

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_PATH = ""
HF_MODEL_REPO = "Jaykumardas/Multilingual_News_Model"

# ── Load model ────────────────────────────────────────────────────────────────
def load_model_and_labels():
    model_source = HF_MODEL_REPO if HF_MODEL_REPO else MODEL_PATH
    print(f"[INFO] Loading from: {model_source}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_source)
        print("[INFO] Tokenizer loaded OK")
    except Exception as e:
        raise RuntimeError(f"Tokenizer load failed: {e}")

    id2label = None
    lmap = os.path.join(model_source, "label_map.json")
    try:
        lmap_path = hf_hub_download(
        repo_id=model_source,
        filename="label_map.json"
    )
        with open(lmap_path, encoding="utf-8") as f:
            lm = json.load(f)
        id2label = {int(k): v for k, v in lm["id2label"].items()}
        print(f"[INFO] id2label loaded from HF: {id2label}")

    except Exception as e:
        print(f"[WARN] label_map.json not found in HF repo: {e}")

    if id2label is None:
        cfg_path = os.path.join(model_source, "config.json")
        if os.path.isfile(cfg_path):
            with open(cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)
            if cfg.get("id2label"):
                id2label = {int(k): v for k, v in cfg["id2label"].items()}
                print(f"[INFO] id2label from config.json: {id2label}")

    if id2label is None:
        raise RuntimeError("label_map.json not found. Re-run your save cell in Kaggle.")

    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_source, num_labels=len(id2label), ignore_mismatched_sizes=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device).eval()
        print(f"[INFO] Model OK — {len(id2label)} classes — {device.upper()}")
    except Exception as e:
        raise RuntimeError(f"Model load failed: {e}")

    return model, tokenizer, id2label, device

try:
    MODEL, TOKENIZER, ID2LABEL, DEVICE = load_model_and_labels()
    CLASS_NAMES  = [ID2LABEL[i] for i in sorted(ID2LABEL)]
    NUM_CLASSES  = len(CLASS_NAMES)
    MODEL_LOADED = True
    print(f"[INFO] Classes: {CLASS_NAMES}")
except Exception as e:
    print(f"[ERROR] {e}")
    MODEL_LOADED = False
    CLASS_NAMES  = ["Model not loaded"]
    NUM_CLASSES  = 1
    ID2LABEL     = {0: "Model not loaded"}
    DEVICE       = "cpu"

# ── Icons / metrics / samples ─────────────────────────────────────────────────
ICONS = {
    "entertainment":"🎬","sports":"🏏","state":"🗺️","national":"🇮🇳",
    "international":"🌏","business":"📈","technology":"💻","science":"🔬",
    "health":"🏥","politics":"🏛️",
}
ICONS.update({k.title(): v for k, v in list(ICONS.items())})

def get_icon(label): return ICONS.get(label, "📰")

REAL_METRICS = {
    "TF-IDF + LR":  {"test_acc":83.84,"test_f1":77.85,"color":"#3b82f6","train_time":"< 2 min"},
    "BiLSTM":       {"test_acc":79.36,"test_f1":67.16,"color":"#8b5cf6","train_time":"~14 min"},
    "XLM-RoBERTa":  {"test_acc":86.12,"test_f1":78.75,"color":"#10b981","train_time":"~45 min"},
}

SAMPLES = {
    "Telugu":   "హైదరాబాద్‌లో క్రికెట్ టోర్నమెంట్ ప్రారంభమైంది; జిల్లా స్థాయి జట్లు పాల్గొంటున్నాయి.",
    "Malayalam":"కേരളത്തിൽ ഇന്ന് കനത്ത മഴ; ഒൻപത് ജില്ലകളിൽ യെല്ലോ അലർട്ട് പ്രഖ്യാപിച്ചു.",
    "Marathi":  "मुंबई शेअर बाजारात आज मोठी तेजी; सेन्सेक्स ५०० अंकांनी वधारला.",
    "Tamil":    "தமிழ்நாட்டில் புதிய தொழில்நுட்ப பூங்கா திறப்பு; ஆயிரக்கணக்கான வேலை வாய்ப்புகள்.",
    "Gujarati": "ગુજરાત ટીમ સ્ટેટ ક્રિકેટ ચેમ્પિયનશિપ જીતી; ખેલાડીઓ ઉત્સાહિત.",
}

# ── Preprocessing ─────────────────────────────────────────────────────────────
def clean_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[\u200b\u200c\u200d\ufeff\u00ad]", "", text)
    text = re.sub(
        r"[^\w\s\u0900-\u097F\u0C00-\u0C7F\u0D00-\u0D7F\u0B80-\u0BFF\u0A80-\u0AFF]",
        " ", text)
    return re.sub(r"\s+", " ", text).strip()

# ── Inference ─────────────────────────────────────────────────────────────────
def predict_text(text):
    if not MODEL_LOADED:
        return {c: 0.0 for c in CLASS_NAMES}, "Model not loaded", 0.0, 0
    t_clean = clean_text(text)
    if not t_clean:
        return {c: 0.0 for c in CLASS_NAMES}, "Empty input", 0.0, 0
    enc = TOKENIZER(t_clean, max_length=128, padding="max_length",
                    truncation=True, return_tensors="pt")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    t0 = time.time()
    with torch.no_grad():
        logits = MODEL(**enc).logits
    ms = int((time.time() - t0) * 1000)
    probs = F.softmax(logits, dim=-1).squeeze().cpu().numpy()
    idx   = int(np.argmax(probs))
    label = ID2LABEL.get(idx, f"class_{idx}")
    return ({ID2LABEL.get(i, f"class_{i}"): float(probs[i]) for i in range(len(probs))},
            label, float(probs[idx]), ms)

# ── Charts ────────────────────────────────────────────────────────────────────
def conf_chart(probs_dict, pred_label):
    paired   = sorted(zip(probs_dict.values(), probs_dict.keys()), reverse=True)
    vals     = [p[0]*100 for p in paired]
    labs     = [p[1]     for p in paired]
    colors   = ["#10b981" if l == pred_label else "#6366f1" if v > 10 else "#334155"
                for l, v in zip(labs, vals)]
    fig, ax  = plt.subplots(figsize=(9, max(4, len(labs)*0.5+1)))
    fig.patch.set_facecolor("#0f172a"); ax.set_facecolor("#0f172a")
    bars = ax.barh(labs[::-1], vals[::-1], color=colors[::-1], height=0.55, edgecolor="none")
    for bar, v in zip(bars, vals[::-1]):
        ax.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
                f"{v:.1f}%", va="center", ha="left", color="#e2e8f0", fontsize=10, fontweight="bold")
    ax.set_xlim(0, 115)
    ax.set_xlabel("Confidence (%)", color="#94a3b8", fontsize=11)
    ax.set_title("Prediction Confidence", color="#f1f5f9", fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(colors="#94a3b8", labelsize=10)
    for s in ax.spines.values(): s.set_visible(False)
    ax.grid(axis="x", color="#1e293b", linewidth=0.8)
    plt.tight_layout(pad=1.5)
    return fig

def metrics_chart():
    models = list(REAL_METRICS.keys())
    accs   = [REAL_METRICS[m]["test_acc"] for m in models]
    f1s    = [REAL_METRICS[m]["test_f1"]  for m in models]
    cols   = [REAL_METRICS[m]["color"]    for m in models]
    x, w   = np.arange(len(models)), 0.32
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0f172a"); ax.set_facecolor("#0f172a")
    b1 = ax.bar(x-w/2, accs, w, label="Test Accuracy (%)", color=[c+"cc" for c in cols], edgecolor="none")
    b2 = ax.bar(x+w/2, f1s,  w, label="Test F1 Macro (%)", color=cols, edgecolor="none", alpha=0.75)
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h+0.5, f"{h:.1f}",
                    ha="center", va="bottom", color="#e2e8f0", fontsize=10, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(models, color="#94a3b8", fontsize=11)
    ax.set_ylim(0, 105); ax.set_ylabel("Score (%)", color="#94a3b8", fontsize=11)
    ax.set_title("Model Comparison — Test Results", color="#f1f5f9", fontsize=13, fontweight="bold", pad=14)
    ax.tick_params(colors="#94a3b8")
    ax.legend(facecolor="#1e293b", edgecolor="none", labelcolor="#e2e8f0")
    for s in ax.spines.values(): s.set_visible(False)
    ax.grid(axis="y", color="#1e293b", linewidth=0.8)
    plt.tight_layout(pad=1.5)
    return fig

_METRICS_FIG = metrics_chart()   # pre-render once

# ── Gradio handlers ───────────────────────────────────────────────────────────
def classify_single(text):
    if not text or not text.strip():
        return '<p style="color:#f87171;padding:20px;">Please enter a headline.</p>', None, None

    pd, label, conf, ms = predict_text(text)
    icon = get_icon(label)
    pct  = conf * 100
    cc   = "#10b981" if pct >= 70 else "#f59e0b" if pct >= 40 else "#ef4444"

    html = f"""
<div style="background:linear-gradient(135deg,#1e293b,#0f172a);border:1px solid #334155;
            border-radius:16px;padding:28px 32px;font-family:sans-serif;
            box-shadow:0 8px 32px rgba(0,0,0,0.4);">
  <div style="display:flex;align-items:center;gap:12px;margin-bottom:18px;">
    <span style="font-size:44px;">{icon}</span>
    <div>
      <div style="font-size:11px;text-transform:uppercase;letter-spacing:2px;color:#64748b;font-weight:600;">
        Predicted Category</div>
      <div style="font-size:30px;font-weight:800;color:#f1f5f9;line-height:1.15;">{label.title()}</div>
    </div>
  </div>
  <div style="display:flex;gap:32px;flex-wrap:wrap;">
    <div>
      <div style="font-size:11px;text-transform:uppercase;letter-spacing:1.5px;color:#64748b;margin-bottom:4px;">Confidence</div>
      <div style="font-size:38px;font-weight:900;color:{cc};">{pct:.1f}%</div>
    </div>
    <div>
      <div style="font-size:11px;text-transform:uppercase;letter-spacing:1.5px;color:#64748b;margin-bottom:4px;">Model</div>
      <div style="font-size:16px;font-weight:600;color:#94a3b8;">XLM-RoBERTa</div>
    </div>
    <div>
      <div style="font-size:11px;text-transform:uppercase;letter-spacing:1.5px;color:#64748b;margin-bottom:4px;">Inference</div>
      <div style="font-size:16px;font-weight:600;color:#94a3b8;">{ms} ms</div>
    </div>
  </div>
  <hr style="border:none;border-top:1px solid #1e293b;margin:18px 0 10px;">
  <div style="font-size:12px;color:#475569;">
    IndicGLUE &nbsp;·&nbsp; 5 languages &nbsp;·&nbsp; {NUM_CLASSES} categories &nbsp;·&nbsp; Test acc: 86.12%
  </div>
</div>"""
    return html, conf_chart(pd, label), pd

def classify_batch(batch_text):
    if not batch_text or not batch_text.strip():
        return '<p style="color:#f87171;padding:20px;">Enter at least one headline.</p>', None
    lines = [l.strip() for l in batch_text.strip().split("\n") if l.strip()][:50]
    rows  = ""
    labels_list = []
    for i, line in enumerate(lines, 1):
        pd, label, conf, _ = predict_text(line)
        icon = get_icon(label); pct = conf*100
        cc   = "#10b981" if pct >= 70 else "#f59e0b" if pct >= 40 else "#ef4444"
        prev = (line[:80]+"…") if len(line) > 80 else line
        labels_list.append(label)
        rows += f"""<tr style="border-bottom:1px solid #1e293b;">
          <td style="padding:10px 8px;color:#64748b;font-size:13px;">{i}</td>
          <td style="padding:10px 8px;color:#cbd5e1;font-size:13px;max-width:340px;word-break:break-word;">{prev}</td>
          <td style="padding:10px 8px;font-size:14px;color:#e2e8f0;">{icon} {label.title()}</td>
          <td style="padding:10px 8px;font-weight:700;color:{cc};font-size:14px;">{pct:.1f}%</td>
        </tr>"""
    from collections import Counter
    counts  = Counter(labels_list)
    summary = " · ".join(f"{get_icon(k)} {k.title()}: {v}" for k,v in counts.most_common(5))
    table   = f"""
<div style="background:#0f172a;border-radius:14px;padding:20px;
            font-family:sans-serif;border:1px solid #1e293b;">
  <div style="font-size:12px;color:#64748b;margin-bottom:14px;text-transform:uppercase;letter-spacing:1.5px;">
    {len(lines)} headlines — {summary}</div>
  <div style="overflow-x:auto;">
  <table style="width:100%;border-collapse:collapse;">
    <thead><tr style="border-bottom:2px solid #334155;">
      <th style="padding:8px;color:#475569;font-size:11px;text-align:left;text-transform:uppercase;">#</th>
      <th style="padding:8px;color:#475569;font-size:11px;text-align:left;text-transform:uppercase;">Headline</th>
      <th style="padding:8px;color:#475569;font-size:11px;text-align:left;text-transform:uppercase;">Category</th>
      <th style="padding:8px;color:#475569;font-size:11px;text-align:left;text-transform:uppercase;">Conf.</th>
    </tr></thead>
    <tbody style="color:#e2e8f0;">{rows}</tbody>
  </table></div>
</div>"""
    # Pie chart
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("#0f172a"); ax.set_facecolor("#0f172a")
    pal = ["#10b981","#6366f1","#f59e0b","#ef4444","#3b82f6","#8b5cf6","#ec4899","#14b8a6","#f97316","#84cc16"]
    cd  = dict(counts)
    wedges, texts, ats = ax.pie(cd.values(), labels=[k.title() for k in cd],
        autopct="%1.0f%%", colors=pal[:len(cd)], startangle=140,
        wedgeprops={"edgecolor":"#0f172a","linewidth":2})
    for t in texts:  t.set_color("#94a3b8"); t.set_fontsize(10)
    for at in ats:   at.set_color("#0f172a"); at.set_fontweight("bold"); at.set_fontsize(9)
    ax.set_title("Category Distribution", color="#f1f5f9", fontsize=13, fontweight="bold", pad=14)
    plt.tight_layout()
    return table, fig

# ── CSS ───────────────────────────────────────────────────────────────────────
# IMPORTANT: No @import (blocked in Kaggle). No body/html background override
# (breaks Kaggle iframe rendering). Only style our own named classes.
CSS = """
* { box-sizing: border-box; }
.gradio-container {
    max-width: 1100px !important;
    margin: 0 auto !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}
.app-header {
    background: linear-gradient(135deg, #0f172a, #1e1b4b 50%, #0f172a);
    border: 1px solid #1e293b; border-radius: 14px;
    padding: 32px 40px 24px; text-align: center; margin-bottom: 8px;
}
.header-badge {
    display: inline-block; background: linear-gradient(90deg, #6366f1, #8b5cf6);
    color: white; font-size: 10px; font-weight: 700; letter-spacing: 2.5px;
    text-transform: uppercase; padding: 4px 14px; border-radius: 20px; margin-bottom: 14px;
}
.header-title { font-size: 38px; font-weight: 800; color: #f1f5f9; line-height: 1.1; margin: 0 0 8px; }
.header-title span {
    background: linear-gradient(90deg, #6366f1, #10b981);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.header-sub { font-size: 14px; color: #64748b; margin: 0; }
.header-stats { display: flex; justify-content: center; gap: 16px; margin-top: 20px; flex-wrap: wrap; }
.stat-pill {
    background: #1e293b; border: 1px solid #334155; border-radius: 8px;
    padding: 7px 16px; font-size: 12px; color: #94a3b8;
}
.stat-pill strong { color: #e2e8f0; }
.tab-nav { background: #0f172a !important; border-bottom: 1px solid #1e293b !important; }
.tab-nav button {
    color: #64748b !important; font-weight: 600 !important; font-size: 13px !important;
    padding: 12px 20px !important; border: none !important;
    border-bottom: 2px solid transparent !important; background: transparent !important;
}
.tab-nav button.selected { color: #6366f1 !important; border-bottom-color: #6366f1 !important; }
textarea, input[type=text] {
    background: #1e293b !important; border: 1px solid #334155 !important;
    color: #e2e8f0 !important; border-radius: 10px !important; font-size: 14px !important;
}
label { color: #94a3b8 !important; font-size: 12px !important; text-transform: uppercase !important; }
button.primary {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important; font-weight: 700 !important;
    border: none !important; border-radius: 10px !important;
}
button.secondary {
    background: #1e293b !important; color: #94a3b8 !important;
    border: 1px solid #334155 !important; border-radius: 8px !important;
}
.app-footer {
    background: #0f172a; border: 1px solid #1e293b; border-radius: 14px;
    padding: 24px 40px; text-align: center; margin-top: 24px;
}
.footer-team { display: flex; justify-content: center; gap: 32px; flex-wrap: wrap; margin-bottom: 14px; }
.footer-member { display: flex; align-items: center; gap: 10px; }
.footer-avatar {
    width: 32px; height: 32px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-weight: 800; font-size: 13px; color: white;
}
.footer-name { font-size: 13px; color: #94a3b8; }
.footer-roll { font-size: 11px; color: #475569; }
.footer-copy { font-size: 12px; color: #64748b; margin-top: 10px; }
footer { display: none !important; }
"""

HEADER = """
<div class="app-header">
  <div class="header-badge">Generative AI Assignment &middot; CBIT &middot; 2025-26</div>
  <h1 class="header-title">Multilingual News<br><span>Classification</span></h1>
  <p class="header-sub">Chaitanya Bharathi Institute of Technology &middot; Dept. of AI &amp; ML</p>
  <div class="header-stats">
    <div class="stat-pill">Model <strong>XLM-RoBERTa</strong></div>
    <div class="stat-pill">Languages <strong>5 Indic</strong></div>
    <div class="stat-pill">Dataset <strong>IndicGLUE</strong></div>
    <div class="stat-pill">Test Acc <strong>86.12%</strong></div>
  </div>
</div>
"""

FOOTER = """
<div class="app-footer">
  <div class="footer-team">
    <div class="footer-member">
      <div class="footer-avatar" style="background:linear-gradient(135deg,#6366f1,#8b5cf6);">J</div>
      <div><div class="footer-name">Jay Kumar Das</div><div class="footer-roll">160123748035</div></div>
    </div>
    <div class="footer-member">
      <div class="footer-avatar" style="background:linear-gradient(135deg,#10b981,#059669);">S</div>
      <div><div class="footer-name">Siddhartha Dontula</div><div class="footer-roll">160123748036</div></div>
    </div>
    <div class="footer-member">
      <div class="footer-avatar" style="background:linear-gradient(135deg,#f59e0b,#d97706);">P</div>
      <div><div class="footer-name">Praneeth Reddy Ganta</div><div class="footer-roll">160123748037</div></div>
    </div>
  </div>
  <div class="footer-copy">
    &copy; 2025-26 &middot; Dept. of AI &amp; ML &middot; CBIT Hyderabad &middot;
    Guided by <strong style="color:#64748b;">Mr. Panigrahi Srikanth</strong>
  </div>
</div>
"""

PROJECT_HTML = """
<div style="font-family:sans-serif;padding:8px 0;">
  <div style="background:#1e293b;border:1px solid #334155;border-radius:12px;padding:20px 24px;margin-bottom:16px;">
    <h3 style="color:#e2e8f0;margin:0 0 8px;">Problem Statement</h3>
    <p style="color:#94a3b8;font-size:13px;line-height:1.6;margin:0;">
      A single unified model that reads Telugu, Malayalam, Marathi, Tamil, and Gujarati natively,
      classifying news headlines into up to 10 categories — no translation required.
    </p>
  </div>
  <div style="background:#1e293b;border:1px solid #334155;border-radius:12px;padding:20px 24px;margin-bottom:16px;">
    <h3 style="color:#e2e8f0;margin:0 0 8px;">Dataset — IndicGLUE (ai4bharat/indic_glue)</h3>
    <p style="color:#94a3b8;font-size:13px;line-height:1.6;margin:0;">
      iNLTK Headlines subsets &mdash; 37,069 labeled headlines across 5 languages.<br>
      <strong style="color:#e2e8f0;">Split:</strong> Train 25,945 &middot; Val 3,707 &middot; Test 7,414
    </p>
  </div>
  <div style="background:#1e293b;border:1px solid #334155;border-radius:12px;padding:20px 24px;">
    <h3 style="color:#e2e8f0;margin:0 0 8px;">Results (Test Set)</h3>
    <p style="color:#94a3b8;font-size:13px;line-height:1.6;margin:0;">
      <strong style="color:#3b82f6;">TF-IDF + LR:</strong> 83.84% &middot; F1 77.85%<br>
      <strong style="color:#8b5cf6;">BiLSTM:</strong> 79.36% &middot; F1 67.16%<br>
      <strong style="color:#10b981;">XLM-RoBERTa:</strong> 86.% &middot; F1 78.75%
    </p>
  </div>
</div>
"""

TEAM_HTML = """
<div style="font-family:sans-serif;padding:8px 0;">
  <div style="text-align:center;margin-bottom:24px;">
    <div style="font-size:22px;font-weight:800;color:#f1f5f9;">Meet the Team</div>
    <div style="font-size:13px;color:#64748b;margin-top:4px;">
      Dept. of AI &amp; ML &middot; CBIT &middot; Guided by <strong style="color:#94a3b8;">Mr. Panigrahi Srikanth</strong>
    </div>
  </div>
  <div style="background:linear-gradient(135deg,#1e293b,#0f172a);border:1px solid #334155;border-top:3px solid #6366f1;border-radius:14px;padding:22px 26px;margin-bottom:14px;">
    <div style="display:flex;align-items:center;gap:14px;margin-bottom:12px;">
      <div style="width:48px;height:48px;border-radius:50%;background:linear-gradient(135deg,#6366f1,#8b5cf6);display:flex;align-items:center;justify-content:center;font-size:18px;font-weight:800;color:white;">J</div>
      <div>
        <div style="font-size:17px;font-weight:700;color:#f1f5f9;">Jay Kumar Das</div>
        <div style="font-size:11px;color:#6366f1;">160123748035 &middot; Phase 1 Lead</div>
      </div>
    </div>
    <p style="color:#94a3b8;font-size:13px;line-height:1.6;margin:0;">
      IndicGLUE data loading, Unicode-safe preprocessing, TF-IDF baseline (84.95%), EDA.
    </p>
  </div>
  <div style="background:linear-gradient(135deg,#1e293b,#0f172a);border:1px solid #334155;border-top:3px solid #10b981;border-radius:14px;padding:22px 26px;margin-bottom:14px;">
    <div style="display:flex;align-items:center;gap:14px;margin-bottom:12px;">
      <div style="width:48px;height:48px;border-radius:50%;background:linear-gradient(135deg,#10b981,#059669);display:flex;align-items:center;justify-content:center;font-size:18px;font-weight:800;color:white;">S</div>
      <div>
        <div style="font-size:17px;font-weight:700;color:#f1f5f9;">Siddhartha Dontula</div>
        <div style="font-size:11px;color:#10b981;">160123748036 &middot; Phase 2 Lead</div>
      </div>
    </div>
    <p style="color:#94a3b8;font-size:13px;line-height:1.6;margin:0;">
      BiLSTM design (60k vocab, GlobalMaxPool), training curves, per-class evaluation (79.36%).
    </p>
  </div>
  <div style="background:linear-gradient(135deg,#1e293b,#0f172a);border:1px solid #334155;border-top:3px solid #f59e0b;border-radius:14px;padding:22px 26px;">
    <div style="display:flex;align-items:center;gap:14px;margin-bottom:12px;">
      <div style="width:48px;height:48px;border-radius:50%;background:linear-gradient(135deg,#f59e0b,#d97706);display:flex;align-items:center;justify-content:center;font-size:18px;font-weight:800;color:white;">P</div>
      <div>
        <div style="font-size:17px;font-weight:700;color:#f1f5f9;">Praneeth Reddy Ganta</div>
        <div style="font-size:11px;color:#f59e0b;">160123748037 &middot; Phase 3 Lead</div>
      </div>
    </div>
    <p style="color:#94a3b8;font-size:13px;line-height:1.6;margin:0;">
      XLM-RoBERTa fine-tuning, full evaluation, Gradio UI deployment (86.12%).
    </p>
  </div>
</div>
"""

# ── Build UI ──────────────────────────────────────────────────────────────────
# ONE with gr.Blocks() block. Nothing opens after it closes. No demo.load().
# The metrics chart uses gr.Plot(value=_METRICS_FIG) — renders immediately.

with gr.Blocks(css=CSS, title="Multilingual News Classification") as demo:

    gr.HTML(HEADER)

    with gr.Tabs():

        with gr.Tab("Classify News"):
            with gr.Row():
                with gr.Column(scale=1):
                    txt_in = gr.Textbox(
                        placeholder="Paste a news headline in any of the 5 supported languages...",
                        lines=4, label="News Headline")
                    gr.HTML('<div style="font-size:11px;color:#475569;margin:8px 0 4px;text-transform:uppercase;letter-spacing:1px;">Load Sample</div>')
                    with gr.Row():
                        for lang in ["Telugu", "Malayalam", "Marathi"]:
                            b = gr.Button(lang, size="sm")
                            b.click(fn=lambda l=lang: SAMPLES.get(l,""), outputs=txt_in)
                    with gr.Row():
                        for lang in ["Tamil", "Gujarati"]:
                            b = gr.Button(lang, size="sm")
                            b.click(fn=lambda l=lang: SAMPLES.get(l,""), outputs=txt_in)
                    go_btn = gr.Button("Classify", variant="primary", size="lg")
                with gr.Column(scale=1):
                    res_html  = gr.HTML()
                    res_chart = gr.Plot()
                    res_json  = gr.JSON(visible=False)
            go_btn.click(fn=classify_single,
                         inputs=txt_in,
                         outputs=[res_html, res_chart, res_json])

        with gr.Tab("Batch Classify"):
            gr.HTML('<div style="background:#1e293b;border:1px solid #334155;border-radius:10px;padding:14px 18px;margin-bottom:12px;font-family:sans-serif;font-size:13px;color:#64748b;"><strong style="color:#e2e8f0;">Batch mode</strong> — one headline per line, max 50.</div>')
            with gr.Row():
                with gr.Column(scale=1):
                    batch_in  = gr.Textbox(placeholder="One headline per line...",
                                           lines=12, label="Headlines")
                    batch_btn = gr.Button("Classify All", variant="primary")
                with gr.Column(scale=1):
                    batch_tbl   = gr.HTML()
                    batch_chart = gr.Plot()
            batch_btn.click(fn=classify_batch,
                            inputs=batch_in,
                            outputs=[batch_tbl, batch_chart])

        with gr.Tab("Model Comparison"):
            gr.Plot(value=_METRICS_FIG)   # pre-rendered — no event needed
            with gr.Row():
                for mname, md in REAL_METRICS.items():
                    with gr.Column():
                        gr.HTML(f"""
<div style="background:#1e293b;border:1px solid {md['color']}40;border-top:3px solid {md['color']};border-radius:12px;padding:18px 20px;font-family:sans-serif;">
  <div style="font-size:14px;font-weight:700;color:#f1f5f9;margin-bottom:12px;">{mname}</div>
  <div style="font-size:22px;font-weight:800;color:{md['color']};">{md['test_acc']}%</div>
  <div style="font-size:11px;color:#475569;text-transform:uppercase;">Test Accuracy</div>
  <div style="font-size:22px;font-weight:800;color:{md['color']};margin-top:8px;">{md['test_f1']}%</div>
  <div style="font-size:11px;color:#475569;text-transform:uppercase;">F1 Macro</div>
  <div style="font-size:13px;color:#64748b;margin-top:10px;">{md['train_time']}</div>
</div>""")

        with gr.Tab("Project Details"):
            gr.HTML(PROJECT_HTML)

        with gr.Tab("Team"):
            gr.HTML(TEAM_HTML)

    gr.HTML(FOOTER)

# ── Launch ────────────────────────────────────────────────────────────────────
# Kill any leftover Gradio server first (re-running a Kaggle cell leaves it alive)
def _free_ports():
    for port in range(7860, 7871):
        try:
            r = subprocess.run(["lsof", "-ti", f"tcp:{port}"],
                               capture_output=True, text=True)
            for pid in r.stdout.strip().split("\n"):
                if pid:
                    os.kill(int(pid), signal.SIGKILL)
                    print(f"[INFO] Freed port {port} (killed PID {pid})")
        except Exception:
            pass

_free_ports()
try:
    demo.close()
except Exception:
    pass

import time as _t; _t.sleep(1)

demo.launch(
    share=True,             # Required in Kaggle — generates gradio.live public URL
    server_port=7860,       # Kaggle proxies this port to its output iframe
    server_name="0.0.0.0",
    show_error=True,
    quiet=False,
)
