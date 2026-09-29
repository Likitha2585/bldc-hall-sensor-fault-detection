"""
Live local dashboard for the BLDC fault detection project.

Reads results_summary.csv and phase_reconstruction_results.csv directly
from disk on every page load -- so if you rerun either experiment, just
refresh the browser and the dashboard updates automatically. No numbers
are hardcoded.

Run with:
    python app.py

Then open the URL it prints (usually http://127.0.0.1:5000) in your
browser. Keep the terminal window open while you're viewing it -- closing
it stops the server.
"""

from flask import Flask, render_template_string
import pandas as pd
import json
import os

app = Flask(__name__)

RESULTS_CSV = "results_summary.csv"
RECON_CSV = "exploration/phase_reconstruction_results.csv"

PAGE_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BLDC fault detection -- live results</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
:root {
  --bg:#f7f7f5; --card:#fff; --border:#e3e2dd; --text:#1a1a18; --text2:#6b6a64;
  --accent:#2f5496; --ok:#1d9e75; --warn:#d85a30;
}
@media (prefers-color-scheme: dark) {
  :root { --bg:#171715; --card:#1f1f1c; --border:#333330; --text:#f0f0ee; --text2:#a3a29c;
    --accent:#7fa8e8; --ok:#5dcaa5; --warn:#f0997b; }
}
* { box-sizing: border-box; }
body { background:var(--bg); color:var(--text); font-family:-apple-system,Segoe UI,Roboto,sans-serif;
  max-width:960px; margin:0 auto; padding:1.5rem 1.25rem 3rem; }
h1 { font-size:1.5rem; margin:0 0 0.2rem; }
.sub { color:var(--text2); font-size:0.9rem; margin:0 0 1.5rem; }
.badge { display:inline-block; background:var(--ok); color:#fff; font-size:0.72rem; font-weight:600;
  padding:0.2rem 0.55rem; border-radius:999px; margin-left:0.5rem; vertical-align:middle; }
h2 { font-size:1.1rem; margin:2rem 0 0.6rem; }
.card { background:var(--card); border:1px solid var(--border); border-radius:12px; padding:1.1rem 1.25rem; margin-bottom:1.1rem; }
table { width:100%; border-collapse:collapse; font-size:0.86rem; }
th,td { text-align:left; padding:0.5rem 0.6rem; border-bottom:1px solid var(--border); }
th { color:var(--text2); font-weight:600; font-size:0.75rem; text-transform:uppercase; }
tr.top td { font-weight:600; color:var(--accent); }
.metric-row { display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); gap:0.75rem; margin:1rem 0; }
.metric { background:var(--bg); border-radius:8px; padding:0.7rem 0.85rem; }
.metric-label { font-size:0.7rem; color:var(--text2); text-transform:uppercase; }
.metric-value { font-size:1.25rem; font-weight:600; margin-top:0.1rem; }
.refresh { font-size:0.8rem; color:var(--text2); margin-top:2rem; }
</style>
</head>
<body>

<h1>BLDC fault detection -- live results <span class="badge">LIVE from disk</span></h1>
<p class="sub">Reads {{ results_csv }} and {{ recon_csv }} fresh on every page load. Rerun an experiment, then refresh this page.</p>

<h2>CNN model comparison</h2>
{% if cnn_available %}
<div class="metric-row">
  <div class="metric"><div class="metric-label">Best model</div><div class="metric-value">{{ best_model }}</div></div>
  <div class="metric"><div class="metric-label">Best avg accuracy</div><div class="metric-value">{{ best_acc }}%</div></div>
  <div class="metric"><div class="metric-label">Models compared</div><div class="metric-value">{{ num_models }}</div></div>
  <div class="metric"><div class="metric-label">Conditions</div><div class="metric-value">{{ num_conditions }}</div></div>
</div>
<div class="card"><canvas id="cnnChart"></canvas></div>
<div class="card"><table>{{ cnn_table|safe }}</table></div>
{% else %}
<div class="card">No {{ results_csv }} found in this folder yet. Run BLDC_Hall_Detection_full.py first.</div>
{% endif %}

<h2>Sensor reconstruction experiment</h2>
{% if recon_available %}
<div class="card"><canvas id="reconChart"></canvas></div>
<div class="card"><table>{{ recon_table|safe }}</table></div>
{% else %}
<div class="card">No {{ recon_csv }} found in this folder yet. Run phase_reconstruction_experiment.py first.</div>
{% endif %}

<p class="refresh">Refresh this page any time to reload the latest data from disk.</p>

<script>
{% if cnn_available %}
new Chart(document.getElementById('cnnChart'), {
  type: 'bar',
  data: {{ cnn_chart_data|safe }},
  options: { responsive:true, scales:{ y:{ beginAtZero:true, max:100 } } }
});
{% endif %}
{% if recon_available %}
new Chart(document.getElementById('reconChart'), {
  type: 'line',
  data: {{ recon_chart_data|safe }},
  options: { responsive:true, scales:{ y:{ beginAtZero:true, max:1 } } }
});
{% endif %}
</script>

</body>
</html>
"""

def load_cnn_data():
    if not os.path.exists(RESULTS_CSV):
        return None
    df = pd.read_csv(RESULTS_CSV)
    df["accuracy_pct"] = (df["test_accuracy"] * 100).round(1)
    pivot = df.pivot_table(index="model", columns="condition", values="accuracy_pct")
    pivot["average"] = pivot.mean(axis=1).round(1)
    pivot = pivot.sort_values("average", ascending=False)
    conditions = [c for c in pivot.columns if c != "average"]

    table_html = "<tr><th>Model</th>" + "".join(f"<th>{c}</th>" for c in conditions) + "<th>Average</th></tr>"
    for i, (model, row) in enumerate(pivot.iterrows()):
        cls = ' class="top"' if i == 0 else ""
        cells = "".join(f"<td>{row[c]}</td>" for c in conditions)
        table_html += f"<tr{cls}><td>{model}</td>{cells}<td>{row['average']}</td></tr>"

    colors = ["#378ADD", "#1D9E75", "#D85A30", "#7F77DD", "#F0997B"]
    datasets = [
        {"label": str(cond), "data": pivot[cond].tolist(), "backgroundColor": colors[i % len(colors)]}
        for i, cond in enumerate(conditions)
    ]
    chart_data = {"labels": pivot.index.tolist(), "datasets": datasets}

    return {
        "table_html": table_html,
        "chart_data": json.dumps(chart_data),
        "best_model": pivot.index[0],
        "best_acc": pivot["average"].iloc[0],
        "num_models": len(pivot),
        "num_conditions": len(conditions),
    }

def load_recon_data():
    if not os.path.exists(RECON_CSV):
        return None
    df = pd.read_csv(RECON_CSV)
    table_html = "<tr>" + "".join(f"<th>{c}</th>" for c in df.columns) + "</tr>"
    for _, row in df.iterrows():
        table_html += "<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>"

    chart_data = {
        "labels": df["condition"].tolist(),
        "datasets": [
            {"label": "ib correlation", "data": df["ib_correlation"].tolist(),
             "borderColor": "#378ADD", "backgroundColor": "#378ADD", "tension": 0.25},
            {"label": "ic correlation", "data": df["ic_correlation"].tolist(),
             "borderColor": "#D85A30", "backgroundColor": "#D85A30", "tension": 0.25},
        ],
    }
    return {"table_html": table_html, "chart_data": json.dumps(chart_data)}

@app.route("/")
def dashboard():
    cnn = load_cnn_data()
    recon = load_recon_data()
    return render_template_string(
        PAGE_TEMPLATE,
        results_csv=RESULTS_CSV,
        recon_csv=RECON_CSV,
        cnn_available=cnn is not None,
        recon_available=recon is not None,
        cnn_table=cnn["table_html"] if cnn else "",
        cnn_chart_data=cnn["chart_data"] if cnn else "{}",
        best_model=cnn["best_model"] if cnn else "",
        best_acc=cnn["best_acc"] if cnn else "",
        num_models=cnn["num_models"] if cnn else "",
        num_conditions=cnn["num_conditions"] if cnn else "",
        recon_table=recon["table_html"] if recon else "",
        recon_chart_data=recon["chart_data"] if recon else "{}",
    )

if __name__ == "__main__":
    print("Starting live dashboard server...")
    print("Open http://127.0.0.1:5000 in your browser")
    print("Press Ctrl+C to stop")
    app.run(debug=True, port=5000)

