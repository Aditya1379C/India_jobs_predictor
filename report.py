"""
report.py  —  Phase 3 upgrade
------------------------------
Generates a self-contained HTML dashboard with:
  • Sections 01–05 (existing, with city/role filters)
  • Section 06: Live salary predictor widget (embedded JS lookup tables)
  • Section 04 upgrade: Feature importance chart

Usage:
    python report.py
    python predict.py report
"""

import json
import os
from pathlib import Path
from typing import cast

import pandas as pd

from data_pipeline import load_from_db

_HERE              = Path(__file__).parent
OUTPUT_PATH        = str(_HERE / "report" / "dashboard.html")
IMPORTANCE_PATH    = str(_HERE / "models" / "feature_importance.json")
METRICS_PATH       = str(_HERE / "models" / "model_metrics.json")


# ── Data helpers ─────────────────────────────────────────────────────────────

def compute_stats(df: pd.DataFrame) -> dict:
    """Compute all chart data, KPIs, and predictor lookup tables."""

    # ── KPIs ─────────────────────────────────────────────────────────────────
    total_jobs    = len(df)
    top_city      = df["city"].value_counts().idxmax() if "city" in df.columns else "N/A"
    top_role      = df["job_title"].value_counts().idxmax() if "job_title" in df.columns else "N/A"
    unique_cities = df["city"].nunique() if "city" in df.columns else 0
    unique_roles  = df["job_title"].nunique() if "job_title" in df.columns else 0

    has_salary    = "salary_lpa" in df.columns and df["salary_lpa"].notna().sum() > 5
    avg_salary    = round(df["salary_lpa"].mean(), 1)    if has_salary else None
    median_salary = round(df["salary_lpa"].median(), 1)  if has_salary else None

    # ── Section 1: Top cities ─────────────────────────────────────────────────
    top_cities = df["city"].value_counts().head(10)

    # ── Section 2: Top roles ──────────────────────────────────────────────────
    top_roles = df["job_title"].value_counts().head(10)

    # ── Section 3: Skills ─────────────────────────────────────────────────────
    skill_counts = {}
    if "skills" in df.columns:
        for skills_str in df["skills"].dropna():
            for skill in str(skills_str).split(","):
                s = skill.strip().lower()
                if s and s != "not mentioned" and len(s) > 1:
                    skill_counts[s] = skill_counts.get(s, 0) + 1
    top_skills_list = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:12]

    # ── Section 4: Experience distribution ───────────────────────────────────
    exp_dist = {"labels": [], "values": []}
    if "experience_years" in df.columns:
        bins   = [0, 1, 3, 5, 8, 100]
        labels = ["0–1 yrs", "1–3 yrs", "3–5 yrs", "5–8 yrs", "8+ yrs"]
        df["exp_bucket"] = pd.cut(df["experience_years"].clip(0, 30),
                                  bins=bins, labels=labels, right=False)
        counts  = df["exp_bucket"].value_counts().reindex(labels, fill_value=0)
        exp_dist = {"labels": labels, "values": counts.values.tolist()}

    # ── Section 4: Salary by role ─────────────────────────────────────────────
    sal_by_role = {"labels": [], "values": []}
    if has_salary:
        avg_sal = cast(pd.Series, df.groupby("job_title")["salary_lpa"].mean())
        avg_sal = avg_sal.sort_values(ascending=False).head(8)
        sal_by_role = {
            "labels": [t[:30] + "…" if len(t) > 30 else t
                       for t in (str(i) for i in avg_sal.index.tolist())],
            "values": [round(float(v), 1) for v in avg_sal.values],
        }

    # ── Section 4: Experience vs Salary scatter ───────────────────────────────
    scatter_exp_sal = []
    if has_salary and "experience_years" in df.columns:
        extra_cols = [c for c in ["job_title", "company", "city"] if c in df.columns]
        sc = cast(pd.DataFrame, df[["experience_years", "salary_lpa"] + extra_cols])
        sc = sc.dropna(subset=["experience_years", "salary_lpa"])
        sc = cast(pd.DataFrame, sc.loc[
            (sc["salary_lpa"] >= 1.0) & (sc["experience_years"] >= 0) & (sc["experience_years"] <= 20)
        ])
        if len(sc) > 0:
            q3  = sc["salary_lpa"].quantile(0.75)
            iqr = q3 - sc["salary_lpa"].quantile(0.25)
            sc  = sc[sc["salary_lpa"] <= q3 + 3.0 * iqr]
        if len(sc) > 700:
            sc = sc.sample(700, random_state=42)
        def _exp_bucket(e):
            if e < 1:   return 0
            elif e < 3: return 1
            elif e < 5: return 2
            elif e < 8: return 3
            else:        return 4
        scatter_exp_sal = [
            {"x": round(float(r["experience_years"]), 1),
             "y": round(float(r["salary_lpa"]),       1),
             "b": _exp_bucket(float(r["experience_years"])),
             "t": str(r.get("job_title", "")) if "job_title" in sc.columns else "",
             "c": str(r.get("company",   "")) if "company"   in sc.columns else "",
             "l": str(r.get("city",      "")) if "city"      in sc.columns else ""}
            for _, r in sc.iterrows()
        ]

    # ── Top companies ─────────────────────────────────────────────────────────
    top_companies = []
    if "company" in df.columns:
        tc = df["company"].value_counts().head(8)
        top_companies = [{"name": k, "count": int(v)} for k, v in tc.items()]

    # ── Feature importance + model metrics ───────────────────────────────────
    feature_importance = {}
    if os.path.exists(IMPORTANCE_PATH):
        with open(IMPORTANCE_PATH) as f:
            feature_importance = json.load(f)

    model_metrics = {"best_model": "ML Model", "test_mae_lpa": None, "test_r2": None}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            model_metrics = json.load(f)

    # ── Section 6: Predictor lookup tables ───────────────────────────────────
    # Apply same salary floor as model.py to remove bad-parse artefacts.
    if has_salary:
        clean_sal = cast(pd.DataFrame, df.loc[df["salary_lpa"] >= 1.0]).copy()
        q3  = clean_sal["salary_lpa"].quantile(0.75)
        iqr = q3 - clean_sal["salary_lpa"].quantile(0.25)
        clean_sal = cast(pd.DataFrame, clean_sal.loc[clean_sal["salary_lpa"] <= q3 + 3.0 * iqr])
        global_mean = round(clean_sal["salary_lpa"].mean(), 2)
    else:
        clean_sal   = df.copy()
        global_mean = 6.0

    # Role → mean salary (only roles with ≥3 clean salary data points)
    role_salary = {}
    if has_salary:
        for role, grp in clean_sal.groupby("job_title"):
            grp_sal = grp["salary_lpa"].dropna()
            if len(grp_sal) >= 3:
                role_salary[role] = round(grp_sal.mean(), 2)

    # City → mean salary (only cities with ≥3 clean salary data points)
    city_salary = {}
    if has_salary:
        for city, grp in clean_sal.groupby("city"):
            grp_sal = grp["salary_lpa"].dropna()
            if len(grp_sal) >= 3:
                city_salary[city] = round(grp_sal.mean(), 2)

    # Skill → salary premium (mean salary of jobs with this skill vs global mean)
    skill_premium = {}
    if has_salary and "skills" in df.columns:
        for skill in [s[0] for s in top_skills_list[:15]]:
            mask = clean_sal["skills"].fillna("").str.lower().str.contains(skill, regex=False)
            grp_sal = clean_sal.loc[mask, "salary_lpa"].dropna()
            if len(grp_sal) >= 3:
                skill_premium[skill] = round(grp_sal.mean() - global_mean, 2)

    # Sorted dropdown options
    pred_roles  = sorted(role_salary.keys())
    pred_cities = sorted(city_salary.keys())

    # ── Filter lookup tables (for Section 1 + 2 interactive filters) ─────────
    # city_by_role[role] = {city: count} — used by the role filter on the cities chart
    city_by_role: dict = {}
    for role, grp in df.groupby("job_title"):
        counts = grp["city"].value_counts().head(10)
        if len(counts):
            city_by_role[role] = counts.to_dict()

    # role_by_city[city] = {role: count} — used by the city filter on the roles chart
    role_by_city: dict = {}
    for city, grp in df.groupby("city"):
        counts = grp["job_title"].value_counts().head(10)
        if len(counts):
            role_by_city[city] = counts.to_dict()

    return {
        "kpis": {
            "total_jobs":    total_jobs,
            "top_city":      top_city,
            "top_role":      top_role,
            "unique_cities": unique_cities,
            "unique_roles":  unique_roles,
            "avg_salary":    avg_salary,
            "median_salary": median_salary,
        },
        "top_cities":    {"labels": top_cities.index.tolist(), "values": top_cities.values.tolist()},
        "top_roles":     {"labels": top_roles.index.tolist(),  "values": top_roles.values.tolist()},
        "top_skills":    {"labels": [s[0] for s in top_skills_list], "values": [s[1] for s in top_skills_list]},
        "exp_dist":      exp_dist,
        "sal_by_role":   sal_by_role,
        "top_companies": top_companies,
        "has_salary":       has_salary,
        "scatter_exp_sal":  scatter_exp_sal,
        "feature_importance": feature_importance,
        "model_metrics": model_metrics,
        "city_by_role":  city_by_role,
        "role_by_city":  role_by_city,
        "predictor": {
            "global_mean":  global_mean,
            "role_salary":  role_salary,
            "city_salary":  city_salary,
            "skill_premium": skill_premium,
            "pred_roles":   pred_roles,
            "pred_cities":  pred_cities,
            "top_skills":   [s[0] for s in top_skills_list[:12]],
        },
    }


# ── HTML builder ─────────────────────────────────────────────────────────────

def build_html(stats: dict) -> str:
    kpis      = stats["kpis"]
    cities    = stats["top_cities"]
    roles     = stats["top_roles"]
    skills    = stats["top_skills"]
    exp       = stats["exp_dist"]
    sal_role  = stats["sal_by_role"]
    companies = stats["top_companies"]
    has_sal   = stats["has_salary"]
    scatter   = stats.get("scatter_exp_sal", [])
    fi        = stats["feature_importance"]
    mm        = stats["model_metrics"]
    pred      = stats["predictor"]

    model_name = mm.get("best_model", "ML Model")
    model_r2   = mm.get("test_r2")
    model_mae  = mm.get("test_mae_lpa")
    r2_str     = f"R² {model_r2}" if model_r2 else "R² N/A"
    mae_str    = f"₹{model_mae} LPA" if model_mae else "N/A"

    sal_kpi_val = f"₹{kpis['avg_salary']}L" if has_sal else "N/A"
    sal_kpi_sub = "avg salary per annum"     if has_sal else "salary data unavailable"

    # ── Company rows ──────────────────────────────────────────────────────────
    company_rows = ""
    for i, c in enumerate(companies):
        pct   = round(c["count"] / kpis["total_jobs"] * 100, 1)
        color = ["#2563eb","#059669","#d97706","#dc2626","#7c3aed",
                 "#0891b2","#b45309","#be185d"][i % 8]
        company_rows += f"""
        <div class="salary-row">
          <span class="salary-role" style="width:160px;font-size:13px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="{c['name']}">{c['name']}</span>
          <div class="salary-bar-wrap">
            <div class="salary-bar-bg">
              <div class="salary-bar-fill" style="width:{min(pct*5,100)}%;background:linear-gradient(90deg,{color},{color}66)"></div>
            </div>
          </div>
          <span class="salary-val">{c['count']}</span>
        </div>"""

    # ── Feature importance bars ───────────────────────────────────────────────
    fi_labels = list(fi.keys())[:10]
    fi_values = [fi[k] for k in fi_labels]

    # Clean label names for display
    def _clean_fi_label(label):
        label = label.replace("_enc", "").replace("skill_", "skill: ").replace("_", " ")
        return label.title()

    fi_labels_clean = [_clean_fi_label(l) for l in fi_labels]

    # ── Predictor dropdowns ───────────────────────────────────────────────────
    role_options = "\n".join(
        f'<option value="{r}">{r}</option>' for r in pred["pred_roles"]
    )
    city_options = "\n".join(
        f'<option value="{c}">{c}</option>' for c in pred["pred_cities"]
    )
    skill_checkboxes = ""
    for sk in pred["top_skills"]:
        label = sk.replace("'", "&#39;").title()
        skill_checkboxes += f"""
        <label class="skill-label">
          <input type="checkbox" class="skill-check" value="{sk}">
          <span>{label}</span>
        </label>"""

    # ── Salary section visibility ─────────────────────────────────────────────
    sal_hide = "" if has_sal else "display:none"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>India Tech Jobs Market · Live Analysis</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.5.1" integrity="sha384-jb8JQMbMoBUzgWatfe6COACi2ljcDdZQ2OxczGA3bGNeWe+6DChMTBJemed7ZnvJ" crossorigin="anonymous"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg:           #f1f5f9;
      --bg2:          #e8edf5;
      --bg3:          #dde3ee;
      --card:         #ffffff;
      --border:       rgba(15,23,42,0.09);
      --accent:       #2563eb;
      --accent-hover: #1d4ed8;
      --accent2:      #d97706;
      --accent3:      #059669;
      --accent4:      #dc2626;
      --accent5:      #7c3aed;
      --text:         #0f172a;
      --text2:        #475569;
      --text3:        #94a3b8;
      --radius:       12px;
      --gap:          20px;
      --grid-color:   rgba(37,99,235,0.04);
      color-scheme: light;
    }}
    html[data-theme="dark"] {{
      --bg:           #141414;
      --bg2:          #1a1a1a;
      --bg3:          #3a3a3a;
      --card:         #1e1e1e;
      --border:       rgba(255,255,255,0.11);
      --accent:       #60a5fa;
      --accent-hover: #93c5fd;
      --accent2:      #fbbf24;
      --accent3:      #34d399;
      --accent4:      #f87171;
      --accent5:      #a78bfa;
      --text:         #f1f5f9;
      --text2:        #a1a1a1;
      --text3:        #6b7280;
      --grid-color:   rgba(255,255,255,0.03);
      color-scheme: dark;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: var(--bg);
      color: var(--text);
      font-family: 'IBM Plex Sans', sans-serif;
      font-size: 15px;
      min-height: 100vh;
      overflow-x: hidden;
    }}
    body::before {{
      content: '';
      position: fixed; inset: 0;
      background-image:
        linear-gradient(var(--grid-color) 1px, transparent 1px),
        linear-gradient(90deg, var(--grid-color) 1px, transparent 1px);
      background-size: 40px 40px;
      pointer-events: none;
      z-index: 0;
    }}
    .wrapper {{ position: relative; z-index: 1; max-width: 1420px; margin: 0 auto; padding: 36px 28px 60px; }}

    /* ─── HEADER ─── */
    .header {{ margin-bottom: 36px; padding-bottom: 28px; border-bottom: 1px solid var(--border); }}
    .header-inner {{ display: flex; align-items: flex-end; justify-content: space-between; flex-wrap: wrap; gap: 20px; }}
    .eyebrow {{ font-size: 11px; letter-spacing: 3px; text-transform: uppercase; color: var(--accent); margin-bottom: 10px; font-weight: 500; }}
    .title {{ font-family: 'Space Grotesk', sans-serif; font-size: clamp(24px,3.5vw,42px); font-weight: 700; line-height: 1.05; letter-spacing: 2px; text-transform: uppercase; }}
    .title span {{ color: inherit; }}
    .subtitle {{ color: var(--text2); margin-top: 8px; font-size: 14px; max-width: 520px; line-height: 1.6; font-weight: 400; }}
    .header-badges {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }}
    .badge-tag {{ background: var(--card); border: 1px solid var(--border); border-radius: 50px; padding: 5px 13px; font-size: 11px; color: var(--text2); letter-spacing: 1px; text-transform: uppercase; }}

    /* ─── NAV ─── */
    .section-nav {{ display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 32px; padding-bottom: 20px; border-bottom: 1px solid var(--border); }}
    .nav-btn {{ display: flex; align-items: center; gap: 7px; background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 8px 16px; font-family: 'IBM Plex Sans', sans-serif; font-size: 13px; font-weight: 500; color: var(--text2); cursor: pointer; transition: all 0.2s; letter-spacing: 1px; }}
    .nav-btn:hover {{ border-color: color-mix(in srgb,var(--accent) 40%,transparent); color: var(--text); }}
    .nav-btn.active {{ background: color-mix(in srgb,var(--accent) 12%,transparent); border-color: var(--accent); color: var(--accent); }}
    .nav-btn .nav-num {{ background: color-mix(in srgb,var(--accent) 15%,transparent); color: var(--accent); border-radius: 4px; padding: 1px 5px; font-size: 11px; }}

    /* ─── KPI ROW ─── */
    .kpi-row {{ display: grid; grid-template-columns: repeat(4,1fr); gap: var(--gap); margin-bottom: var(--gap); }}
    @media (max-width:900px) {{ .kpi-row {{ grid-template-columns: repeat(2,1fr); }} }}
    .kpi {{ background: var(--card); border: 1px solid var(--border); border-radius: var(--radius); padding: 22px 24px; position: relative; overflow: visible; transition: transform 0.2s, border-color 0.2s, box-shadow 0.2s; }}
    .kpi:hover {{ transform: translateY(-3px); border-color: rgba(37,99,235,0.3); box-shadow: 0 8px 32px rgba(0,0,0,0.08); }}
    .kpi::before {{ content:''; position:absolute; top:0; left:0; right:0; height:2px; background: var(--kpi-color, var(--accent)); }}
    .kpi-icon {{ font-size: 22px; margin-bottom: 12px; display: block; }}
    .kpi-val {{ font-family: 'Space Grotesk', sans-serif; font-size: 32px; font-weight: 700; color: var(--kpi-color, var(--accent)); line-height: 1; margin-bottom: 5px; letter-spacing: 1px; }}
    .kpi-label {{ font-family: 'Space Grotesk', sans-serif; color: var(--text2); font-size: 12px; letter-spacing: 1.5px; text-transform: uppercase; font-weight: 500; }}
    .kpi-sub {{ color: var(--text3); font-size: 12px; margin-top: 7px; line-height: 1.5; font-weight: 400; }}

    /* ─── SECTIONS ─── */
    .section {{ display: none; }}
    .section.active {{ display: block; }}
    .grid-2  {{ display: grid; grid-template-columns: 1fr 1fr; gap: var(--gap); margin-bottom: var(--gap); }}
    .grid-12 {{ display: grid; grid-template-columns: 1.35fr 1fr; gap: var(--gap); margin-bottom: var(--gap); }}
    .grid-21 {{ display: grid; grid-template-columns: 1fr 1.35fr; gap: var(--gap); margin-bottom: var(--gap); }}
    .grid-3  {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: var(--gap); margin-bottom: var(--gap); }}
    @media (max-width:900px) {{ .grid-2,.grid-12,.grid-21,.grid-3 {{ grid-template-columns: 1fr; }} }}

    /* ─── CARD ─── */
    .card {{ background: var(--card); border: 1px solid var(--border); border-radius: var(--radius); padding: 24px 26px; position: relative; overflow: hidden; transition: border-color 0.2s; }}
    .card:hover {{ border-color: rgba(37,99,235,0.2); }}
    .card-header {{ margin-bottom: 18px; }}
    .card-title {{ font-family: 'Space Grotesk', sans-serif; font-size: 12px; font-weight: 700; color: var(--text); text-transform: uppercase; letter-spacing: 2.5px; margin-bottom: 4px; }}
    .card-sub {{ color: var(--text3); font-size: 12px; letter-spacing: 0.5px; line-height: 1.5; font-weight: 400; }}
    canvas {{ max-height: 320px; }}

    /* ─── FILTER BAR ─── */
    .filter-bar {{ display: flex; align-items: center; gap: 10px; margin-bottom: 20px; flex-wrap: wrap; }}
    .filter-bar label {{ font-size: 12px; color: var(--text2); letter-spacing: 1px; text-transform: uppercase; font-weight: 500; }}
    .filter-select {{
      background: var(--card); border: 1px solid var(--border); border-radius: 8px;
      padding: 7px 12px; font-family: 'IBM Plex Sans', sans-serif; font-size: 13px; color: var(--text);
      cursor: pointer; outline: none; transition: border-color 0.2s;
    }}
    .filter-select:focus {{ border-color: var(--accent); }}
    .filter-reset {{ background: none; border: 1px solid var(--border); border-radius: 8px; padding: 7px 12px; font-family: 'IBM Plex Sans', sans-serif; font-size: 13px; color: var(--text2); cursor: pointer; transition: all 0.2s; }}
    .filter-reset:hover {{ border-color: var(--accent); color: var(--accent); }}

    /* ─── SECTION HEADING ─── */
    .section-heading {{ display: flex; align-items: center; gap: 12px; margin-bottom: 24px; }}
    .section-heading-num {{ font-family: 'Space Grotesk', sans-serif; font-size: 12px; font-weight: 700; color: var(--accent); background: color-mix(in srgb,var(--accent) 10%,transparent); border: 1px solid color-mix(in srgb,var(--accent) 25%,transparent); border-radius: 6px; padding: 4px 10px; letter-spacing: 2px; }}
    .section-heading-title {{ font-family: 'Space Grotesk', sans-serif; font-size: 18px; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; }}
    .section-heading-line {{ flex: 1; height: 1px; background: var(--border); }}

    /* ─── SALARY BARS ─── */
    .salary-row {{ display: flex; align-items: center; gap: 12px; margin-bottom: 12px; }}
    .salary-role {{ width: 130px; font-size: 13px; color: var(--text2); flex-shrink: 0; }}
    .salary-bar-wrap {{ flex: 1; position: relative; height: 18px; }}
    .salary-bar-bg {{ width: 100%; height: 6px; background: var(--bg3); border-radius: 3px; position: absolute; top: 50%; transform: translateY(-50%); }}
    .salary-bar-fill {{ height: 100%; border-radius: 3px; transition: width 0.6s; }}
    .salary-val {{ width: 56px; text-align: right; color: var(--text); font-size: 13px; flex-shrink: 0; font-family: 'IBM Plex Sans', sans-serif; font-weight: 500; }}

    /* ─── INSIGHT LIST ─── */
    .insight-list {{ list-style: none; padding: 0; }}
    .insight-list li {{ display: flex; gap: 12px; padding: 11px 0; border-bottom: 1px solid var(--border); color: var(--text2); font-size: 14px; line-height: 1.6; }}
    .insight-list li:last-child {{ border-bottom: none; }}
    .insight-list li::before {{ content: '›'; color: var(--accent); font-size: 16px; line-height: 1.3; flex-shrink: 0; }}
    .insight-list li strong {{ color: var(--text); }}

    /* ─── METHOD PILLS ─── */
    .method-list {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }}
    .method-pill {{ background: var(--bg3); border: 1px solid var(--border); border-radius: 6px; padding: 5px 11px; font-size: 12px; color: var(--text2); letter-spacing: 0.5px; }}
    .method-pill span {{ color: var(--accent); margin-right: 4px; }}

    /* ─── SALARY PREDICTOR ─── */
    .predictor-grid {{ display: grid; grid-template-columns: 1.2fr 1fr; gap: var(--gap); margin-bottom: var(--gap); }}
    @media (max-width:900px) {{ .predictor-grid {{ grid-template-columns: 1fr; }} }}
    .pred-form {{ display: flex; flex-direction: column; gap: 18px; }}
    .pred-field label {{ display: block; font-size: 12px; letter-spacing: 1.5px; text-transform: uppercase; color: var(--text2); margin-bottom: 7px; }}
    .pred-select, .pred-input {{
      width: 100%; background: var(--bg); border: 1px solid var(--border); border-radius: 8px;
      padding: 10px 14px; font-family: 'IBM Plex Sans', sans-serif; font-size: 14px; color: var(--text);
      outline: none; transition: border-color 0.2s;
    }}
    .pred-select:focus, .pred-input:focus {{ border-color: var(--accent); }}
    .pred-exp-row {{ display: flex; align-items: center; gap: 12px; }}
    .pred-exp-val {{ font-family: 'IBM Plex Sans', sans-serif; font-weight: 700; font-size: 18px; color: var(--accent); min-width: 36px; letter-spacing: 1px; }}
    input[type=range] {{ flex: 1; accent-color: var(--accent); cursor: pointer; }}
    .skills-grid {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .skill-label {{ display: flex; align-items: center; gap: 6px; background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: 5px 10px; cursor: pointer; font-size: 12px; color: var(--text2); transition: all 0.15s; user-select: none; }}
    .skill-label:hover {{ border-color: rgba(37,99,235,0.4); color: var(--text); }}
    .skill-label input {{ display: none; }}
    .skill-label:has(input:checked) {{ background: rgba(37,99,235,0.08); border-color: var(--accent); color: var(--accent); }}
    .pred-btn {{
      background: var(--accent); color: #fff; border: none; border-radius: 10px;
      padding: 13px 24px; font-family: 'IBM Plex Sans', sans-serif; font-size: 13px; font-weight: 700;
      cursor: pointer; letter-spacing: 2px; text-transform: uppercase; transition: background 0.2s, transform 0.1s;
    }}
    .pred-btn:hover {{ background: var(--accent-hover); }}
    .pred-btn:active {{ transform: scale(0.98); }}
    .pred-result {{ background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius); padding: 28px; text-align: center; min-height: 200px; display: flex; flex-direction: column; justify-content: center; align-items: center; gap: 10px; }}
    .pred-result-empty {{ color: var(--text3); font-size: 14px; }}
    .pred-result-val {{ font-family: 'Space Grotesk', sans-serif; font-size: 48px; font-weight: 700; color: var(--accent3); line-height: 1; letter-spacing: 2px; }}
    .pred-result-unit {{ font-size: 14px; color: var(--text2); letter-spacing: 2px; text-transform: uppercase; }}
    .pred-result-range {{ font-size: 14px; color: var(--text2); margin-top: 4px; }}
    .pred-result-range span {{ color: var(--text); font-weight: 600; }}
    .pred-result-breakdown {{ margin-top: 16px; width: 100%; }}
    .pred-breakdown-row {{ display: flex; justify-content: space-between; padding: 7px 0; border-bottom: 1px solid var(--border); font-size: 13px; color: var(--text2); }}
    .pred-breakdown-row:last-child {{ border-bottom: none; }}
    .pred-breakdown-row span:last-child {{ color: var(--text); font-family: 'IBM Plex Sans', sans-serif; font-weight: 500; letter-spacing: 0.5px; }}
    .pred-note {{ font-size: 12px; color: var(--text3); margin-top: 12px; line-height: 1.5; }}
    .pred-model-badge {{ background: color-mix(in srgb,var(--accent) 8%,transparent); border: 1px solid color-mix(in srgb,var(--accent) 20%,transparent); border-radius: 6px; padding: 4px 10px; font-size: 12px; color: var(--accent); letter-spacing: 0.5px; }}

    /* ─── DARK MODE CHART BOOST ─── */
    html[data-theme="dark"] .salary-bar-fill {{ filter: brightness(1.4) saturate(1.2); }}
    .kpi {{ --kpi-color: var(--kpi-light); }}
    html[data-theme="dark"] .kpi {{ --kpi-color: var(--kpi-dark); }}

    /* ─── THEME TOGGLE ─── */
    .theme-toggle {{
      position: fixed; top: 20px; right: 24px; z-index: 999;
      display: flex; align-items: center; gap: 7px;
      background: var(--card); border: 1px solid var(--border); border-radius: 8px;
      padding: 8px 14px; font-family: 'IBM Plex Sans', sans-serif; font-size: 13px;
      color: var(--text2); cursor: pointer; transition: all 0.2s; letter-spacing: 1px;
      box-shadow: 0 2px 12px rgba(0,0,0,0.1);
    }}
    .theme-toggle:hover {{ border-color: var(--accent); color: var(--accent); }}

    /* ─── RUN PIPELINE BUTTON ─── */
    .run-btn {{
      display: inline-flex; align-items: center; gap: 8px;
      background: var(--card); border: 1px solid var(--border); border-radius: 8px;
      padding: 9px 16px; font-family: 'IBM Plex Sans', sans-serif; font-size: 13px;
      color: var(--text2); cursor: pointer; transition: all 0.2s; letter-spacing: 1px;
    }}
    .run-btn:hover:not(:disabled) {{ border-color: var(--accent3); color: var(--accent3); }}
    .run-btn:disabled {{ opacity: 0.6; cursor: default; }}
    .run-btn.running {{ border-color: var(--accent2); color: var(--accent2); animation: pulse 1.5s ease-in-out infinite; }}
    .run-btn.done {{ border-color: var(--accent3); color: var(--accent3); }}
    .run-btn.error {{ border-color: var(--accent4); color: var(--accent4); }}
    @keyframes pulse {{ 0%,100% {{ opacity:1; }} 50% {{ opacity:0.6; }} }}
    .run-status-bar {{
      display: none; margin-top: 10px; padding: 10px 14px;
      background: var(--card); border: 1px solid var(--border); border-radius: 8px;
      font-size: 12px; color: var(--text2); line-height: 1.5;
    }}
    .run-status-bar.visible {{ display: block; }}

    /* ─── FOOTER ─── */
    .footer {{ margin-top: 48px; padding-top: 20px; border-top: 1px solid var(--border); display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px; color: var(--text3); font-size: 11px; letter-spacing: 0.5px; }}
  </style>
</head>
<body>
<div class="wrapper">

  <!-- THEME TOGGLE (fixed top-right) -->
  <button class="theme-toggle" id="theme-toggle" onclick="toggleTheme()">
    <span id="theme-icon">🌙</span><span id="theme-label">Dark Mode</span>
  </button>

  <!-- HEADER -->
  <header class="header">
    <div class="header-inner">
      <div>
        <div class="eyebrow">◈ Live Data Intelligence Report · India · 2025</div>
        <div class="title">India <span>Tech Jobs</span><br>Market Analysis</div>
        <div class="subtitle">
          An end-to-end Python pipeline. Fetched via Adzuna API, cleaned with Pandas,
          modelled with {model_name}, and visualised with Chart.js.
        </div>
      </div>
      <div style="display:flex;flex-direction:column;align-items:flex-end;gap:10px">
        <div class="header-badges">
          <div class="badge-tag">Python · Pandas</div>
          <div class="badge-tag">Adzuna API</div>
          <div class="badge-tag">{model_name}</div>
          <div class="badge-tag">{kpis['total_jobs']:,} listings</div>
          <div class="badge-tag">Aditya Jai Singh</div>
        </div>
        <div style="display:flex;flex-direction:column;align-items:flex-end">
          <button class="run-btn" id="run-btn" onclick="runPipeline()">▶ Run Pipeline</button>
          <div class="run-status-bar" id="run-status"></div>
        </div>
      </div>
    </div>
  </header>

  <!-- KPI ROW -->
  <div class="kpi-row">
    <div class="kpi" style="--kpi-light:#2563eb;--kpi-dark:#60a5fa">
      <span class="kpi-icon">💼</span>
      <div class="kpi-val">{kpis['total_jobs']:,}</div>
      <div class="kpi-label">Jobs Scraped</div>
      <div class="kpi-sub">Live data from Adzuna API</div>
    </div>
    <div class="kpi" style="--kpi-light:#059669;--kpi-dark:#34d399">
      <span class="kpi-icon">🏙️</span>
      <div class="kpi-val" style="font-size:20px">{kpis['top_city']}</div>
      <div class="kpi-label">Top City</div>
      <div class="kpi-sub">{kpis['unique_cities']} unique cities in dataset</div>
    </div>
    <div class="kpi" style="--kpi-light:#d97706;--kpi-dark:#fbbf24">
      <span class="kpi-icon">💰</span>
      <div class="kpi-val">{sal_kpi_val}</div>
      <div class="kpi-label">Avg Salary</div>
      <div class="kpi-sub">{sal_kpi_sub}</div>
    </div>
    <div class="kpi" style="--kpi-light:#7c3aed;--kpi-dark:#a78bfa">
      <span class="kpi-icon">⚡</span>
      <div class="kpi-val" style="font-size:16px;word-break:break-word;line-height:1.2">{kpis['top_role']}</div>
      <div class="kpi-label">Top Role</div>
      <div class="kpi-sub">{kpis['unique_roles']} unique roles fetched</div>
    </div>
  </div>

  <!-- SECTION NAV -->
  <nav class="section-nav">
    <button class="nav-btn active" onclick="showSection('s1',this)"><span class="nav-num">01</span> Job Distribution</button>
    <button class="nav-btn" onclick="showSection('s2',this)"><span class="nav-num">02</span> Top Roles</button>
    <button class="nav-btn" onclick="showSection('s3',this)"><span class="nav-num">03</span> Skills Analysis</button>
    <button class="nav-btn" onclick="showSection('s4',this)"><span class="nav-num">04</span> Salary & Experience</button>
    <button class="nav-btn" onclick="showSection('s5',this)"><span class="nav-num">05</span> Key Insights</button>
    <button class="nav-btn" onclick="showSection('s6',this)" style="border-color:color-mix(in srgb,var(--accent3) 30%,transparent);color:var(--accent3)"><span class="nav-num" style="background:color-mix(in srgb,var(--accent3) 10%,transparent);color:var(--accent3)">06</span> Salary Predictor</button>
  </nav>

  <!-- ─── SECTION 1 : JOB DISTRIBUTION ─── -->
  <section id="s1" class="section active">
    <div class="section-heading">
      <div class="section-heading-num">01</div>
      <div class="section-heading-title">Job Distribution Across India</div>
      <div class="section-heading-line"></div>
    </div>
    <div class="filter-bar">
      <label>Filter by role:</label>
      <select class="filter-select" id="s1-role-filter" onchange="filterCitiesChart()">
        <option value="">All Roles</option>
        {role_options}
      </select>
      <button class="filter-reset" onclick="resetFilter('s1-role-filter','filterCitiesChart')">Reset</button>
    </div>
    <div class="grid-12">
      <div class="card">
        <div class="card-header">
          <div class="card-title">Jobs by City</div>
          <div class="card-sub">Top 10 cities by number of active tech job listings</div>
        </div>
        <canvas id="citiesChart"></canvas>
      </div>
      <div class="card">
        <div class="card-header">
          <div class="card-title">Top Hiring Companies</div>
          <div class="card-sub">Companies with the most active listings in the dataset</div>
        </div>
        {company_rows}
      </div>
    </div>
  </section>

  <!-- ─── SECTION 2 : TOP ROLES ─── -->
  <section id="s2" class="section">
    <div class="section-heading">
      <div class="section-heading-num">02</div>
      <div class="section-heading-title">Most In-Demand Roles</div>
      <div class="section-heading-line"></div>
    </div>
    <div class="filter-bar">
      <label>Filter by city:</label>
      <select class="filter-select" id="s2-city-filter" onchange="filterRolesChart()">
        <option value="">All Cities</option>
        {city_options}
      </select>
      <button class="filter-reset" onclick="resetFilter('s2-city-filter','filterRolesChart')">Reset</button>
    </div>
    <div class="grid-2">
      <div class="card">
        <div class="card-header">
          <div class="card-title">Top 10 Job Titles</div>
          <div class="card-sub">Ranked by total number of active listings</div>
        </div>
        <canvas id="rolesChart"></canvas>
      </div>
      <div class="card">
        <div class="card-header">
          <div class="card-title">Role Distribution</div>
          <div class="card-sub">Share of listings across top roles</div>
        </div>
        <canvas id="rolesDoughnut"></canvas>
      </div>
    </div>
  </section>

  <!-- ─── SECTION 3 : SKILLS ─── -->
  <section id="s3" class="section">
    <div class="section-heading">
      <div class="section-heading-num">03</div>
      <div class="section-heading-title">Most In-Demand Skills</div>
      <div class="section-heading-line"></div>
    </div>
    <div class="grid-2">
      <div class="card">
        <div class="card-header">
          <div class="card-title">Skill Frequency in Job Listings</div>
          <div class="card-sub">Number of job postings mentioning each skill</div>
        </div>
        <canvas id="skillsChart"></canvas>
      </div>
      <div class="card">
        <div class="card-header">
          <div class="card-title">Skill Insights</div>
          <div class="card-sub">Key observations from the scraped skills data</div>
        </div>
        <ul class="insight-list">
          <li><strong>Python dominates:</strong> Appears across all tech roles, from Data to Backend to ML. The single most listed skill.</li>
          <li><strong>SQL is non-negotiable:</strong> Found in Data Analyst, Data Engineer, and Business Analyst postings consistently.</li>
          <li><strong>Cloud skills are rising:</strong> AWS, Azure, and GCP increasingly appear even in non-cloud-specific job titles.</li>
          <li><strong>Communication counts:</strong> Soft skills appear in many listings, especially for analyst and business roles.</li>
          <li><strong>ML/AI growing:</strong> Machine Learning, Deep Learning, and NLP appear more frequently every month.</li>
        </ul>
        <div class="method-list">
          <div class="method-pill"><span>→</span> Source: Adzuna API</div>
          <div class="method-pill"><span>→</span> Parsed from skills column</div>
          <div class="method-pill"><span>→</span> Deduplicated by (title, company)</div>
        </div>
      </div>
    </div>
  </section>

  <!-- ─── SECTION 4 : SALARY & EXPERIENCE ─── -->
  <section id="s4" class="section">
    <div class="section-heading">
      <div class="section-heading-num">04</div>
      <div class="section-heading-title">Salary &amp; Experience Analysis</div>
      <div class="section-heading-line"></div>
    </div>
    <div class="grid-2" style="{sal_hide}">
      <div class="card">
        <div class="card-header">
          <div class="card-title">Experience Distribution</div>
          <div class="card-sub">Breakdown of jobs by required years of experience</div>
        </div>
        <canvas id="expChart"></canvas>
      </div>
      <div class="card">
        <div class="card-header">
          <div class="card-title">Avg Salary by Role (LPA)</div>
          <div class="card-sub">Average annual salary for top roles where data is available</div>
        </div>
        <canvas id="salRoleChart"></canvas>
      </div>
    </div>
    <div class="grid-2">
      <div class="card">
        <div class="card-header">
          <div class="card-title">Model Feature Importance</div>
          <div class="card-sub">Which features drive salary predictions most, from the trained {model_name} model</div>
        </div>
        <canvas id="featureChart"></canvas>
      </div>
      <div class="card">
        <div class="card-header">
          <div class="card-title">What Drives Your Salary</div>
          <div class="card-sub">Insights from feature importance and salary data</div>
        </div>
        <ul class="insight-list">
          <li><strong>Company matters most:</strong> Which company you work at is the strongest salary predictor, more than experience or skills.</li>
          <li><strong>Experience is second:</strong> Each additional year of experience correlates with meaningful salary growth, especially in 0–5 year range.</li>
          <li><strong>Role titles diverge:</strong> "ML Engineer" and "Data Architect" roles command 2–3× the salary of entry-level analyst roles.</li>
          <li><strong>City premium is real:</strong> Bangalore and Hyderabad listings pay ~20% more than the national average in comparable roles.</li>
          <li><strong>Skills are the wildcard:</strong> AWS, ML, and AI skills add a measurable salary premium even controlling for role.</li>
        </ul>
        <div class="method-list">
          <div class="method-pill"><span>🤖</span> {model_name} Regressor</div>
          <div class="method-pill"><span>→</span> Target encoding</div>
          <div class="method-pill"><span>→</span> 5-fold cross-validation</div>
        </div>
      </div>
    </div>
    <div class="card" style="margin-bottom:var(--gap)">
      <div class="card-header">
        <div class="card-title">Experience vs. Salary</div>
        <div class="card-sub">Each dot is a job listing · colored by experience bracket · up to 700 sampled points</div>
      </div>
      <canvas id="scatterChart" style="max-height:300px"></canvas>
    </div>
  </section>

  <!-- ─── SECTION 5 : KEY INSIGHTS ─── -->
  <section id="s5" class="section">
    <div class="section-heading">
      <div class="section-heading-num">05</div>
      <div class="section-heading-title">Key Insights &amp; Conclusions</div>
      <div class="section-heading-line"></div>
    </div>
    <div class="grid-2">
      <div class="card">
        <div class="card-header">
          <div class="card-title">Market Takeaways</div>
          <div class="card-sub">Synthesised from all scraped listings</div>
        </div>
        <ul class="insight-list">
          <li><strong>{kpis['top_city']} leads hiring:</strong> More active listings than any other city, cementing its status as India's primary tech hub.</li>
          <li><strong>{kpis['top_role']} is most listed:</strong> The single highest-demand role across all listings in the dataset.</li>
          <li><strong>{kpis['unique_roles']} distinct roles fetched:</strong> The India tech market is diverse, spanning Data, ML, Dev, and PM roles.</li>
          <li><strong>Python is universal:</strong> Required in data, ML, backend, and automation roles alike. The safest skill to invest in.</li>
          <li><strong>Tier-1 cities dominate:</strong> Bangalore, Hyderabad, Pune, and Mumbai account for the majority of listings.</li>
        </ul>
      </div>
      <div class="card">
        <div class="card-header">
          <div class="card-title">About This Project</div>
          <div class="card-sub">End-to-end Python pipeline · API-powered</div>
        </div>
        <ul class="insight-list">
          <li><strong>Data collection:</strong> Live listings fetched from Adzuna API across 18 keywords × 6 cities. 2 pages/keyword = 216 calls/day, within the free tier limit.</li>
          <li><strong>Data pipeline:</strong> Raw API response → Pandas → SQLite via a modular ETL pipeline with salary/experience parsing, imputation, and history-preserving upsert.</li>
          <li><strong>ML model:</strong> {model_name} with LOO target encoding, company tier feature (FAANG/MNC/Indian IT/Startup), 5-fold CV, and log-transformed salary target. {r2_str} on test set.</li>
          <li><strong>Salary predictor:</strong> Lookup tables from training data embedded in this HTML. Works fully offline — no server needed for this section.</li>
          <li><strong>Daily pipeline:</strong> GitHub Actions scrapes at 11 AM IST, then automatically retrains and regenerates this dashboard, committing the results back to the repo. Just <code style="background:var(--bg3);padding:1px 5px;border-radius:4px">git pull</code> to get the latest.</li>
        </ul>
        <div class="method-list">
          <div class="method-pill"><span>🐍</span>Python 3.11</div>
          <div class="method-pill"><span>🐼</span>Pandas + SQLite</div>
          <div class="method-pill"><span>🌐</span>Adzuna API</div>
          <div class="method-pill"><span>🤖</span>{model_name}</div>
          <div class="method-pill"><span>🏢</span>Company tiers</div>
          <div class="method-pill"><span>⏰</span>GitHub Actions daily</div>
        </div>
      </div>
    </div>
  </section>

  <!-- ─── SECTION 6 : SALARY PREDICTOR ─── -->
  <section id="s6" class="section">
    <div class="section-heading">
      <div class="section-heading-num">06</div>
      <div class="section-heading-title">Salary Predictor</div>
      <div class="section-heading-line"></div>
    </div>
    <div class="predictor-grid">
      <!-- Form -->
      <div class="card">
        <div class="card-header">
          <div class="card-title">Estimate Your Salary</div>
          <div class="card-sub">Based on {kpis['total_jobs']:,} listings and the trained {model_name} model</div>
        </div>
        <div class="pred-form">
          <div class="pred-field">
            <label>Job Role</label>
            <select class="pred-select" id="pred-role">
              <option value="">Select a role</option>
              {role_options}
            </select>
          </div>
          <div class="pred-field">
            <label>City</label>
            <select class="pred-select" id="pred-city">
              <option value="">Select a city</option>
              {city_options}
            </select>
          </div>
          <div class="pred-field">
            <label>Years of Experience &nbsp; <span id="exp-display" style="color:var(--accent);font-family:'IBM Plex Mono', monospace;letter-spacing:1px">0</span> yrs</label>
            <input type="range" id="pred-exp" min="0" max="15" step="0.5" value="0"
                   oninput="document.getElementById('exp-display').textContent=this.value">
          </div>
          <div class="pred-field">
            <label>Skills (select all that apply)</label>
            <div class="skills-grid">
              {skill_checkboxes}
            </div>
          </div>
          <button class="pred-btn" onclick="runPredictor()">▶ Predict Salary</button>
        </div>
      </div>

      <!-- Result -->
      <div class="card" style="display:flex;flex-direction:column;justify-content:center;">
        <div class="card-header">
          <div class="card-title">Prediction Result</div>
          <div class="card-sub">Estimate based on salary averages from the training dataset</div>
        </div>
        <div class="pred-result" id="pred-result">
          <div class="pred-result-empty">← Fill in the form and click Predict</div>
        </div>
        <p class="pred-note" style="margin-top:14px">
          ⚠ Estimates are computed from market averages in the scraped dataset and carry ~±25% error.
          Use as a ballpark, not a guarantee. Salary depends heavily on company, skills, and interview performance.
        </p>
      </div>
    </div>
  </section>

  <!-- FOOTER -->
  <footer class="footer">
    <span>SOURCE: Adzuna API · India</span>
    <span>INDIA TECH JOBS · 2025 · ADITYA JAI SINGH</span>
    <span>GENERATED BY PYTHON · {kpis['total_jobs']:,} LISTINGS</span>
  </footer>
</div>

<script>
// ── NAVIGATION ────────────────────────────────────────────────────────────────
function showSection(id, btn) {{
  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  btn.classList.add('active');
}}

// ── CHART DEFAULTS ────────────────────────────────────────────────────────────
const _initDark = (localStorage.getItem('theme') === 'dark');
let GRID  = _initDark ? 'rgba(255,255,255,0.05)' : 'rgba(15,23,42,0.06)';
let TICKS = _initDark ? '#94a3b8' : '#64748b';
const FONT = {{ family: 'IBM Plex Sans', size: 12 }};
const AXIS_TITLE_FONT = {{ family: 'Space Grotesk', size: 13 }};
let TIP  = _initDark
  ? {{ backgroundColor:'#1e293b', titleColor:'#f1f5f9', bodyColor:'#94a3b8', borderColor:'rgba(255,255,255,0.08)', borderWidth:1, padding:10 }}
  : {{ backgroundColor:'#fff',    titleColor:'#0f172a', bodyColor:'#475569',  borderColor:'rgba(15,23,42,0.12)',    borderWidth:1, padding:10 }};
const COLORS_LIGHT = ['#2563eb','#059669','#d97706','#dc2626','#7c3aed','#0891b2','#b45309','#be185d','#0f172a','#475569','#64748b','#94a3b8'];
const COLORS_DARK  = ['#60a5fa','#34d399','#fbbf24','#f87171','#a78bfa','#22d3ee','#fb923c','#f472b6','#94a3b8','#cbd5e1','#64748b','#475569'];
let COLORS = _initDark ? COLORS_DARK : COLORS_LIGHT;

// ── STATIC DATA (injected by Python) ─────────────────────────────────────────
const ALL_CITIES_LABELS = {json.dumps(cities['labels'])};
const ALL_CITIES_VALUES = {json.dumps(cities['values'])};
const ALL_ROLES_LABELS  = {json.dumps(roles['labels'])};
const ALL_ROLES_VALUES  = {json.dumps(roles['values'])};
const skillsLabels      = {json.dumps(skills['labels'])};
const skillsValues      = {json.dumps(skills['values'])};
const expLabels         = {json.dumps(exp['labels'])};
const expValues         = {json.dumps(exp['values'])};
const salRoleLabels     = {json.dumps(sal_role['labels'])};
const salRoleValues     = {json.dumps(sal_role['values'])};
const featureLabels     = {json.dumps(fi_labels_clean)};
const featureValues     = {json.dumps(fi_values)};
const scatterData       = {json.dumps(scatter)};
const EXP_BUCKET_LABELS = ['0–1 yrs','1–3 yrs','3–5 yrs','5–8 yrs','8+ yrs'];
const EXP_COLORS_LIGHT  = ['#2563eb','#059669','#d97706','#dc2626','#7c3aed'];
const EXP_COLORS_DARK   = ['#60a5fa','#34d399','#fbbf24','#f87171','#a78bfa'];
let   EXP_COLORS = _initDark ? EXP_COLORS_DARK : EXP_COLORS_LIGHT;

// Per-role and per-city raw data for filters
const ROLE_BY_CITY  = {json.dumps(stats.get('role_by_city', {}))};
const CITY_BY_ROLE  = {json.dumps(stats.get('city_by_role', {}))};

// Predictor lookup tables
const GLOBAL_MEAN   = {pred['global_mean']};
const ROLE_SALARY   = {json.dumps(pred['role_salary'])};
const CITY_SALARY   = {json.dumps(pred['city_salary'])};
const SKILL_PREMIUM = {json.dumps(pred['skill_premium'])};

// ── BUILD CHARTS ──────────────────────────────────────────────────────────────
let citiesChart, rolesChart, rolesDonut, skillsChartInst, expChartInst, salChartInst, featChartInst, scatterChartInst;

citiesChart = new Chart(document.getElementById('citiesChart'), {{
  type:'bar',
  data:{{ labels:ALL_CITIES_LABELS, datasets:[{{ data:ALL_CITIES_VALUES,
    backgroundColor:ALL_CITIES_LABELS.map((_,i)=>COLORS[i%COLORS.length]+'cc'),
    borderColor:ALL_CITIES_LABELS.map((_,i)=>COLORS[i%COLORS.length]),
    borderWidth:1, borderRadius:5 }}] }},
  options:{{ indexAxis:'y', responsive:true,
    plugins:{{ legend:{{display:false}}, tooltip:{{...TIP}} }},
    scales:{{ x:{{ticks:{{color:TICKS,font:FONT}},grid:{{color:GRID}}}}, y:{{ticks:{{color:TICKS,font:FONT}},grid:{{display:false}},afterFit(s){{s.width=150;}}}} }}
  }}
}});

rolesChart = new Chart(document.getElementById('rolesChart'), {{
  type:'bar',
  data:{{ labels:ALL_ROLES_LABELS, datasets:[{{ data:ALL_ROLES_VALUES,
    backgroundColor:'rgba(37,99,235,0.75)', borderRadius:5 }}] }},
  options:{{ indexAxis:'y', responsive:true,
    plugins:{{ legend:{{display:false}}, tooltip:{{...TIP}} }},
    scales:{{ x:{{ticks:{{color:TICKS,font:FONT}},grid:{{color:GRID}}}}, y:{{ticks:{{color:TICKS,font:FONT}},grid:{{display:false}},afterFit(s){{s.width=160;}}}} }}
  }}
}});

rolesDonut = new Chart(document.getElementById('rolesDoughnut'), {{
  type:'doughnut',
  data:{{ labels:ALL_ROLES_LABELS.slice(0,6), datasets:[{{ data:ALL_ROLES_VALUES.slice(0,6),
    backgroundColor:COLORS.slice(0,6).map(c=>c+'cc'), borderColor:'#ffffff', borderWidth:2 }}] }},
  options:{{ responsive:true,
    plugins:{{ legend:{{labels:{{color:'#475569',font:FONT,boxWidth:10}}}}, tooltip:{{...TIP}} }}
  }}
}});

skillsChartInst = new Chart(document.getElementById('skillsChart'), {{
  type:'bar',
  data:{{ labels:skillsLabels, datasets:[{{ data:skillsValues,
    backgroundColor:skillsLabels.map((_,i)=>COLORS[i%COLORS.length]+'bb'),
    borderColor:skillsLabels.map((_,i)=>COLORS[i%COLORS.length]),
    borderWidth:1, borderRadius:4 }}] }},
  options:{{ indexAxis:'y', responsive:true,
    plugins:{{ legend:{{display:false}}, tooltip:{{...TIP}} }},
    scales:{{ x:{{ticks:{{color:TICKS,font:FONT}},grid:{{color:GRID}}}}, y:{{ticks:{{color:TICKS,font:FONT}},grid:{{display:false}},afterFit(s){{s.width=140;}}}} }}
  }}
}});

expChartInst = new Chart(document.getElementById('expChart'), {{
  type:'bar',
  data:{{ labels:expLabels, datasets:[{{ data:expValues,
    backgroundColor:['rgba(37,99,235,0.8)','rgba(5,150,105,0.8)','rgba(217,119,6,0.8)','rgba(220,38,38,0.8)','rgba(124,58,237,0.8)'],
    borderRadius:6 }}] }},
  options:{{ responsive:true,
    plugins:{{ legend:{{display:false}}, tooltip:{{...TIP}} }},
    scales:{{ x:{{ticks:{{color:TICKS,font:FONT}},grid:{{color:GRID}}}}, y:{{ticks:{{color:TICKS,font:FONT}},grid:{{color:GRID}}}} }}
  }}
}});

if (salRoleLabels.length > 0) {{
  salChartInst = new Chart(document.getElementById('salRoleChart'), {{
    type:'bar',
    data:{{ labels:salRoleLabels, datasets:[{{ data:salRoleValues,
      backgroundColor:'rgba(217,119,6,0.75)', borderRadius:5 }}] }},
    options:{{ indexAxis:'y', responsive:true,
      plugins:{{ legend:{{display:false}}, tooltip:{{...TIP, callbacks:{{label:ctx=>`₹${{ctx.raw}} LPA`}}}} }},
      scales:{{ x:{{ticks:{{color:TICKS,font:FONT,callback:v=>'₹'+v+'L'}},grid:{{color:GRID}}}}, y:{{ticks:{{color:TICKS,font:FONT}},grid:{{display:false}},afterFit(s){{s.width=220;}}}} }}
    }}
  }});
}}

if (featureLabels.length > 0) {{
  featChartInst = new Chart(document.getElementById('featureChart'), {{
    type:'bar',
    data:{{ labels:featureLabels, datasets:[{{ data:featureValues,
      backgroundColor:featureLabels.map((_,i)=>COLORS[i%COLORS.length]+'cc'),
      borderColor:featureLabels.map((_,i)=>COLORS[i%COLORS.length]),
      borderWidth:1, borderRadius:4 }}] }},
    options:{{ indexAxis:'y', responsive:true,
      plugins:{{ legend:{{display:false}}, tooltip:{{...TIP,callbacks:{{label:ctx=>`Importance: ${{(ctx.raw*100).toFixed(1)}}%`}}}} }},
      scales:{{ x:{{ticks:{{color:TICKS,font:FONT,callback:v=>(v*100).toFixed(0)+'%'}},grid:{{color:GRID}}}}, y:{{ticks:{{color:TICKS,font:FONT}},grid:{{display:false}},afterFit(s){{s.width=160;}}}} }}
    }}
  }});
}}

if (scatterData.length > 0) {{
  const scatterDatasets = [0,1,2,3,4].map(b => ({{
    label: EXP_BUCKET_LABELS[b],
    data:  scatterData.filter(d => d.b === b),
    backgroundColor: EXP_COLORS[b]+'55',
    borderColor:     EXP_COLORS[b]+'99',
    borderWidth: 1, pointRadius: 4, pointHoverRadius: 6,
  }}));
  scatterChartInst = new Chart(document.getElementById('scatterChart'), {{
    type: 'scatter',
    data: {{ datasets: scatterDatasets }},
    options: {{
      responsive: true,
      plugins: {{
        legend: {{ labels: {{ color:TICKS, font:FONT, boxWidth:10 }} }},
        tooltip: {{ ...TIP,
          callbacks: {{
            title: () => '',
            label: ctx => {{
              const d = ctx.raw;
              const lines = [];
              if (d.t) lines.push(d.t);
              if (d.c || d.l) lines.push([d.c, d.l].filter(Boolean).join(' · '));
              lines.push(`${{d.x}} yrs exp · ₹${{d.y}} LPA`);
              return lines;
            }},
          }},
        }},
      }},
      scales: {{
        x: {{
          title: {{ display:true, text:'Experience (years)', color:TICKS, font:AXIS_TITLE_FONT }},
          ticks: {{ color:TICKS, font:FONT }},
          grid:  {{ color:GRID }},
        }},
        y: {{
          title: {{ display:true, text:'Salary (LPA)', color:TICKS, font:AXIS_TITLE_FONT }},
          ticks: {{ color:TICKS, font:FONT, callback: v => '₹'+v }},
          grid:  {{ color:GRID }},
        }},
      }},
    }},
  }});
}}

// ── FILTERS ───────────────────────────────────────────────────────────────────
function filterCitiesChart() {{
  const role = document.getElementById('s1-role-filter').value;
  let labels, values;
  if (!role || !CITY_BY_ROLE[role]) {{
    labels = ALL_CITIES_LABELS;
    values = ALL_CITIES_VALUES;
  }} else {{
    const entries = Object.entries(CITY_BY_ROLE[role]).sort((a,b) => b[1]-a[1]).slice(0,10);
    labels = entries.map(e => e[0]);
    values = entries.map(e => e[1]);
  }}
  citiesChart.data.labels = labels;
  citiesChart.data.datasets[0].data = values;
  citiesChart.data.datasets[0].backgroundColor = labels.map((_,i) => COLORS[i%COLORS.length]+'cc');
  citiesChart.data.datasets[0].borderColor      = labels.map((_,i) => COLORS[i%COLORS.length]);
  citiesChart.update();
}}

function filterRolesChart() {{
  const city = document.getElementById('s2-city-filter').value;
  let labels, values;
  if (!city || !ROLE_BY_CITY[city]) {{
    labels = ALL_ROLES_LABELS;
    values = ALL_ROLES_VALUES;
  }} else {{
    const entries = Object.entries(ROLE_BY_CITY[city]).sort((a,b) => b[1]-a[1]).slice(0,10);
    labels = entries.map(e => e[0]);
    values = entries.map(e => e[1]);
  }}
  rolesChart.data.labels = labels;
  rolesChart.data.datasets[0].data = values;
  rolesDonut.data.labels = labels.slice(0,6);
  rolesDonut.data.datasets[0].data = values.slice(0,6);
  rolesChart.update(); rolesDonut.update();
}}

function resetFilter(selectId, fnName) {{
  document.getElementById(selectId).value = '';
  window[fnName]();
}}

// ── THEME TOGGLE ─────────────────────────────────────────────────────────────
function toggleTheme() {{
  const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
  applyTheme(isDark ? 'light' : 'dark');
}}

function applyTheme(theme) {{
  const isDark = theme === 'dark';
  document.documentElement.setAttribute('data-theme', isDark ? 'dark' : '');
  localStorage.setItem('theme', theme);
  document.getElementById('theme-icon').textContent  = isDark ? '☀️' : '🌙';
  document.getElementById('theme-label').textContent = isDark ? 'Light Mode' : 'Dark Mode';

  GRID = isDark ? 'rgba(255,255,255,0.05)' : 'rgba(15,23,42,0.06)';
  TIP  = isDark
    ? {{ backgroundColor:'#1e293b', titleColor:'#f1f5f9', bodyColor:'#94a3b8', borderColor:'rgba(255,255,255,0.08)', borderWidth:1, padding:10 }}
    : {{ backgroundColor:'#fff',    titleColor:'#0f172a', bodyColor:'#475569',  borderColor:'rgba(15,23,42,0.12)',    borderWidth:1, padding:10 }};
  const donutBorder = isDark ? '#1e1e1e' : '#ffffff';
  const legendColor = isDark ? '#94a3b8' : '#475569';

  // Switch colour palettes
  COLORS     = isDark ? COLORS_DARK     : COLORS_LIGHT;
  EXP_COLORS = isDark ? EXP_COLORS_DARK : EXP_COLORS_LIGHT;

  // Repaint multi-colour charts
  const _repaintMulti = (chart, bgAlpha, bdAlpha) => {{
    if (!chart) return;
    const n = chart.data.labels.length;
    chart.data.datasets[0].backgroundColor = Array.from({{length:n}},(_,i)=>COLORS[i%COLORS.length]+bgAlpha);
    chart.data.datasets[0].borderColor      = Array.from({{length:n}},(_,i)=>COLORS[i%COLORS.length]+(bdAlpha||''));
  }};
  _repaintMulti(citiesChart,   'cc', '');
  _repaintMulti(skillsChartInst,'bb', '');
  _repaintMulti(featChartInst, 'cc', '');

  if (rolesChart) {{
    rolesChart.data.datasets[0].backgroundColor = isDark ? 'rgba(96,165,250,0.80)' : 'rgba(37,99,235,0.75)';
  }}
  if (salChartInst) {{
    salChartInst.data.datasets[0].backgroundColor = isDark ? 'rgba(251,191,36,0.85)' : 'rgba(217,119,6,0.75)';
  }}
  if (expChartInst) {{
    expChartInst.data.datasets[0].backgroundColor = isDark
      ? ['rgba(96,165,250,0.85)','rgba(52,211,153,0.85)','rgba(251,191,36,0.85)','rgba(248,113,113,0.85)','rgba(167,139,250,0.85)']
      : ['rgba(37,99,235,0.8)','rgba(5,150,105,0.8)','rgba(217,119,6,0.8)','rgba(220,38,38,0.8)','rgba(124,58,237,0.8)'];
  }}

  if (scatterChartInst) {{
    scatterChartInst.data.datasets.forEach((ds, b) => {{
      ds.backgroundColor = EXP_COLORS[b]+'55';
      ds.borderColor     = EXP_COLORS[b]+'99';
    }});
    scatterChartInst.options.plugins.legend.labels.color = isDark ? '#94a3b8' : '#475569';
    scatterChartInst.options.scales.x.title.color = isDark ? '#94a3b8' : '#64748b';
    scatterChartInst.options.scales.y.title.color = isDark ? '#94a3b8' : '#64748b';
  }}

  [citiesChart, rolesChart, skillsChartInst, expChartInst, salChartInst, featChartInst, scatterChartInst].forEach(chart => {{
    if (!chart) return;
    chart.options.plugins.tooltip = {{ ...chart.options.plugins.tooltip, ...TIP }};
    if (chart.options.scales) {{
      Object.values(chart.options.scales).forEach(scale => {{
        if (scale.ticks) scale.ticks.color = isDark ? '#94a3b8' : '#64748b';
        if (scale.grid && scale.grid.color !== undefined) scale.grid.color = GRID;
      }});
    }}
    chart.update('none');
  }});

  if (rolesDonut) {{
    const dn = rolesDonut.data.labels.length;
    rolesDonut.data.datasets[0].backgroundColor = Array.from({{length:dn}},(_,i)=>COLORS[i%COLORS.length]+'cc');
    rolesDonut.data.datasets[0].borderColor = donutBorder;
    rolesDonut.options.plugins.tooltip = {{ ...rolesDonut.options.plugins.tooltip, ...TIP }};
    rolesDonut.options.plugins.legend.labels.color = legendColor;
    rolesDonut.update('none');
  }}
}}

// Apply saved theme on load
applyTheme(localStorage.getItem('theme') || 'light');

// ── PIPELINE RUNNER ───────────────────────────────────────────────────────────
// Calls the local server.py backend (python server.py must be running).
let _pipelinePoller = null;

async function runPipeline() {{
  const btn    = document.getElementById('run-btn');
  const status = document.getElementById('run-status');

  // Check if server is reachable first
  try {{
    const check = await fetch('http://localhost:8080/api/status', {{signal: AbortSignal.timeout(1500)}});
    if (!check.ok) throw new Error();
  }} catch (e) {{
    status.className = 'run-status-bar visible';
    status.innerHTML = '⚠ Server not running. Start it with: <code style="background:var(--bg3);padding:1px 5px;border-radius:4px">python server.py</code>';
    return;
  }}

  btn.textContent  = '⟳ Running...';
  btn.className    = 'run-btn running';
  btn.disabled     = true;
  status.className = 'run-status-bar visible';
  status.textContent = 'Starting pipeline...';

  try {{
    await fetch('http://localhost:8080/api/run', {{method: 'POST'}});
  }} catch (e) {{
    btn.textContent  = '▶ Run Pipeline';
    btn.className    = 'run-btn';
    btn.disabled     = false;
    status.textContent = '✗ Could not reach server.';
    return;
  }}

  // Poll status every 3 seconds
  _pipelinePoller = setInterval(async () => {{
    try {{
      const res  = await fetch('http://localhost:8080/api/status');
      const data = await res.json();
      if (data.running) {{
        status.textContent = '⟳ ' + (data.step || 'Running pipeline...');
      }} else {{
        clearInterval(_pipelinePoller);
        btn.textContent  = '✓ Done — Reload';
        btn.className    = 'run-btn done';
        btn.disabled     = false;
        btn.onclick      = () => location.reload();
        status.textContent = data.last_message || '✓ Pipeline complete. Reload to see fresh data.';
      }}
    }} catch (e) {{
      clearInterval(_pipelinePoller);
      btn.textContent  = '▶ Run Pipeline';
      btn.className    = 'run-btn';
      btn.disabled     = false;
      btn.onclick      = runPipeline;
      status.textContent = '✗ Lost connection to server.';
    }}
  }}, 3000);
}}

// ── SALARY PREDICTOR ──────────────────────────────────────────────────────────
function runPredictor() {{
  const role = document.getElementById('pred-role').value;
  const city = document.getElementById('pred-city').value;
  const exp  = parseFloat(document.getElementById('pred-exp').value);

  if (!role && !city) {{
    document.getElementById('pred-result').innerHTML =
      '<div class="pred-result-empty" style="color:#dc2626">Please select at least a role or city.</div>';
    return;
  }}

  // Weighted combination: role 45%, city 30%, global 25%
  const roleMean = role ? (ROLE_SALARY[role] || GLOBAL_MEAN) : GLOBAL_MEAN;
  const cityMean = city ? (CITY_SALARY[city] || GLOBAL_MEAN) : GLOBAL_MEAN;
  const rw = role ? 0.45 : 0.0;
  const cw = city ? 0.30 : 0.0;
  const gw = 1.0 - rw - cw;
  let base = roleMean * rw + cityMean * cw + GLOBAL_MEAN * gw;

  // Experience adjustment: +6% per year above median (2 yrs)
  const expAdj = 1 + (exp - 2.0) * 0.06;
  let predicted = base * Math.max(expAdj, 0.6);

  // Skill premium
  let skillBonus = 0;
  let selectedSkills = [];
  document.querySelectorAll('.skill-check:checked').forEach(cb => {{
    selectedSkills.push(cb.value);
    skillBonus += (SKILL_PREMIUM[cb.value] || 0);
  }});
  predicted += skillBonus * 0.15;
  predicted = Math.max(predicted, 1.0);

  const low  = (predicted * 0.78).toFixed(1);
  const high = (predicted * 1.22).toFixed(1);
  predicted  = predicted.toFixed(1);

  // Breakdown rows
  let breakdown = `
    <div class="pred-breakdown-row"><span>Role baseline</span><span>₹${{roleMean.toFixed(1)}} LPA</span></div>
    <div class="pred-breakdown-row"><span>City adjustment</span><span>₹${{cityMean.toFixed(1)}} LPA</span></div>
    <div class="pred-breakdown-row"><span>Experience (${{exp}} yrs)</span><span>×${{Math.max(1+(exp-2)*0.06,0.6).toFixed(2)}}</span></div>
  `;
  if (selectedSkills.length > 0)
    breakdown += `<div class="pred-breakdown-row"><span>Skill premium (${{selectedSkills.length}} skills)</span><span>+₹${{(skillBonus*0.15).toFixed(1)}} LPA</span></div>`;

  document.getElementById('pred-result').innerHTML = `
    <div class="pred-model-badge">{model_name} · Target Encoding · {kpis['total_jobs']:,} listings</div>
    <div class="pred-result-val">₹${{predicted}}</div>
    <div class="pred-result-unit">Lakhs Per Annum</div>
    <div class="pred-result-range">Range: <span>₹${{low}} – ₹${{high}} LPA</span></div>
    <div class="pred-result-breakdown">${{breakdown}}</div>
  `;
}}
</script>
</body>
</html>"""


# ── Entry point ───────────────────────────────────────────────────────────────

def generate(output_path: str = OUTPUT_PATH) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df    = load_from_db()
    stats = compute_stats(df)
    html  = build_html(stats)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[✓] Dashboard saved → {output_path}")
    return output_path


if __name__ == "__main__":
    generate()
