"""
app.py — FIFA World Cup 2026 Match Predictor
Deploy: Hugging Face Spaces (SDK: Gradio)

Model: XGBoost trained on competitive matches only (excludes friendlies)
Accuracy: 0.5933 | F1-macro: 0.4439 | 27 features
Validated with walk-forward across 7 temporal splits (0.579-0.593)

Required files in this Space:
  - wc2026_model.pkl          (required)
  - wc2026_features.pkl        (required)
  - latest_team_stats.pkl      (optional — real recent-form snapshot)
  - pair_history.pkl           (optional — real head-to-head history)
  - wc2026.db                  (optional — created automatically on first run)

If latest_team_stats.pkl / pair_history.pkl are missing, the app falls back
to neutral defaults (predictions then rely mostly on Elo ratings).
"""

import gradio as gr
import numpy as np
import sqlite3
import pickle
import os

# ── Load model (required) ──────────────────────────────────────────────────────
with open("wc2026_model.pkl",    "rb") as f: model    = pickle.load(f)
with open("wc2026_features.pkl", "rb") as f: FEATURES = pickle.load(f)

MODEL_NAME = "XGBoost (Competitive Matches)"

# ── Load optional precomputed stats (recent form + head-to-head) ──────────────
LATEST_TEAM_STATS = {}
PAIR_HISTORY = {}

if os.path.exists("latest_team_stats.pkl"):
    with open("latest_team_stats.pkl", "rb") as f:
        LATEST_TEAM_STATS = pickle.load(f)
    print(f"✅ latest_team_stats.pkl cargado — {len(LATEST_TEAM_STATS)} equipos")
else:
    print("⚠️  latest_team_stats.pkl no encontrado — usando defaults neutrales")

if os.path.exists("pair_history.pkl"):
    with open("pair_history.pkl", "rb") as f:
        PAIR_HISTORY = pickle.load(f)
    print(f"✅ pair_history.pkl cargado — {len(PAIR_HISTORY)} pares de equipos")
else:
    print("⚠️  pair_history.pkl no encontrado — head-to-head usará default neutral")

# ── Elo ratings (pre-tournament snapshot June 2026) ───────────────────────────
ELO_2026 = {
    "Spain":2155,"Argentina":2113,"France":2062,"England":2020,"Brazil":1988,
    "Portugal":1984,"Colombia":1977,"Netherlands":1944,"Germany":1925,"Belgium":1900,
    "Morocco":1880,"Italy":1870,"Uruguay":1860,"Croatia":1855,"Japan":1850,
    "Mexico":1845,"United States":1840,"Ecuador":1820,"Senegal":1815,"Australia":1810,
    "Switzerland":1805,"Denmark":1800,"Serbia":1795,"Poland":1790,"South Korea":1785,
    "Tunisia":1775,"Canada":1770,"Costa Rica":1755,"Cameroon":1750,"Ghana":1745,
    "Iran":1740,"Saudi Arabia":1715,"Qatar":1700,"Panama":1695,"Venezuela":1690,
    "Paraguay":1685,"Peru":1680,"Chile":1675,"Algeria":1670,"Egypt":1665,
    "Nigeria":1660,"South Africa":1650,"Kenya":1600,"New Zealand":1580,
    "Indonesia":1560,"Honduras":1555,"Jamaica":1550,"Haiti":1530,"Angola":1525,"Ukraine":1810,
}

WC2026_TEAMS = sorted(ELO_2026.keys())

# ── Default stats for teams without recent history ────────────────────────────
DEFAULT_STATS = {
    'pts_avg_5':1.2, 'pts_avg_10':1.2,
    'gd_avg_5':0.0,  'gd_avg_10':0.0,
    'gf_avg_5':1.2,
    'win_streak':0, 'days_since_last':30
}

WC2026_MATCHES = [
    # ── Group A ───────────────────────────────────────────────────────────────
    (1,  "Mexico",       "South Africa", "2026-06-11", "Group A"),
    (2,  "South Korea",  "Czechia",      "2026-06-11", "Group A"),
    (3,  "Czechia",      "South Africa", "2026-06-18", "Group A"),
    (4,  "Mexico",       "South Korea",  "2026-06-18", "Group A"),
    (5,  "Czechia",      "Mexico",       "2026-06-24", "Group A"),
    (6,  "South Africa", "South Korea",  "2026-06-24", "Group A"),
    # ── Group B ───────────────────────────────────────────────────────────────
    (7,  "Canada",       "Bosnia and Herzegovina", "2026-06-12", "Group B"),
    (8,  "Qatar",        "Switzerland",  "2026-06-13", "Group B"),
    (9,  "Switzerland",  "Bosnia and Herzegovina", "2026-06-18", "Group B"),
    (10, "Canada",       "Qatar",        "2026-06-18", "Group B"),
    (11, "Switzerland",  "Canada",       "2026-06-24", "Group B"),
    (12, "Bosnia and Herzegovina", "Qatar", "2026-06-24", "Group B"),
    # ── Group C ───────────────────────────────────────────────────────────────
    (13, "Brazil",       "Morocco",      "2026-06-13", "Group C"),
    (14, "Haiti",        "Scotland",     "2026-06-13", "Group C"),
    (15, "Scotland",     "Morocco",      "2026-06-19", "Group C"),
    (16, "Brazil",       "Haiti",        "2026-06-19", "Group C"),
    (17, "Scotland",     "Brazil",       "2026-06-24", "Group C"),
    (18, "Morocco",      "Haiti",        "2026-06-24", "Group C"),
    # ── Group D ───────────────────────────────────────────────────────────────
    (19, "USA",          "Paraguay",     "2026-06-12", "Group D"),
    (20, "Australia",    "Turkiye",      "2026-06-13", "Group D"),
    (21, "USA",          "Australia",    "2026-06-19", "Group D"),
    (22, "Turkiye",      "Paraguay",     "2026-06-19", "Group D"),
    (23, "Turkiye",      "USA",          "2026-06-25", "Group D"),
    (24, "Paraguay",     "Australia",    "2026-06-25", "Group D"),
    # ── Group E ───────────────────────────────────────────────────────────────
    (25, "Germany",      "Curacao",      "2026-06-14", "Group E"),
    (26, "Ivory Coast",  "Ecuador",      "2026-06-14", "Group E"),
    (27, "Germany",      "Ivory Coast",  "2026-06-20", "Group E"),
    (28, "Ecuador",      "Curacao",      "2026-06-20", "Group E"),
    (29, "Ecuador",      "Germany",      "2026-06-25", "Group E"),
    (30, "Curacao",      "Ivory Coast",  "2026-06-25", "Group E"),
    # ── Group F ───────────────────────────────────────────────────────────────
    (31, "Netherlands",  "Japan",        "2026-06-14", "Group F"),
    (32, "Sweden",       "Tunisia",      "2026-06-14", "Group F"),
    (33, "Netherlands",  "Sweden",       "2026-06-20", "Group F"),
    (34, "Tunisia",      "Japan",        "2026-06-20", "Group F"),
    (35, "Japan",        "Sweden",       "2026-06-25", "Group F"),
    (36, "Tunisia",      "Netherlands",  "2026-06-25", "Group F"),
    # ── Group G ───────────────────────────────────────────────────────────────
    (37, "Belgium",      "Egypt",        "2026-06-15", "Group G"),
    (38, "Iran",         "New Zealand",  "2026-06-15", "Group G"),
    (39, "Belgium",      "Iran",         "2026-06-21", "Group G"),
    (40, "New Zealand",  "Egypt",        "2026-06-21", "Group G"),
    (41, "Egypt",        "Iran",         "2026-06-26", "Group G"),
    (42, "New Zealand",  "Belgium",      "2026-06-26", "Group G"),
    # ── Group H ───────────────────────────────────────────────────────────────
    (43, "Spain",        "Cape Verde",   "2026-06-15", "Group H"),
    (44, "Saudi Arabia", "Uruguay",      "2026-06-15", "Group H"),
    (45, "Spain",        "Saudi Arabia", "2026-06-21", "Group H"),
    (46, "Uruguay",      "Cape Verde",   "2026-06-21", "Group H"),
    (47, "Cape Verde",   "Saudi Arabia", "2026-06-26", "Group H"),
    (48, "Uruguay",      "Spain",        "2026-06-26", "Group H"),
    # ── Group I ───────────────────────────────────────────────────────────────
    (49, "France",       "Senegal",      "2026-06-16", "Group I"),
    (50, "Iraq",         "Norway",       "2026-06-16", "Group I"),
    (51, "France",       "Iraq",         "2026-06-22", "Group I"),
    (52, "Norway",       "Senegal",      "2026-06-22", "Group I"),
    (53, "Norway",       "France",       "2026-06-26", "Group I"),
    (54, "Senegal",      "Iraq",         "2026-06-26", "Group I"),
    # ── Group J ───────────────────────────────────────────────────────────────
    (55, "Argentina",    "Algeria",      "2026-06-16", "Group J"),
    (56, "Austria",      "Jordan",       "2026-06-17", "Group J"),
    (57, "Argentina",    "Austria",      "2026-06-22", "Group J"),
    (58, "Jordan",       "Algeria",      "2026-06-22", "Group J"),
    (59, "Algeria",      "Austria",      "2026-06-27", "Group J"),
    (60, "Jordan",       "Argentina",    "2026-06-27", "Group J"),
    # ── Group K ───────────────────────────────────────────────────────────────
    (61, "Portugal",     "DR Congo",     "2026-06-17", "Group K"),
    (62, "Uzbekistan",   "Colombia",     "2026-06-17", "Group K"),
    (63, "Portugal",     "Uzbekistan",   "2026-06-23", "Group K"),
    (64, "Colombia",     "DR Congo",     "2026-06-23", "Group K"),
    (65, "DR Congo",     "Uzbekistan",   "2026-06-27", "Group K"),
    (66, "Colombia",     "Portugal",     "2026-06-27", "Group K"),
    # ── Group L ───────────────────────────────────────────────────────────────
    (67, "England",      "Croatia",      "2026-06-17", "Group L"),
    (68, "Ghana",        "Panama",       "2026-06-17", "Group L"),
    (69, "England",      "Ghana",        "2026-06-23", "Group L"),
    (70, "Panama",       "Croatia",      "2026-06-23", "Group L"),
    (71, "Croatia",      "Ghana",        "2026-06-27", "Group L"),
    (72, "Panama",       "England",      "2026-06-27", "Group L"),
]

# ── Database setup ─────────────────────────────────────────────────────────────
DB_PATH = "wc2026.db"

def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            total_points INTEGER DEFAULT 0
        )""")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS matches (
            id INTEGER PRIMARY KEY,
            home_team TEXT, away_team TEXT,
            match_date TEXT, stage TEXT,
            actual_result TEXT, home_score INTEGER, away_score INTEGER,
            ai_prediction TEXT, ai_confidence REAL
        )""")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER, match_id INTEGER,
            prediction TEXT,
            pred_home_score INTEGER, pred_away_score INTEGER,
            points_earned INTEGER DEFAULT 0,
            UNIQUE(user_id, match_id)
        )""")
    conn.commit()
    for mid, h, a, dt, stage in WC2026_MATCHES:
        conn.execute(
            "INSERT OR IGNORE INTO matches (id,home_team,away_team,match_date,stage) VALUES (?,?,?,?,?)",
            (mid, h, a, dt, stage)
        )
    conn.commit()
    return conn

conn = get_conn()

# ── Head-to-head lookup ─────────────────────────────────────────────────────────
def get_h2h(team_a, team_b):
    """Promedio de puntos históricos de team_a frente a team_b."""
    key = tuple(sorted([team_a, team_b]))
    hist = PAIR_HISTORY.get(key, [])
    if not hist:
        return 1.2, 0
    pts_a = []
    for (d_, pts_val, ref_team) in hist:
        if ref_team == team_a:
            pts_a.append(pts_val)
        else:
            pts_a.append(3 - pts_val if pts_val != 1 else 1)
    return float(np.mean(pts_a)), len(hist)


# ── Prediction engine (27 features) ────────────────────────────────────────────
def predict_match(home_team, away_team):
    """
    Predice el resultado de un partido usando las 27 features del modelo
    final (XGBoost entrenado en partidos competitivos, Accuracy 0.5933).
    Retorna (prediction_label, probabilities_dict, elo_diff)
    """
    hs  = LATEST_TEAM_STATS.get(home_team, DEFAULT_STATS)
    as_ = LATEST_TEAM_STATS.get(away_team, DEFAULT_STATS)

    h_elo = ELO_2026.get(home_team, 1600)
    a_elo = ELO_2026.get(away_team, 1600)
    elo_diff  = h_elo - a_elo
    elo_ratio = h_elo / a_elo

    h2h_pts, h2h_n = get_h2h(home_team, away_team)

    diff_pts_5  = hs['pts_avg_5']  - as_['pts_avg_5']
    diff_pts_10 = hs['pts_avg_10'] - as_['pts_avg_10']
    diff_gd_5   = hs['gd_avg_5']   - as_['gd_avg_5']
    diff_gd_10  = hs['gd_avg_10']  - as_['gd_avg_10']
    diff_streak = hs['win_streak'] - as_['win_streak']
    elo_x_form  = elo_diff * diff_pts_5
    total_goals_exp = hs['gf_avg_5'] + as_['gf_avg_5']

    feat_map = {
        'elo_diff': elo_diff, 'elo_ratio': elo_ratio,
        'home_elo': h_elo, 'away_elo': a_elo,
        'is_neutral': 1, 'is_world_cup': 1,
        'home_pts_avg_5': hs['pts_avg_5'], 'away_pts_avg_5': as_['pts_avg_5'],
        'home_pts_avg_10': hs['pts_avg_10'], 'away_pts_avg_10': as_['pts_avg_10'],
        'home_gd_avg_5': hs['gd_avg_5'], 'away_gd_avg_5': as_['gd_avg_5'],
        'home_gd_avg_10': hs['gd_avg_10'], 'away_gd_avg_10': as_['gd_avg_10'],
        'diff_pts_5': diff_pts_5, 'diff_pts_10': diff_pts_10,
        'diff_gd_5': diff_gd_5, 'diff_gd_10': diff_gd_10,
        'home_win_streak': hs['win_streak'], 'away_win_streak': as_['win_streak'],
        'diff_streak': diff_streak,
        'h2h_home_avg_pts': h2h_pts, 'h2h_matches_count': h2h_n,
        'home_days_since_last': hs['days_since_last'], 'away_days_since_last': as_['days_since_last'],
        'elo_x_form': elo_x_form, 'total_goals_exp': total_goals_exp,
    }

    fv = np.array([[feat_map[f] for f in FEATURES]], dtype=float)

    proba = model.predict_proba(fv)[0]
    pred  = model.predict(fv)[0]

    label_map  = {0:'Home Win', 1:'Draw', 2:'Away Win'}
    proba_dict = {
        f'{home_team} Win': round(float(proba[0])*100, 1),
        'Draw':              round(float(proba[1])*100, 1),
        f'{away_team} Win': round(float(proba[2])*100, 1),
    }
    return label_map[pred], proba_dict, elo_diff


def ai_analysis(home_team, away_team):
    pred, proba, elo_diff = predict_match(home_team, away_team)
    h_elo = ELO_2026.get(home_team, 1600)
    a_elo = ELO_2026.get(away_team, 1600)
    try:
        import anthropic
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=180,
            messages=[{"role":"user","content":(
                f"Analiza el partido Copa del Mundo 2026: {home_team} vs {away_team}. "
                f"Elo {home_team}={h_elo}, Elo {away_team}={a_elo}, diferencia={elo_diff:+d}. "
                f"Predicción: {pred} ({proba}). "
                f"Genera 2 oraciones en español, estilo periodismo deportivo."
            )}]
        )
        return msg.content[0].text
    except Exception:
        stronger = home_team if elo_diff > 0 else away_team
        max_k    = max(proba, key=proba.get)
        return (
            f"{stronger} llega con ventaja Elo de {abs(elo_diff)} puntos "
            f"({h_elo} vs {a_elo}). "
            f"El modelo proyecta {max_k} con {proba[max_k]}% de probabilidad."
        )


# ── App functions ──────────────────────────────────────────────────────────────
def app_predict(home_team, away_team):
    if home_team == away_team:
        return "❌ Selecciona dos equipos diferentes.", "", ""
    pred, proba, elo_diff = predict_match(home_team, away_team)
    emoji   = {"Home Win":"🏠✅","Draw":"🤝","Away Win":"✈️✅"}
    res_str = f"{emoji.get(pred,'')} **Predicción: {pred}**"
    prob_md = "| Resultado | Probabilidad |\n|-----------|:------------:|\n"
    for k, v in proba.items():
        prob_md += f"| {k} | {'█'*int(v/5)} {v}% |\n"
    h_elo = ELO_2026.get(home_team, 1600)
    a_elo = ELO_2026.get(away_team, 1600)
    elo_str = (
        f"**Elo {home_team}**: {h_elo}  \n"
        f"**Elo {away_team}**: {a_elo}  \n"
        f"**Diferencia**: {elo_diff:+d}"
    )
    analysis = ai_analysis(home_team, away_team)
    return res_str, prob_md + "\n" + elo_str, f"🤖 {analysis}"


def app_matches():
    rows = conn.execute(
        "SELECT id,home_team,away_team,match_date,stage,ai_prediction,ai_confidence "
        "FROM matches ORDER BY match_date LIMIT 24"
    ).fetchall()
    md = "| # | Partido | Fecha | Fase | IA | Confianza |\n"
    md += "|---|---------|-------|------|:--:|:---------:|\n"
    for mid, h, a, dt, s, aip, aic in rows:
        conf = f"{aic:.1f}%" if aic else "—"
        md  += f"| {mid} | {h} vs {a} | {dt} | {s} | {aip or '—'} | {conf} |\n"
    return md


def app_submit(username, match_id, prediction, home_score, away_score):
    if not username.strip():
        return "❌ Ingresa un nombre de usuario."
    try:
        conn.execute("INSERT OR IGNORE INTO users (username) VALUES (?)", (username.strip(),))
        conn.commit()
        uid = conn.execute("SELECT id FROM users WHERE username=?", (username.strip(),)).fetchone()[0]
        try:
            hs  = int(home_score) if str(home_score).strip() not in ("","None") else None
            as_ = int(away_score) if str(away_score).strip() not in ("","None") else None
        except (ValueError, TypeError):
            hs = as_ = None
        conn.execute(
            "INSERT OR IGNORE INTO predictions (user_id,match_id,prediction,pred_home_score,pred_away_score) VALUES (?,?,?,?,?)",
            (uid, int(match_id), prediction, hs, as_)
        )
        conn.commit()
        return f"✅ **{username}** predijo Partido #{int(match_id)}: {prediction}"
    except Exception as e:
        return f"❌ Error: {e}"


def app_leaderboard():
    rows = conn.execute("""
        SELECT u.username, u.total_points, COUNT(p.id),
               SUM(CASE WHEN p.points_earned >= 3 THEN 1 ELSE 0 END)
        FROM users u
        LEFT JOIN predictions p ON u.id = p.user_id
        GROUP BY u.id ORDER BY u.total_points DESC
    """).fetchall()
    if not rows:
        return "Sin predicciones aún. ¡Sé el primero!"
    md  = "| Pos | Usuario | Puntos | Predicciones | Correctas |\n"
    md += "|:---:|---------|:------:|:------------:|:---------:|\n"
    medals = ["🥇","🥈","🥉"]
    for i, (u, p, pr, c) in enumerate(rows, 1):
        medal = medals[i-1] if i <= 3 else str(i)
        md   += f"| {medal} | {u} | {p} | {pr} | {c or 0} |\n"
    return md


# ── Inicializar predicciones IA en la BD (al arrancar la app) ─────────────────
for mid, h, a in conn.execute("SELECT id,home_team,away_team FROM matches").fetchall():
    pred, proba, _ = predict_match(h, a)
    conf = float(max(proba.values()))
    conn.execute("UPDATE matches SET ai_prediction=?, ai_confidence=? WHERE id=?", (pred, conf, mid))
conn.commit()


# ── Gradio UI ──────────────────────────────────────────────────────────────────
data_status = "✅ Forma reciente + H2H reales" if LATEST_TEAM_STATS else "⚠️ Solo Elo (defaults neutrales)"

with gr.Blocks(
    title="⚽ WC 2026 Predictor",
    theme=gr.themes.Soft(primary_hue="blue")
) as demo:

    gr.Markdown(
        "# ⚽ FIFA World Cup 2026 — Match Predictor\n"
        "### Powered by Machine Learning + Claude AI\n"
        f"*Model: {MODEL_NAME} | Accuracy: 0.5933 | 27 features | "
        f"Trained on competitive matches (1872-2026) | Data: {data_status}*"
    )

    with gr.Tabs():

        with gr.TabItem("🤖 Predicción IA"):
            with gr.Row():
                h_dd = gr.Dropdown(WC2026_TEAMS, label="🏠 Equipo Local",    value="Argentina")
                a_dd = gr.Dropdown(WC2026_TEAMS, label="✈️ Equipo Visitante", value="France")
            btn    = gr.Button("🔮 Predecir resultado", variant="primary")
            r_out  = gr.Markdown(label="Resultado")
            p_out  = gr.Markdown(label="Probabilidades")
            an_out = gr.Markdown(label="Análisis IA")
            btn.click(fn=app_predict, inputs=[h_dd, a_dd], outputs=[r_out, p_out, an_out])

        with gr.TabItem("📅 Próximos Partidos"):
            m_out = gr.Markdown()
            gr.Button("🔄 Actualizar", variant="secondary").click(fn=app_matches, outputs=m_out)
            demo.load(fn=app_matches, outputs=m_out)

        with gr.TabItem("📝 Mi Predicción"):
            u_in   = gr.Textbox(label="👤 Usuario", placeholder="ej: futbolero99")
            mid_in = gr.Number(label="🆔 ID Partido (ver tabla Próximos Partidos)", value=1, precision=0)
            p_in   = gr.Radio(["Home Win","Draw","Away Win"], label="🎯 Tu predicción", value="Home Win")
            with gr.Row():
                hs_in = gr.Number(label="⚽ Goles local (opcional, +2 pts)", value=None, precision=0)
                as_in = gr.Number(label="⚽ Goles visitante (opcional)",     value=None, precision=0)
            s_out = gr.Markdown()
            gr.Button("📤 Enviar predicción", variant="primary").click(
                fn=app_submit, inputs=[u_in, mid_in, p_in, hs_in, as_in], outputs=s_out
            )

        with gr.TabItem("🏆 Leaderboard"):
            lb_out = gr.Markdown()
            gr.Button("🔄 Actualizar", variant="secondary").click(fn=app_leaderboard, outputs=lb_out)
            demo.load(fn=app_leaderboard, outputs=lb_out)

    gr.Markdown(
        "---\n"
        "**Scoring**: +3 resultado correcto · +2 marcador exacto · 0 fallo  \n"
        "*Data: martj42/Kaggle (CC BY-SA) · Elo: eloratings.net*"
    )

if __name__ == "__main__":
    demo.launch()
