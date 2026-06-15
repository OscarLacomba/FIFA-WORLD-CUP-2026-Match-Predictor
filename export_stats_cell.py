# ── Exportar snapshot de forma reciente y head-to-head para app.py ────────────
# Ejecutar esta celda DESPUÉS de la celda de predict_match (donde se construye
# latest_team_stats y existe pair_history en memoria)

import pickle

with open('latest_team_stats.pkl', 'wb') as f:
    pickle.dump(latest_team_stats, f)

with open('pair_history.pkl', 'wb') as f:
    pickle.dump(pair_history, f)

print(f'✅ latest_team_stats.pkl guardado — {len(latest_team_stats)} equipos')
print(f'✅ pair_history.pkl guardado — {len(pair_history)} pares de equipos')

# Descargar a tu computadora
from google.colab import files
files.download('latest_team_stats.pkl')
files.download('pair_history.pkl')
