$ErrorActionPreference = 'Stop'
python - <<'PY'
import datetime as dt
from weather_agent.evaluator import evaluate_recent

res = evaluate_recent(
    city='Tianjin',
    start_date=dt.date(2025,3,1),
    end_date=dt.date(2025,3,3),
    truth_policy='require',
    min_truth_coverage=0.6,
)
print('json:', res.get('reports', {}).get('json'))
print('markdown:', res.get('reports', {}).get('markdown'))
print('html:', res.get('reports', {}).get('html'))
PY
