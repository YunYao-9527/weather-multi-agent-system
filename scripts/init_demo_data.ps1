$ErrorActionPreference = 'Stop'
python - <<'PY'
import datetime as dt
from weather_agent.truth_labels import build_truth_label_artifact, TruthConfig

art = build_truth_label_artifact(
    city='Tianjin',
    start_date=dt.date(2025,3,1),
    end_date=dt.date(2025,3,3),
    cfg=TruthConfig(),
    force_rebuild=True,
)
print('truth artifact:', art.get('artifact_path'))
print('coverage:', art.get('meta', {}).get('label_coverage_ratio'))
PY
