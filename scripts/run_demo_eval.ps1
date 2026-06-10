$ErrorActionPreference = 'Stop'

python -c "import datetime as dt; from weather_agent.evaluator import evaluate_recent; res=evaluate_recent(city='Tianjin', start_date=dt.date(2025,4,10), end_date=dt.date(2025,4,12), truth_policy='require', min_truth_coverage=0.6, min_total_positive_labels=5, force_rebuild_truth=True); print('statistical_validity:', res.get('statistical_validity')); print('improvements:', res.get('improvements')); print('reports:', res.get('reports'))"
