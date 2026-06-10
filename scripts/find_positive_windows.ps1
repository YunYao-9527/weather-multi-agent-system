param(
  [string]$City = "Tianjin",
  [string]$SearchStart = "2025-03-01",
  [string]$SearchEnd = "2025-10-31",
  [int]$WindowDays = 3,
  [int]$StepDays = 1,
  [double]$MinTruthCoverage = 0.6,
  [int]$MinTotalPositiveLabels = 1,
  [int]$MinTrainPositiveLabels = 1,
  [int]$MinCalibrationPositiveLabels = 1,
  [string]$HeadlineTiers = "gold,silver",
  [int]$TopK = 20,
  [switch]$ForceRebuildTruth
)

$ErrorActionPreference = 'Stop'

$args = @(
  "--city", $City,
  "--search-start", $SearchStart,
  "--search-end", $SearchEnd,
  "--window-days", "$WindowDays",
  "--step-days", "$StepDays",
  "--min-truth-coverage", "$MinTruthCoverage",
  "--min-total-positive-labels", "$MinTotalPositiveLabels",
  "--min-train-positive-labels", "$MinTrainPositiveLabels",
  "--min-calibration-positive-labels", "$MinCalibrationPositiveLabels",
  "--headline-tiers", $HeadlineTiers,
  "--top-k", "$TopK"
)

if ($ForceRebuildTruth) {
  $args += "--force-rebuild-truth"
}

python -m weather_agent.window_selector @args
