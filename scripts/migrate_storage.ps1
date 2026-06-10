$ErrorActionPreference = 'Stop'

param(
  [switch]$DryRun,
  [switch]$NoBackup
)

$args = @()
if ($DryRun) { $args += "--dry-run" }
if ($NoBackup) { $args += "--no-backup" }

python -m weather_agent.migrations @args
