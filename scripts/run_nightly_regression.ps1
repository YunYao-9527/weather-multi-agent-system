$ErrorActionPreference = 'Stop'

param(
  [switch]$NoGate
)

$args = @()
if (-not $NoGate) { $args += "--enforce-gate" }

python -m weather_agent.nightly @args
