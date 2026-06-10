if (-not $env:AGENT_API_TOKEN) { $env:AGENT_API_TOKEN = 'agent-dev-token' }
if (-not $env:AGENT_ENABLE_AUTH) { $env:AGENT_ENABLE_AUTH = '0' }
if (-not $env:AGENT_CORS_ORIGINS) { $env:AGENT_CORS_ORIGINS = '*' }
$hostAddress = $env:AGENT_API_HOST
if (-not $hostAddress) { $hostAddress = '0.0.0.0' }

$port = $env:AGENT_API_PORT
if (-not $port) { $port = '8000' }

$reload = $env:AGENT_API_RELOAD
if (-not $reload) { $reload = "1" }

if ($reload -eq "1") {
  python -m uvicorn weather_agent.api:app --host $hostAddress --port $port --reload
} else {
  python -m uvicorn weather_agent.api:app --host $hostAddress --port $port
}
