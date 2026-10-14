# 课堂行为检测 MCP 服务一键重启脚本
# 用法: 在 PowerShell 中执行  .\scripts\restart_mcp_server.ps1
# 说明: 代码本身不读 .env, 必须由启动环境注入。本脚本负责从 scripts/.env 注入环境变量。
$ErrorActionPreference = "Stop"
$ProjectDir = Resolve-Path (Join-Path $PSScriptRoot '..') | Select-Object -ExpandProperty Path
$Python = if ($env:MCP_PYTHON) { $env:MCP_PYTHON } else { (Get-Command python -ErrorAction SilentlyContinue).Source }
if (-not $Python) { Write-Error '未找到 Python，请先安装 Python 3.12 并在环境变量 MCP_PYTHON 中指定 python.exe 路径'; exit 1 }
$Script = "$ProjectDir\scripts\mcp_server.py"
$LogFile = "$ProjectDir\scripts\server.log"
$Port = 9800

# ---- 从 .env 注入环境变量(代码不读 .env, 必须由启动环境提供) ----
$envFile = "$ProjectDir\scripts\.env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith('#')) {
            if ($line -match '^([^=]+)=(.*)$') {
                $k = $matches[1].Trim()
                $v = $matches[2].Trim()
                Set-Item -Path "env:$k" -Value $v
            }
        }
    }
    Write-Host "已从 .env 注入环境变量" -ForegroundColor Gray
} else {
    Write-Host "未找到 .env, 将使用代码默认值" -ForegroundColor Yellow
}

function Stop-OldServer {
    $lines = netstat -ano | Select-String ":$Port\s+.*LISTENING\s+(\d+)"
    $pids = $lines | ForEach-Object { $_.Matches[0].Groups[1].Value } | Sort-Object -Unique
    foreach ($pid in $pids) {
        if ($pid -and $pid -ne "0") {
            Write-Host "发现旧服务占用 ${Port}, PID=$pid, 正在停止..." -ForegroundColor Yellow
            Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
            Start-Sleep -Seconds 1
        }
    }
}

function Start-NewServer {
    if (Test-Path $LogFile) {
        $rotate = "$LogFile.$((Get-Date).ToString('yyyyMMddHHmmss'))"
        Move-Item $LogFile $rotate -Force
        Write-Host "旧日志已归档: $rotate" -ForegroundColor Gray
    }
    $proc = Start-Process -FilePath $Python -ArgumentList $Script `
        -WorkingDirectory $ProjectDir `
        -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $LogFile -RedirectStandardError $LogFile
    Write-Host "新服务已启动, PID=$($proc.Id), 端口=$Port" -ForegroundColor Green
    Write-Host "日志: $LogFile" -ForegroundColor Gray
    Write-Host "SSE:  http://127.0.0.1:${Port}/sse" -ForegroundColor Cyan
    Write-Host "REST: http://127.0.0.1:${Port}/tools/call" -ForegroundColor Cyan
    Write-Host "PROJ: $env:MCP_PROJECT_DIR" -ForegroundColor Cyan
    Write-Host "记得确认 ngrok / cloudflared 隧道仍指向 127.0.0.1:${Port}" -ForegroundColor Yellow
}

Stop-OldServer
Start-Sleep -Seconds 1
Start-NewServer
Write-Host "等待模型加载..." -ForegroundColor Cyan
Start-Sleep -Seconds 10
$health = curl -s -H "Authorization: Bearer $env:MCP_AUTH_TOKEN" "http://127.0.0.1:${Port}/healthz"
Write-Host "healthz: $health"
