@echo off
chcp 65001 >nul
cd /d "%~dp0"
title 课堂行为检测 MCP 服务

REM ---- 1) 注入 scripts\.env 环境变量(代码本身不读.env, 必须由这里注入) ----
if exist "scripts\.env" (
  for /f "usebackq eol=# tokens=1,* delims==" %%a in ("scripts\.env") do (
    if not "%%a"=="" set "%%a=%%b"
  )
)
REM 没设 MCP_PROJECT_DIR 则自动用本目录(解压到任意路径都能跑)
if not defined MCP_PROJECT_DIR set "MCP_PROJECT_DIR=%~dp0"

REM ---- 端口 / 并发数兜底 ----
if not defined MCP_PORT set "MCP_PORT=9800"
if not defined MCP_MAX_CONCURRENT set "MCP_MAX_CONCURRENT=2"

REM ---- 2) 检查系统 Python ----
where python >nul 2>nul || (
  echo [错误] 没找到 python, 请先安装 Python 3.12 (64位) 并勾选 "Add to PATH"
  echo 官方下载: https://www.python.org/downloads/
  pause & exit /b 1
)

REM ---- 3) 自动建虚拟环境并安装依赖(仅首次, 之后秒开) ----
if not exist ".venv\Scripts\python.exe" (
  echo 首次运行: 正在创建虚拟环境 .venv ...
  python -m venv .venv
)
set "PY=.venv\Scripts\python.exe"
if not exist ".venv\Lib\site-packages\ultralytics" (
  echo 首次运行: 正在安装依赖(约几分钟, 只需一次) ...
  "%PY%" -m pip install --upgrade pip
  "%PY%" -m pip install -r requirements.txt
)

REM ---- 4) 杀掉旧进程 ----
for /f "tokens=5" %%p in ('netstat -ano ^| findstr :%MCP_PORT% ^| findstr LISTENING') do taskkill /f /pid %%p >nul 2>nul

REM ---- 5) 启动 ----
echo 项目目录: %MCP_PROJECT_DIR%
echo 端口: %MCP_PORT%
if not defined MCP_AUTH_TOKEN (echo 鉴权: 未设置token(本地调试免鉴权模式) ^| 生产环境请在 scripts\.env 设 MCP_AUTH_TOKEN) else (echo 鉴权: 已启用 Bearer Token)
echo 正在启动 uvicorn... (首次加载模型约需 10-30 秒，看到 "Uvicorn running" 即成功)
echo 启动后访问 http://127.0.0.1:%MCP_PORT%/healthz 验证
echo.
"%PY%" -m uvicorn scripts.mcp_server:app --host 0.0.0.0 --port %MCP_PORT%
pause
