#!/bin/bash
# macOS: double-click this file to start Paper Agent in the background.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Finder-launched shells may not include the Codex CLI installation directory.
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

PYTHON="$PROJECT_DIR/.venv/bin/python"
APP_URL="http://127.0.0.1:8000"
LOG_FILE="$PROJECT_DIR/data/server.log"
PID_FILE="$PROJECT_DIR/data/server.pid"

if [[ ! -x "$PYTHON" ]]; then
    printf '未找到 Python 虚拟环境：%s\n请先在项目目录安装依赖。\n' "$PYTHON" >&2
    exit 1
fi

mkdir -p "$PROJECT_DIR/data"

is_running() {
    /usr/bin/curl --fail --silent --max-time 2 "$APP_URL/health" |
        "$PYTHON" -c 'import json, sys; d = json.load(sys.stdin); sys.exit(0 if d.get("message", "").startswith("Welcome to Paper Agent.") and d.get("api_prefix") == "/api" else 1)' 2>/dev/null
}

open_page() {
    printf '访问地址：%s\n日志文件：%s\n' "$APP_URL" "$LOG_FILE"
    /usr/bin/open "$APP_URL" >/dev/null 2>&1 || true
}

if is_running; then
    printf 'Paper Agent 已在运行，无需重复启动。\n'
    open_page
    exit 0
fi

if /usr/sbin/lsof -nP -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1; then
    printf '端口 8000 已被占用，但 Paper Agent 尚未就绪。请检查占用该端口的服务。\n' >&2
    exit 1
fi

nohup "$PYTHON" -u -m uvicorn src.main:app \
    --host 127.0.0.1 --port 8000 \
    > "$LOG_FILE" 2>&1 < /dev/null &
SERVER_PID=$!
printf '%s\n' "$SERVER_PID" > "$PID_FILE"

STARTUP_DEADLINE=$((SECONDS + 20))
while (( SECONDS < STARTUP_DEADLINE )); do
    if is_running; then
        printf 'Paper Agent 已在后台启动（PID: %s）。可以关闭此终端窗口。\n' "$SERVER_PID"
        open_page
        exit 0
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        break
    fi
    sleep 1
done

printf '暂未确认服务启动成功，请查看日志：%s\n' "$LOG_FILE" >&2
exit 1
