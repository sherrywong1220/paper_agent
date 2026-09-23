"""Exercise the CLI subprocess boundary without network calls or user credentials."""
import asyncio
import json
import os
from pathlib import Path
import sys
from unittest.mock import AsyncMock, patch

import pytest
from sqlmodel import select

from src.config import settings
from src.models import LLMUsage
from src.services import codex_cli as cli
from src.services.llm import LLMService
from src.services.settings_service import get_llm_config, update_settings


@pytest.fixture
def fake_cli(tmp_path, monkeypatch):
    executable = tmp_path / "fake codex"
    executable.write_text(f"#!{sys.executable}\n" + r'''
import json, os, pathlib, sys, time
args = sys.argv[1:]
if args == ["login", "status"]:
    print("Logged in using ChatGPT")
    sys.exit(0)
prompt = sys.stdin.read()
if "FAKE_TIMEOUT" in prompt:
    pathlib.Path(os.environ["TEST_PID_FILE"]).write_text(str(os.getpid()))
    time.sleep(60)
if "FAKE_QUOTA" in prompt:
    print(json.dumps({"type": "error", "message": "usage_limit_reached sk-secret-never-log"}))
    sys.exit(1)
if "FAKE_AUTH" in prompt:
    print("Please log in", file=sys.stderr)
    sys.exit(1)
if "FAKE_ERROR" in prompt:
    print("unknown model sk-secret-never-log", file=sys.stderr)
    sys.exit(2)
output = pathlib.Path(args[args.index("--output-last-message") + 1])
if "FAKE_BAD" in prompt:
    output.write_text('{"result": "not a JSON object"}')
elif "FAKE_MISSING" not in prompt:
    answer = {"score": 91, "relevance": 5, "novelty": 4, "clarity": 4,
              "one_line_reason": "相关", "risk_flags": []}
    if "FAKE_INSPECT" in prompt:
        answer = {"args": args, "cwd": os.getcwd(), "prompt": prompt,
                  "has_api_key": any(k in os.environ for k in ("CODEX_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY")),
                  "codex_home": os.environ.get("CODEX_HOME")}
    output.write_text(json.dumps({"result": json.dumps(answer, ensure_ascii=False)}), encoding="utf-8")
print(json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "Ignore this progress text"}}))
print(json.dumps({"type": "turn.completed", "usage": {"input_tokens": 120, "cached_input_tokens": 20, "output_tokens": 30}}))
''', encoding="utf-8")
    executable.chmod(0o700)
    monkeypatch.setattr(settings, "CODEX_CLI_PATH", str(executable))
    monkeypatch.setattr(settings, "LLM_BACKEND", "codex-cli")
    monkeypatch.setattr(settings, "CODEX_MODEL", "")
    monkeypatch.setattr(cli, "_paused_until", 0)
    monkeypatch.setattr(cli, "_pause_reason", "")
    return executable


async def test_subprocess_stdin_isolation_and_structured_output(fake_cli, monkeypatch):
    for key in ("CODEX_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.setenv(key, "secret")
    monkeypatch.setenv("CODEX_HOME", "/example/auth-store")
    prompt = 'FAKE_INSPECT 中文 $(touch SHOULD_NOT_EXIST) `echo unsafe`\n' + "论文 " * 20000
    result = await cli.generate(prompt, model="test-model", json_mode=True)
    data = json.loads(result.text)
    assert prompt in data["prompt"] and prompt not in data["args"]
    assert not data["has_api_key"] and data["codex_home"] == "/example/auth-store"
    assert data["args"][data["args"].index("--sandbox") + 1] == "read-only"
    assert 'forced_login_method="chatgpt"' in data["args"]
    assert 'approval_policy="never"' in data["args"]
    assert "--ignore-user-config" in data["args"] and "--ignore-rules" in data["args"]
    assert "test-model" in data["args"]
    assert not Path(data["cwd"]).exists()  # all prompt/output files removed
    assert result.usage == {"prompt_tokens": 120, "completion_tokens": 30}


async def test_login_status_is_sanitized(fake_cli):
    status = await cli.status()
    assert status["installed"] and status["subscription_login"]
    assert "ChatGPT" not in json.dumps(status)  # no raw command output


@pytest.mark.parametrize("prompt", ["FAKE_BAD", "FAKE_MISSING"])
async def test_invalid_or_missing_result_rejected(fake_cli, prompt):
    with pytest.raises(cli.CodexCLIError, match="invalid structured result"):
        await cli.generate(prompt, json_mode=True)


@pytest.mark.parametrize("prompt, expected", [("FAKE_QUOTA", "usage limit"), ("FAKE_AUTH", "authentication")])
async def test_auth_and_quota_pause_without_retry(fake_cli, prompt, expected):
    with pytest.raises(cli.CodexCLIError, match=expected) as error:
        await cli.generate(prompt)
    assert "sk-secret" not in str(error.value)
    with patch.object(cli, "_run", AsyncMock()) as run:
        with pytest.raises(cli.CodexCLIError, match="paused"):
            await cli.generate("next paper")
        run.assert_not_called()


async def test_other_errors_are_sanitized(fake_cli):
    with pytest.raises(cli.CodexCLIError, match="exit 2") as error:
        await cli.generate("FAKE_ERROR")
    assert "sk-secret" not in str(error.value)


async def test_timeout_kills_subprocess(fake_cli, monkeypatch, tmp_path):
    pid_file = tmp_path / "pid"
    monkeypatch.setenv("TEST_PID_FILE", str(pid_file))
    monkeypatch.setattr(settings, "CODEX_TIMEOUT_SECONDS", 0.2)
    with pytest.raises(cli.CodexCLIError, match="timed out"):
        await cli.generate("FAKE_TIMEOUT")
    pid = int(pid_file.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


async def test_cancellation_kills_subprocess(fake_cli, monkeypatch, tmp_path):
    pid_file = tmp_path / "pid"
    monkeypatch.setenv("TEST_PID_FILE", str(pid_file))
    task = asyncio.create_task(cli.generate("FAKE_TIMEOUT"))
    async with asyncio.timeout(5):
        while not pid_file.exists():
            await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)


async def test_serializes_calls_across_service_instances(fake_cli):
    running = peak = 0

    async def fake_run(args, **kwargs):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.02)
        Path(args[args.index("--output-last-message") + 1]).write_text('{"result":"ok"}')
        running -= 1
        return 0, "", ""

    with patch.object(cli, "_run", side_effect=fake_run):
        await asyncio.gather(*(cli.generate("paper") for _ in range(4)))
    assert peak == 1


async def test_missing_cli_is_actionable(monkeypatch):
    monkeypatch.setattr(settings, "CODEX_CLI_PATH", "/nonexistent/codex")
    monkeypatch.setattr(cli, "_paused_until", 0)
    assert not (await cli.status())["installed"]
    with pytest.raises(cli.CodexCLIError, match="not found"):
        await cli.generate("paper")


def test_backend_switch_and_settings(client, env_file, fake_cli):
    update_settings({"LLM_BACKEND": "api"})
    original = get_llm_config()
    response = client.put("/api/settings", json={"values": {"LLM_BACKEND": "codex-cli"}})
    assert response.status_code == 200
    assert response.json()["provider"]["requires_api_key"] is False
    with patch("src.main.model_catalog.refresh", AsyncMock()) as refresh:
        response = client.put("/api/settings/llm", json={"codex_model": "test-codex", "score_threshold": 89})
        assert response.status_code == 200
        assert response.json()["models"]["summary"] == "codex/test-codex"
        assert client.get("/api/models").json()["models"] == []
        estimate = client.get("/api/llm/estimate").json()
        assert estimate["available"] is False and estimate["total_per_day"] is None
        refresh.assert_not_called()
    assert client.get("/api/llm/codex-status").json()["subscription_login"] is True
    assert client.put("/api/settings/llm", json={"codex_model": "anthropic/model"}).status_code == 400
    assert client.put("/api/settings", json={"values": {"CODEX_TIMEOUT_SECONDS": 0}}).status_code == 400
    update_settings({"LLM_BACKEND": "api"})
    assert get_llm_config().stage1_model == original.stage1_model
    update_settings({"LLM_BACKEND": "codex-cli", "CODEX_MODEL": ""})
    assert get_llm_config().stage1_model == "codex/default"


async def test_llm_dispatch_and_unpriced_usage(fake_cli, session, client):
    with patch("src.services.llm.AsyncOpenAI") as api, \
         patch("src.database.engine", session.bind), \
         patch("src.services.llm.model_catalog.estimate_cost", return_value=99) as pricing:
        service = LLMService()
        result = await service._chat("score_stage1", "score this paper", json_mode=True, temperature=0)
        assert json.loads(result)["score"] == 91
        api.assert_not_called()
        pricing.assert_not_called()
    row = session.exec(select(LLMUsage)).one()
    assert row.model == "codex/default" and row.cost is None and not row.cost_estimated
    assert row.total_tokens == 150 and row.success
    usage = client.get("/api/llm/usage").json()
    assert usage["periods"]["all_time"]["cost"] is None
    assert usage["periods"]["all_time"]["unpriced_calls"] == 1
    assert usage["breakdown"][0]["cost_per_call"] is None


async def test_cli_failure_does_not_fall_back_to_api(fake_cli):
    with patch("src.services.llm.AsyncOpenAI") as api, patch("src.services.llm._record_usage"):
        result = await LLMService()._chat("summarize", "FAKE_ERROR", json_mode=False, temperature=0)
    assert result is None
    api.assert_not_called()


async def test_no_api_key_skips_auto_embeddings(fake_cli, monkeypatch):
    from src.services import embedding_service
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "")
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "")
    with patch.object(embedding_service, "embed_papers", AsyncMock()) as embed:
        assert await embedding_service.embed_new_papers(["some-paper"]) == 0
        embed.assert_not_called()
