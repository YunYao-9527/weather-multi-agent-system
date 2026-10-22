"""Bound paid model usage in public demonstrations, without storing credentials."""
import json
import os
from datetime import datetime, timezone
from threading import Lock


class DemoLLMLimitError(RuntimeError):
    pass


class DemoCallBudget:
    def __init__(self):
        self._lock = Lock()
        self._day = ""
        self._used = 0

    def reserve(self, payload):
        if os.getenv("PUBLIC_DEMO", "0") != "1":
            return
        size = len(json.dumps(payload, ensure_ascii=False))
        max_chars = min(max(int(os.getenv("LLM_INPUT_CHAR_LIMIT", "24000")), 1), 40000)
        if size > max_chars:
            raise DemoLLMLimitError("演示会话过长，请刷新页面开始新会话。")
        limit = min(max(int(os.getenv("LLM_DAILY_CALL_LIMIT", "40")), 0), 200)
        day = datetime.now(timezone.utc).date().isoformat()
        with self._lock:
            if self._day != day:
                self._day, self._used = day, 0
            if self._used >= limit:
                raise DemoLLMLimitError("公开演示的模型调用额度已用完，请稍后再试；规则和业务工具仍可使用。")
            self._used += 1


model_budget = DemoCallBudget()

