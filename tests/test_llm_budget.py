from concurrent.futures import ThreadPoolExecutor

import pytest

from weather_agent.llm_budget import DemoCallBudget, DemoLLMLimitError


def test_daily_budget_is_atomic_and_resets_on_new_day(monkeypatch):
    monkeypatch.setenv('PUBLIC_DEMO', '1')
    monkeypatch.setenv('LLM_DAILY_CALL_LIMIT', '3')
    budget = DemoCallBudget()
    def attempt(_):
        try:
            budget.reserve({'input': 'test'})
            return 1
        except DemoLLMLimitError:
            return 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(attempt, range(20))) == 3
    budget._day = '1970-01-01'
    budget.reserve({'input': 'next day'})
    assert budget._used == 1


def test_oversized_input_is_rejected_before_reserving_a_call(monkeypatch):
    monkeypatch.setenv('PUBLIC_DEMO', '1')
    monkeypatch.setenv('LLM_INPUT_CHAR_LIMIT', '20')
    budget = DemoCallBudget()
    with pytest.raises(DemoLLMLimitError):
        budget.reserve({'input': 'x' * 100})
    assert budget._used == 0


def test_private_runtime_does_not_apply_public_budget(monkeypatch):
    monkeypatch.setenv('PUBLIC_DEMO', '0')
    monkeypatch.setenv('LLM_DAILY_CALL_LIMIT', '0')
    DemoCallBudget().reserve({'input': 'private'})

