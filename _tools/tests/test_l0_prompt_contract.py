"""L0 prompt-contract regression (без API, CI-safe).

Закрепляет инвариант: каждый ключ, который orchestrator РЕАЛЬНО подставляет в
промпт агента (`run_pipeline()` → `run_agent()` → `fill_prompt()`), присутствует
в блоке `## Промпт` этого агента как `<KEY>` плейсхолдер, а сам блок извлекается
`extract_prompt()` целиком (не обрезается вложенным code-fence).

ЗАЧЕМ: плейсхолдеры промпта (`<НОВОСТЬ>` / `<STATE_JSON>` / `<RAG_ANALOGS>` / …) обязаны
совпадать с ключами, которые подставляет `fill_prompt`, иначе данные не инжектятся в
реальном LLM-режиме, а dry-run-заглушка (`_stub_response`) этого не покажет (она не
вызывает `fill_prompt`). Отдельно проверяется, что вложенный ```-fence не обрывает
`extract_prompt` (вложенные блоки — на `~~~`).

Эти тесты ловят оба класса регрессий офлайн, до траты токенов.
Контракт синхронизирован с `orchestrator.run_pipeline()` (ключи подстановки).
"""

import pytest

import orchestrator as O

# Ключи, которые orchestrator подставляет каждому агенту на основном пути
# (см. run_pipeline: Agent1 НОВОСТЬ/ИСТОЧНИК/ДАТА; Agent2 STATE_JSON;
#  Agent3 STATE_JSON+RAG_ANALOGS; Agent4 STATE_JSON+BIFURCATION+OSL_RESULTS;
#  Agent5 STATE_JSON [+MODE только на shortcut-пути, поэтому не обязателен]).
REQUIRED = {
    1: ["НОВОСТЬ", "ИСТОЧНИК", "ДАТА"],
    2: ["STATE_JSON"],
    3: ["STATE_JSON", "RAG_ANALOGS"],
    4: ["STATE_JSON", "BIFURCATION", "OSL_RESULTS"],
    5: ["STATE_JSON"],
}
FILES = {
    1: "agent_1_classifier.md",
    2: "agent_2_context_rag.md",
    3: "agent_3_backtest_analog.md",
    4: "agent_4_impact.md",
    5: "agent_5_summary.md",
}
AGENTS_DIR = O.AGENTS_DIR


def _prompt(n):
    return O.extract_prompt(AGENTS_DIR / FILES[n])


@pytest.mark.parametrize("n", sorted(REQUIRED))
def test_required_placeholders_present(n):
    """Каждый ключ подстановки orchestrator присутствует как <KEY> в промпте."""
    prompt = _prompt(n)
    missing = [k for k in REQUIRED[n] if f"<{k}>" not in prompt]
    assert not missing, (
        f"Agent {n} ({FILES[n]}): orchestrator подставляет {missing}, но в "
        f"промпте нет плейсхолдеров {[f'<{k}>' for k in missing]}. "
        f"fill_prompt оставит их без замены → данные не дойдут до LLM."
    )


@pytest.mark.parametrize("n", sorted(FILES))
def test_fill_prompt_leaves_no_required_placeholder(n):
    """После fill_prompt со всеми ключами orchestrator ни один <KEY> не остаётся.

    Прямая симуляция run_agent(): VAULT_ROOT/RADAR_ROOT инжектятся всегда."""
    prompt = _prompt(n)
    subs = {"VAULT_ROOT": "V", "RADAR_ROOT": "R"}
    subs.update({k: f"<<{k}_VALUE>>" for k in REQUIRED[n]})
    filled = O.fill_prompt(prompt, subs)
    leftover = [k for k in REQUIRED[n] if f"<{k}>" in filled]
    assert not leftover, (
        f"Agent {n} ({FILES[n]}): после подстановки остались незаменёнными "
        f"{[f'<{k}>' for k in leftover]}."
    )


@pytest.mark.parametrize("n", sorted(FILES))
def test_prompt_not_truncated_by_nested_fence(n):
    """Извлечённый промпт доходит до финальной строки «Выход — …».

    Вложенные блоки используют ~~~ (не ```), иначе ```-fence обрывает extract_prompt."""
    prompt = _prompt(n)
    assert "Выход —" in prompt, (
        f"Agent {n} ({FILES[n]}): extract_prompt вернул {len(prompt)} симв. без "
        f"финальной строки «Выход —» — вероятно, промпт обрезан вложенным "
        f"code-fence. Замени внутренние ``` на ~~~."
    )
