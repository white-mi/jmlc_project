"""CI-safe тесты gold-set L0-eval (N=50): схема, покрытие 27 подкатегорий, валидность
меток, наличие boundary-кейсов. Без API (импорт eval_l0_classifier ленив к anthropic)."""

import json

import eval_l0_classifier as E

GOLD50 = json.loads((E.TOOLS_DIR / "data" / "l0_gold_set_50.json").read_text(encoding="utf-8"))
TAXONOMY = {
    "1": {"1", "2", "3", "4", "5"},
    "2": {"1", "2", "3", "4", "5", "6"},
    "3": {"1", "2", "3", "4", "5"},
    "4": {"1", "2", "3", "4", "5", "6", "7"},
    "5": {"1", "2", "3", "4"},
}
VALID_SUBS = {f"{m}.{s}" for m, subs in TAXONOMY.items() for s in subs}  # 27 подкатегорий


def test_gold50_size_and_fields():
    items = GOLD50["items"]
    assert len(items) == 50
    for it in items:
        for f in ("id", "gold_main", "gold_sub", "boundary", "source", "date", "news"):
            assert f in it, f"{it.get('id')}: нет поля {f}"
        assert it["news"].strip() and len(it["news"]) > 20, f"{it['id']}: пустая/короткая news"


def test_gold50_unique_ids():
    ids = [it["id"] for it in GOLD50["items"]]
    assert len(set(ids)) == len(ids), "дубли id"


def test_gold50_labels_valid():
    for it in GOLD50["items"]:
        assert it["gold_sub"] in VALID_SUBS, f"{it['id']}: невалидная подкат {it['gold_sub']}"
        assert it["gold_main"] == it["gold_sub"].split(".")[0], f"{it['id']}: main≠префикс sub"


def test_gold50_covers_all_27_subcategories():
    covered = {it["gold_sub"] for it in GOLD50["items"]}
    assert covered == VALID_SUBS, f"не покрыты подкатегории: {sorted(VALID_SUBS - covered)}"


def test_gold50_has_boundary_cases():
    nb = sum(bool(it["boundary"]) for it in GOLD50["items"])
    assert nb >= 5, f"слишком мало boundary-кейсов: {nb}"


def test_results_showcase_matches_gold():
    res = json.loads((E.TOOLS_DIR / "data" / "l0_eval_results.json").read_text(encoding="utf-8"))
    assert res["gold_set_n"] == len(GOLD50["items"])
    assert res["gold_set"].endswith("l0_gold_set_50.json")
    assert any(m["alias"] == "haiku" for m in res["models"]), "нет Haiku в витрине результатов"
