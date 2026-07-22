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


def _showcase():
    return json.loads((E.TOOLS_DIR / "data" / "l0_eval_results.json").read_text(encoding="utf-8"))


def test_results_showcase_matches_gold():
    res = _showcase()
    assert res["gold_set_n"] == len(GOLD50["items"])
    assert res["gold_set"].endswith("l0_gold_set_50.json")
    assert any(m["alias"] == "haiku" for m in res["models"]), "нет Haiku в витрине результатов"


# Порог качества классификатора. Прогон требует API-ключа и в CI не выполняется, поэтому
# гейтим САМУ ВИТРИНУ: её нельзя тихо переписать числами хуже заявленных в docs/L0_EVAL.md.
MIN_SUBCATEGORY_ACCURACY = 0.90
MIN_CI_LOWER = 0.83


def test_showcase_meets_declared_thresholds():
    full = [m for m in _showcase()["models"] if m.get("n") == len(GOLD50["items"])]
    assert full, "в витрине нет ни одного прогона на полном gold-set"
    for m in full:
        acc = m["subcategory_accuracy"]
        assert acc >= MIN_SUBCATEGORY_ACCURACY, f"{m['alias']}: subcategory accuracy {acc} упала"
        lo = m["subcategory_accuracy_ci95"][0]
        assert lo >= MIN_CI_LOWER, (
            f"{m['alias']}: нижняя граница 95% CI {lo} ниже заявленной — на N=50 это уже "
            f"не «почти то же самое»"
        )


def test_showcase_misses_are_explained():
    """Каждый промах разобран: без разбора «94 %» превращается в цифру без содержания,
    и непонятно, где ошибка модели, а где спорная эталонная метка."""
    gold_by_id = {it["id"]: it for it in GOLD50["items"]}
    # Строки частичных прогонов (подмножество gold-set) имеют другую схему полей —
    # разбор промахов требуем с полных прогонов.
    for m in [m for m in _showcase()["models"] if "subcategory_correct" in m]:
        n_wrong = m["n"] - int(m["subcategory_correct"].split("/")[0])
        assert (
            len(m.get("misses", [])) == n_wrong
        ), f"{m['alias']}: промахов {n_wrong}, а разобрано {len(m.get('misses', []))}"
        for miss in m.get("misses", []):
            assert miss["id"] in gold_by_id, f"{m['alias']}: промах по неизвестному id {miss['id']}"
            assert (
                gold_by_id[miss["id"]]["gold_sub"] == miss["gold"]
            ), f"{m['alias']}/{miss['id']}: эталон в витрине расходится с gold-set"
            assert miss.get("note", "").strip(), f"{m['alias']}/{miss['id']}: промах без разбора"
