"""CI-safe тесты РЕАЛЬНОГО silver-gold-set L0-eval (l0_gold_set_real.json): схема,
покрытие 27 подкатегорий, валидность меток, ПРОВЕНАНС (реальный url на каждую новость),
метки silver с указанным аннотатором. Без API."""

import json

import eval_l0_classifier as E

REAL = json.loads((E.TOOLS_DIR / "data" / "l0_gold_set_real.json").read_text(encoding="utf-8"))
TAXONOMY = {
    "1": {"1", "2", "3", "4", "5"},
    "2": {"1", "2", "3", "4", "5", "6"},
    "3": {"1", "2", "3", "4", "5"},
    "4": {"1", "2", "3", "4", "5", "6", "7"},
    "5": {"1", "2", "3", "4"},
}
VALID_SUBS = {f"{m}.{s}" for m, subs in TAXONOMY.items() for s in subs}  # 27 подкатегорий


def test_real_size_and_fields():
    items = REAL["items"]
    assert len(items) == REAL["n"] == 93
    for it in items:
        for f in ("id", "gold_main", "gold_sub", "boundary", "source", "date", "url", "news"):
            assert f in it, f"{it.get('id')}: нет поля {f}"
        assert it["news"].strip() and len(it["news"]) > 30, f"{it['id']}: пустая/короткая news"


def test_real_provenance_urls():
    """Каждая новость — реальная, с http(s)-URL первоисточника (то, что убивает
    претензию «синтетика»)."""
    for it in REAL["items"]:
        assert it["url"].startswith("http"), f"{it['id']}: url без провенанса ({it['url']!r})"


def test_real_unique_ids():
    ids = [it["id"] for it in REAL["items"]]
    assert len(set(ids)) == len(ids), "дубли id"


def test_real_labels_valid():
    for it in REAL["items"]:
        assert it["gold_sub"] in VALID_SUBS, f"{it['id']}: невалидная подкат {it['gold_sub']}"
        assert it["gold_main"] == it["gold_sub"].split(".")[0], f"{it['id']}: main≠префикс sub"


def test_real_covers_all_27_subcategories():
    covered = {it["gold_sub"] for it in REAL["items"]}
    assert covered == VALID_SUBS, f"не покрыты подкатегории: {sorted(VALID_SUBS - covered)}"


def test_real_has_boundary_cases():
    nb = sum(bool(it["boundary"]) for it in REAL["items"])
    assert nb >= 8, f"слишком мало boundary-кейсов: {nb}"


def test_real_label_grade_is_silver_with_annotator():
    """Метки честно помечены как silver (не gold) и назван аннотатор — методологический
    контракт: аннотатор ≠ тестируемый классификатор."""
    assert REAL.get("label_grade") == "silver"
    assert "opus" in REAL.get("annotator", "").lower()


def test_real_loads_via_harness():
    """Реальный gold-set читается тем же load_gold, что и прод-прогон."""
    items = E.load_gold(E.TOOLS_DIR / "data" / "l0_gold_set_real.json")
    assert len(items) == 93
    assert E.build_prompt(items[0])  # промпт строится без ошибок


# --- Витрина реального прогона (гейтим, чтобы числа нельзя было тихо занизить) ---


def _real_showcase():
    return json.loads(
        (E.TOOLS_DIR / "data" / "l0_eval_real_results.json").read_text(encoding="utf-8")
    )


# Пороги на РЕАЛЬНОМ наборе ниже, чем на синтетике: реальные новости объективно труднее
# (sub 86% против 96%). Гейт защищает витрину от тихой деградации.
MIN_SUB_REAL = 0.80
MIN_SUB_CI_LOWER_REAL = 0.72
MIN_MAIN_REAL = 0.92


def test_real_showcase_matches_gold():
    res = _real_showcase()
    assert res["gold_set_n"] == 93
    assert res["gold_set"].endswith("l0_gold_set_real.json")
    assert res["label_grade"] == "silver"
    assert any(m["alias"] == "haiku" for m in res["models"])


def test_real_showcase_meets_thresholds():
    haiku = next(m for m in _real_showcase()["models"] if m["alias"] == "haiku")
    assert haiku["n"] == 93
    assert haiku["subcategory_accuracy"] >= MIN_SUB_REAL, "sub-accuracy на реальном наборе упала"
    assert haiku["subcategory_accuracy_ci95"][0] >= MIN_SUB_CI_LOWER_REAL, "нижняя граница CI упала"
    assert haiku["main_category_accuracy"] >= MIN_MAIN_REAL, "main-accuracy упала"


def test_real_showcase_misses_explained():
    """Каждый sub-промах реального прогона разобран (что модель, что спорная метка) —
    ровно столько записей, сколько промахов, и каждая с непустым note и верным эталоном."""
    gold_by_id = {it["id"]: it for it in REAL["items"]}
    haiku = next(m for m in _real_showcase()["models"] if m["alias"] == "haiku")
    n_wrong = haiku["n"] - int(haiku["subcategory_correct"].split("/")[0])
    assert len(haiku["misses"]) == n_wrong, f"промахов {n_wrong}, разобрано {len(haiku['misses'])}"
    for miss in haiku["misses"]:
        assert miss["id"] in gold_by_id, f"промах по неизвестному id {miss['id']}"
        assert (
            gold_by_id[miss["id"]]["gold_sub"] == miss["gold"]
        ), f"{miss['id']}: эталон расходится"
        assert miss.get("note", "").strip(), f"{miss['id']}: промах без разбора"


# --- Held-out набор (валидация улучшений промпта; должен быть disjoint с dev) ---

HELDOUT = json.loads(
    (E.TOOLS_DIR / "data" / "l0_gold_set_heldout.json").read_text(encoding="utf-8")
)


def test_heldout_schema_and_provenance():
    items = HELDOUT["items"]
    assert len(items) == HELDOUT["n"] >= 29
    for it in items:
        for f in ("id", "gold_main", "gold_sub", "boundary", "source", "date", "url", "news"):
            assert f in it, f"{it.get('id')}: нет поля {f}"
        assert it["url"].startswith("http"), f"{it['id']}: url без провенанса"
        assert it["gold_sub"] in VALID_SUBS, f"{it['id']}: невалидная подкат"
        assert it["gold_main"] == it["gold_sub"].split(".")[0], f"{it['id']}: main≠префикс"
        assert len(it["news"]) > 30, f"{it['id']}: короткий news"


def test_heldout_disjoint_from_dev_by_url():
    """Held-out не должен пересекаться с dev по URL — иначе это не held-out."""
    dev_urls = {it["url"] for it in REAL["items"]}
    ho_urls = {it["url"] for it in HELDOUT["items"]}
    overlap = dev_urls & ho_urls
    assert not overlap, f"held-out пересекается с dev по URL: {overlap}"


def test_heldout_unique_ids_and_silver():
    ids = [it["id"] for it in HELDOUT["items"]]
    assert len(set(ids)) == len(ids), "дубли id в held-out"
    assert HELDOUT.get("label_grade") == "silver"
    assert "opus" in HELDOUT.get("annotator", "").lower()
