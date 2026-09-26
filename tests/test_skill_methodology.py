"""Skill datastory-analysis: оформление, связь с этапами LangGraph и то, что методика реально применяется."""
import re
import shutil
from pathlib import Path

import pytest

from datastory.errors import LLMError
from datastory.insights import verifier
from datastory.insights.verifier import METHODOLOGY_RULE, NUMBERS_GROUNDED, Violation
from datastory.skills import Skill, SkillError, load_skill, parse_skill
from datastory.workflow.graph import WorkflowDeps
from datastory.workflow.methodology import ENFORCEMENT, SKILL_NAME, STAGE_SECTIONS, methodology_prompt
from datastory.workflow.runner import AnalysisRunner
from tests.helpers import ScriptedLLM
from tests.workflow_helpers import approve, good_insight, llm_for, make_runner

ROOT = Path(__file__).resolve().parent.parent
SKILL_DIR = ROOT / ".claude" / "skills" / SKILL_NAME
SKILL_PATH = SKILL_DIR / "SKILL.md"
STEP_TOPICS = [
    "Проверка структуры данных", "Определение показателей и измерений", "Проверка наличия бизнес-определений",
    "Выбор подходящего типа визуализации", "Выполнение точных расчётов через MCP", "Выявление изменений показателей",
    "Формирование выводов на основе числовых доказательств", "Проверка достоверности выводов",
]


@pytest.fixture(scope="module")
def skill() -> Skill:
    return load_skill(SKILL_NAME)


def rule_texts(skill: Skill) -> dict[str, str]:
    """id правила -> его текст из SKILL.md."""
    return {m[1]: m[2].strip() for m in re.finditer(r"^- \*\*([A-Z]\d+)\.\*\* (.+)$", skill.body, re.MULTILINE)}


# ================================================================== оформление Skill
def test_skill_is_stored_at_the_standard_location_with_yaml_frontmatter():
    assert SKILL_PATH.is_file()
    text = SKILL_PATH.read_text(encoding="utf-8")
    assert text.startswith("---\nname: datastory-analysis\ndescription: ")
    assert text.count("\n---\n") >= 1


def test_frontmatter_has_name_and_description_with_conditions_of_use(skill):
    assert skill.name == "datastory-analysis" == SKILL_DIR.name
    assert len(skill.description) > 80
    assert "Применяй" in skill.description and "Не применяй" in skill.description  # условия применения в описании


def test_body_has_all_required_sections(skill):
    titles = [s.title for s in skill.sections]
    for required in ("Когда применять", "Последовательность анализа", "Правила выбора визуализации", "Правила анализа и формирования выводов"):
        assert required in titles
    assert "Не применяй методику" in skill.body


def test_analysis_sequence_has_the_eight_steps_in_order(skill):
    steps = [s.title for s in skill.sections if s.title.startswith("Шаг ")]
    assert [re.sub(r"^Шаг \d\. ", "", t) for t in steps] == STEP_TOPICS
    assert [t[:6] for t in steps] == [f"Шаг {i}." for i in range(1, 9)]


def test_visualization_rules_cover_the_required_chart_choices(skill):
    section = skill.select(r"^Правила выбора визуализации").lower()
    for phrase in ("line chart", "bar chart", "pie chart", "временных рядов", "разными единицами измерения", "без явного обоснования"):
        assert phrase in section


def test_analysis_rules_cover_the_required_prohibitions(skill):
    section = skill.select(r"^Правила анализа").lower()
    for phrase in (
        "не придумывай значения", "не придумывай определения", "причинную связь без подтверждения", "без сравнения чисел",
        "арифметические расчёты", "отсутствующие в исходных данных", "расчёт, факт из документа и предположение",
    ):
        assert phrase in section


# ================================================================== разбор и загрузка
def test_missing_or_broken_skill_files_are_reported(tmp_path):
    with pytest.raises(SkillError, match="не найден"):
        load_skill("datastory-analysis", root=tmp_path)
    with pytest.raises(SkillError, match="frontmatter"):
        parse_skill("# без frontmatter", tmp_path / "SKILL.md")
    with pytest.raises(SkillError, match="name и description"):
        parse_skill("---\nname: x\n---\nтекст", tmp_path / "SKILL.md")
    other = tmp_path / "datastory-analysis"
    other.mkdir()
    (other / "SKILL.md").write_text("---\nname: другое\ndescription: описание\n---\nтекст", encoding="utf-8")
    with pytest.raises(SkillError, match="не совпадает"):
        load_skill("datastory-analysis", root=tmp_path)


def test_skill_directory_can_be_configured(tmp_path, monkeypatch):
    shutil.copytree(SKILL_DIR, tmp_path / SKILL_NAME)
    monkeypatch.setenv("SKILLS_DIR", str(tmp_path))
    from datastory.config import get_settings

    get_settings.cache_clear()
    assert load_skill(SKILL_NAME).path == tmp_path / SKILL_NAME / "SKILL.md"


# ================================================================== методика и код не расходятся
def test_every_rule_of_the_skill_is_enforced_somewhere_in_code_and_vice_versa(skill):
    assert set(skill.rule_ids) == set(ENFORCEMENT), "правила в SKILL.md и таблица проверок в коде должны совпадать"
    assert len(skill.rule_ids) == len(set(skill.rule_ids))  # идентификаторы не повторяются
    assert all(place.strip() for place in ENFORCEMENT.values())


def test_every_verifier_check_points_to_a_rule_of_the_skill(skill):
    codes = {getattr(verifier, n) for n in dir(verifier) if n.isupper() and isinstance(getattr(verifier, n), str) and "-" in getattr(verifier, n)}
    assert codes == set(METHODOLOGY_RULE)
    assert set(METHODOLOGY_RULE.values()) <= set(skill.rule_ids)
    assert "(правило методики A1)" in str(Violation(NUMBERS_GROUNDED, "тест"))


# ================================================================== этапы получают разные части Skill
def test_each_llm_stage_gets_only_its_own_sections(skill):
    plan, insights = methodology_prompt(skill, "plan"), methodology_prompt(skill, "insights")
    for step in ("Шаг 1.", "Шаг 2.", "Шаг 3.", "Шаг 4."):
        assert step in plan and step not in insights
    for step in ("Шаг 5.", "Шаг 6.", "Шаг 7.", "Шаг 8."):
        assert step in insights and step not in plan
    assert "**V4.**" in plan and "**V4.**" not in insights  # правила визуализации нужны плану, а не выводам
    for prompt in (plan, insights):
        assert "**A3.**" in prompt and SKILL_NAME in prompt  # правила анализа — на обоих этапах
    chat = methodology_prompt(skill, "chat")  # ответ на вопрос: определения, расчёты, изменения, выводы, проверка
    for step in ("Шаг 3.", "Шаг 5.", "Шаг 6.", "Шаг 7.", "Шаг 8."):
        assert step in chat
    for step in ("Шаг 1.", "Шаг 2.", "Шаг 4."):
        assert step not in chat
    assert "**V4.**" not in chat and "**A7.**" in chat
    assert set(STAGE_SECTIONS) == {"plan", "insights", "chat"}


# ================================================================== методика реально используется воркфлоу
def run_with_llm(client, datasets, llm, **kwargs):
    runner = make_runner(client, llm, **kwargs)
    snapshot = runner.start(datasets["demo"])
    return runner, snapshot, runner.resume(snapshot.thread_id, approve(snapshot))


def test_plan_and_insight_stages_send_the_skill_text_to_the_llm(client, datasets, skill):
    llm = llm_for()
    run_with_llm(client, datasets, llm)
    rules = rule_texts(skill)

    plan_system = llm.prompts("PlanDraft")[0][0]
    assert methodology_prompt(skill, "plan") in plan_system
    assert rules["V4"] in plan_system and rules["V3"] in plan_system and rules["A2"] in plan_system

    insight_systems = {system for system, _ in llm.prompts("InsightDraft")}
    assert len(insight_systems) == 1 and len(llm.prompts("InsightDraft")) == 4  # один и тот же текст методики на каждый график
    insight_system = insight_systems.pop()
    assert methodology_prompt(skill, "insights") in insight_system
    for rule in ("A1", "A3", "A4", "A5", "A6", "A7"):
        assert rules[rule] in insight_system
    assert rules["V4"] not in insight_system

    intent_llm = llm_for()
    make_runner(client, intent_llm).start(datasets["demo"], "покажи успешность")
    assert SKILL_NAME not in intent_llm.prompts("IntentDraft")[0][0]  # разбор запроса — не аналитический этап


def test_editing_the_skill_changes_what_the_llm_receives(client, datasets, tmp_path):
    """Текст методики не зашит в код: правка SKILL.md сразу видна в промптах обоих этапов."""
    shutil.copytree(SKILL_DIR, tmp_path / SKILL_NAME)
    path = tmp_path / SKILL_NAME / "SKILL.md"
    path.write_text(path.read_text(encoding="utf-8") + "\n- **A8.** МАРКЕР-НОВОГО-ПРАВИЛА-731 всегда указывай единицы измерения.\n", encoding="utf-8")
    llm = llm_for()
    run_with_llm(client, datasets, llm, skill=load_skill(SKILL_NAME, root=tmp_path))
    assert "МАРКЕР-НОВОГО-ПРАВИЛА-731" in llm.prompts("PlanDraft")[0][0]
    assert all("МАРКЕР-НОВОГО-ПРАВИЛА-731" in system for system, _ in llm.prompts("InsightDraft"))


def test_skill_text_is_not_copied_into_python_sources(skill):
    sources = "\n".join(p.read_text(encoding="utf-8") for p in (ROOT / "datastory").rglob("*.py"))
    for rule_id, text in rule_texts(skill).items():
        assert text[:40] not in sources, f"текст правила {rule_id} продублирован в коде"
    for section in skill.sections:
        first_line = next((ln for ln in section.text.splitlines() if len(ln) > 60), None)
        if first_line:
            assert first_line[:60] not in sources


def test_result_records_which_stages_used_the_methodology(client, datasets, skill):
    _, _, done = run_with_llm(client, datasets, llm_for())
    info = done.result.methodology
    assert info.skill == SKILL_NAME and info.stages == ["plan", "insights"] and info.rules == skill.rule_ids


def test_without_an_llm_the_rules_are_applied_by_code_and_the_skill_is_not_sent(client, datasets):
    _, _, done = run_with_llm(client, datasets, None)
    assert done.result.methodology is None
    assert all(i.generated_by == "rules" for i in done.result.insights)


def test_llm_is_not_called_without_the_methodology(client, datasets):
    def unavailable():
        raise SkillError("Skill «datastory-analysis» не найден: ожидается файл x/SKILL.md.")

    llm = llm_for()
    runner = AnalysisRunner(WorkflowDeps(client=lambda: client, llm=lambda: llm, kb=lambda: None, skill=unavailable))
    snapshot = runner.start(datasets["demo"])
    done = runner.resume(snapshot.thread_id, approve(snapshot))
    assert llm.prompts("PlanDraft") == [] and llm.prompts("InsightDraft") == []  # модель без методики не вызывается
    assert done.result.methodology is None and all(i.generated_by == "rules" for i in done.result.insights)
    assert any("Методика не загружена" in w and "план анализа" in w for w in snapshot.warnings)
    assert any("аналитические выводы" in w for w in done.result.warnings)


# ================================================================== правила методики действуют на результат
def test_visualization_rules_v1_to_v4_are_applied_to_llm_plans(client, datasets, skill):
    from datastory.workflow.models import ChartDraft, PlanDraft

    def plan(system, user, n):
        return PlanDraft(
            goal="x",
            charts=[
                ChartDraft(analysis="success_dynamics", title="Успешность", chart_type="pie", x_column="Month", rationale="?"),
                ChartDraft(analysis="channel_distribution", title="Каналы", chart_type="line", x_column="Channel", rationale="?"),
            ],
        )

    snapshot = make_runner(client, llm_for(plan=plan)).start(datasets["demo"])
    kinds = {c.analysis: c.chart_type for c in snapshot.request.plan.charts}
    assert kinds["success_dynamics"] == "line"  # V1, V4
    assert kinds["channel_distribution"] == "pie"  # V3: три категории
    assert len(snapshot.request.plan.charts) <= 4  # V6


def test_analysis_rules_are_enforced_when_the_llm_ignores_the_methodology(client, datasets):
    """A1 и A3: даже если модель нарушит правила из промпта, вывод не будет опубликован."""

    def defiant(system, user, n):
        draft = good_insight(system, user, n)
        return draft.model_copy(update={"summary": draft.summary + " Успешность выросла до 99,9% из-за обновления."})

    _, _, done = run_with_llm(client, datasets, llm_for(insight=defiant))
    assert all(c.fallback for c in done.result.insight_checks)
    joined = " ".join(v for c in done.result.insight_checks for v in c.violations)
    assert "правило методики A1" in joined and "правило методики A3" in joined
    assert all("из-за обновления" not in i.text and "99,9" not in i.text for i in done.result.insights)


def test_llm_failure_does_not_skip_the_methodology_rules(client, datasets):
    failing = ScriptedLLM(PlanDraft=lambda s, u, n: LLMError("сбой"), InsightDraft=lambda s, u, n: LLMError("сбой"))
    _, _, done = run_with_llm(client, datasets, failing)
    assert done.result.insights and all("Причина изменения по данным не установлена" in (i.limitation or "") or i.chart_id == "channel_distribution" for i in done.result.insights)
