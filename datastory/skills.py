"""Загрузка Skill: файл .claude/skills/<name>/SKILL.md с YAML frontmatter и телом в Markdown.

Методика хранится только в SKILL.md. Код не содержит копий её текста: этапы LangGraph получают инструкции из этого
файла через load_skill() и Skill.for_stage(). Правки SKILL.md сразу меняют поведение LLM-этапов.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from datastory.config import Settings, get_settings

SKILL_FILE = "SKILL.md"
HEADING = re.compile(r"^(#{1,3})\s+(.+?)\s*$")
RULE_ID = re.compile(r"\*\*([A-Z]\d+)\.\*\*")


class SkillError(RuntimeError):
    """Skill не найден или оформлен неверно. Сообщение написано для пользователя."""

    def __init__(self, user_message: str):
        super().__init__(user_message)
        self.user_message = user_message


@dataclass(frozen=True)
class Section:
    level: int
    title: str
    text: str  # строки под заголовком до следующего заголовка

    def render(self) -> str:
        return f"{'#' * self.level} {self.title}\n{self.text}".rstrip()


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    body: str
    path: Path
    sections: tuple[Section, ...]

    @property
    def rule_ids(self) -> list[str]:
        """Идентификаторы правил (V1, A3…), объявленные в теле Skill."""
        return RULE_ID.findall(self.body)

    def select(self, *patterns: str) -> str:
        """Разделы, заголовки которых подходят под регулярные выражения, в порядке их следования в Skill."""
        compiled = [re.compile(p) for p in patterns]
        chosen = [s for s in self.sections if any(c.search(s.title) for c in compiled)]
        if not chosen:
            raise SkillError(f"В Skill «{self.name}» нет разделов, подходящих под {patterns}: проверьте {self.path}.")
        return "\n\n".join(s.render() for s in chosen)


def parse_skill(text: str, path: Path) -> Skill:
    match = re.match(r"\A---\s*\n(.*?)\n---\s*\n(.*)\Z", text.lstrip("﻿"), re.DOTALL)
    if not match:
        raise SkillError(f"Файл {path} должен начинаться с YAML frontmatter между строками «---».")
    try:
        meta = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError as exc:
        raise SkillError(f"YAML frontmatter в {path} не читается: {exc}") from None
    name, description = meta.get("name"), meta.get("description")
    if not isinstance(name, str) or not isinstance(description, str) or not name.strip() or not description.strip():
        raise SkillError(f"В frontmatter файла {path} должны быть непустые поля name и description.")

    sections: list[Section] = []
    level, title, lines = 0, "", []
    for line in match.group(2).splitlines():
        heading = HEADING.match(line)
        if heading:
            if title:
                sections.append(Section(level, title, "\n".join(lines).strip("\n")))
            level, title, lines = len(heading.group(1)), heading.group(2), []
        else:
            lines.append(line)
    if title:
        sections.append(Section(level, title, "\n".join(lines).strip("\n")))
    return Skill(name=name.strip(), description=description.strip(), body=match.group(2).strip(), path=path, sections=tuple(sections))


def load_skill(name: str, root: Path | None = None, settings: Settings | None = None) -> Skill:
    """Читает Skill с диска при каждом обращении (файл небольшой), поэтому правки видны без перезапуска."""
    root = root or (settings or get_settings()).skills_dir
    path = root / name / SKILL_FILE
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        raise SkillError(f"Skill «{name}» не найден: ожидается файл {path}.") from None
    skill = parse_skill(text, path)
    if skill.name != name:
        raise SkillError(f"Имя в frontmatter ({skill.name!r}) не совпадает с каталогом Skill ({name!r}): {path}.")
    return skill
