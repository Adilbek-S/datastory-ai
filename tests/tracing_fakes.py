"""Проверка трассировки без сети: клиент LangSmith, который запоминает run-ы в памяти, и «модель» с настоящим вызовом LangChain.

Это проверка того, что код ОТПРАВЛЯЕТ нужные span-ы (имена, вложенность, метаданные). Что трассы доходят до облака LangSmith,
она не доказывает: для этого нужен ключ (см. LANGSMITH_SETUP.md).
"""
from langchain_core.runnables import RunnableLambda
from langsmith import Client


class RecordingClient(Client):
    def __init__(self):
        super().__init__(api_key="test-key", api_url="http://127.0.0.1:9", auto_batch_tracing=False)
        self.records: dict[str, dict] = {}

    def create_run(self, name, inputs, run_type, **kw):
        extra = kw.get("extra") or {}
        self.records[str(kw["id"])] = {
            "id": str(kw["id"]), "name": name, "run_type": run_type, "parent": str(kw["parent_run_id"]) if kw.get("parent_run_id") else None,
            "inputs": inputs, "outputs": None, "error": None, "tags": list(kw.get("tags") or []), "metadata": dict(extra.get("metadata") or {}),
        }

    def update_run(self, run_id, **kw):
        run = self.records.get(str(run_id))
        if run is None:
            return
        run["outputs"] = kw.get("outputs") or run["outputs"]
        run["error"] = kw.get("error") or run["error"]
        run["metadata"].update((kw.get("extra") or {}).get("metadata") or {})
        run["tags"] = list(dict.fromkeys([*run["tags"], *(kw.get("tags") or [])]))

    # ------------------------------------------------------------------ удобные запросы
    def named(self, name: str) -> list[dict]:
        return [r for r in self.records.values() if r["name"] == name]

    def names(self) -> set[str]:
        return {r["name"] for r in self.records.values()}

    def ancestors(self, run: dict) -> list[str]:
        found, parent = [], run["parent"]
        while parent:
            found.append(self.records[parent]["name"])
            parent = self.records[parent]["parent"]
        return found


class FakeChat:
    """Вместо ChatOpenAI: with_structured_output возвращает настоящую цепочку LangChain, поэтому run-ы создаются как в бою."""

    def __init__(self, responders):
        self.responders = responders  # имя схемы -> функция (system, user, n) -> модель
        self.counts: dict[str, int] = {}

    def with_structured_output(self, schema):
        def answer(messages):
            n = self.counts[schema.__name__] = self.counts.get(schema.__name__, 0) + 1
            return self.responders[schema.__name__](messages[0].content, messages[1].content, n)

        return RunnableLambda(answer)
