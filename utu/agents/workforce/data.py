from dataclasses import dataclass, field
from typing import Literal

from ..common import TaskRecorder


@dataclass
class Subtask:
    task_id: int
    task_name: str
    task_description: str = None
    task_status: Literal["not started", "in progress", "completed", "success", "failed", "partial success"] = (
        "not started"
    )
    task_result: str = None
    task_result_detailed: str = None
    task_mode: str = None # ["ASSIGN_AGENT", "DIRECT_ANSWER"]
    assigned_agent: str = None
    task_direct_answer: str = None  # for DIRECT_ANSWER mode

    @property
    def formatted_with_result(self) -> str:
        infos = [
            f"<task_id:{self.task_id}>{self.task_name}</task_id:{self.task_id}>",
            f"<task_status>{self.task_status}</task_status>",
        ]
        if self.task_result is not None:
            infos.append(f"<task_result>{self.task_result}</task_result>")
        return "\n".join(infos)


@dataclass
class WorkforceTaskRecorder(TaskRecorder):
    overall_task: str = ""
    executor_agent_kwargs_list: list[dict] = field(default_factory=list)
    task_plan: list[Subtask] = field(default_factory=list)
    failure_info: str = ""
    experience_from_failure: str = ""

    tentative_answer: str = ""
    tentative_answer_confidence: str = ""
    tentative_answer_uniqueness_assessment: str = ""

    @property
    def executor_agents_info(self) -> str:
        """Get the executor agents info."""
        tools_str = []
        for agent_kwargs in self.executor_agent_kwargs_list:
            tools_str.append(
                f"- {agent_kwargs['name']}: {agent_kwargs['description']}\n"
                f"  Available tools: {', '.join(agent_kwargs['toolnames'])}"
            )
        return "\n".join(tools_str)

    @property
    def executor_agents_names(self) -> str:
        return str([agent_kwargs["name"] for agent_kwargs in self.executor_agent_kwargs_list])

    # -----------------------------------------------------------
    @property
    def formatted_task_plan_list_with_task_results(self) -> list[str]:
        """Format the task plan for display."""
        return [task.formatted_with_result for task in self.task_plan]

    @property
    def formatted_task_plan(self) -> str:
        """Format the task plan for display."""
        formatted_plan_list = []
        for task in self.task_plan:
            formatted_plan_list.append(f"{task.task_id}. {task.task_name} - Status: {task.task_status}")
        return "\n".join(formatted_plan_list)

    # -----------------------------------------------------------
    def plan_init(self, plan_list: list[Subtask]) -> None:
        self.task_plan = plan_list

    def plan_update(self, task: Subtask, updated_plan: list[str]) -> None:
        finished_tasks = self.task_plan[: task.task_id]
        new_tasks = [Subtask(task_id=task.task_id + i, task_name=t) for i, t in enumerate(updated_plan, 1)]
        self.task_plan = finished_tasks + new_tasks

    # -----------------------------------------------------------
    def get_next_task(self) -> Subtask | None:
        if self.task_plan is None:
            return None
        for task in self.task_plan:
            if task.task_status == "not started":
                return task
        return None

    @property
    def has_failed_task(self) -> bool:
        if self.task_plan is None:
            return False
        return any(task.task_status == "failed" for task in self.task_plan)

    # -----------------------------------------------------------
    def update_failure_info(self, failure_info: str) -> None:
        self.failure_info = failure_info
    
    def update_experience_from_failure(self, experience_from_failure: str) -> None:
        self.experience_from_failure = experience_from_failure

    def update_tentative_answer(self, answer: str, confidence: str, uniqueness_assessment: str) -> None:
        self.tentative_answer = answer
        self.tentative_answer_confidence = confidence
        self.tentative_answer_uniqueness_assessment = uniqueness_assessment

    def check_tentative_answer_quality(self) -> tuple[bool, str]:
        answer_quality_acceptable = (
            self.tentative_answer_confidence in ["high", "medium"]
            and self.tentative_answer_uniqueness_assessment != "non-unique"
        )
        failure_reasons = []
        if not answer_quality_acceptable:
            if self.tentative_answer_confidence == "low":
                failure_reasons.append("answer confidence too low")
            if self.tentative_answer_uniqueness_assessment in ["unclear", "non-unique"]:
                failure_reasons.append("answer uniqueness insufficient")
        failure_reason = " and ".join(failure_reasons)
        return answer_quality_acceptable, failure_reason
