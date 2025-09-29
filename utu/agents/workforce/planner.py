"""
- [ ] standardize parser?
"""

import re

from ...config import AgentConfig
from ...utils import FileUtils, get_logger
from ..llm_agent import LLMAgent
from .data import Subtask, WorkforceTaskRecorder

logger = get_logger(__name__)
PROMPTS = FileUtils.load_prompts("agents/workforce/planner.yaml")


class PlannerAgent:
    """Task planner that handles task decomposition."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.llm = LLMAgent(config.workforce_planner_model)
        self.llm.set_instructions(PROMPTS["TASK_PLAN_SYS_PROMPT"])
        self._modify_plan_budgets = config.workforce_config.get("planner_modify_budgets", 3)

    async def plan_task(self, recorder: WorkforceTaskRecorder) -> None:
        """Plan tasks based on the overall task and available agents."""
        if recorder.failure_info is None:
            plan_prompt = PROMPTS["TASK_PLAN_PROMPT"].format(
                overall_task=recorder.overall_task,
                executor_agents_info=recorder.executor_agents_info,
            )
        else:
            plan_prompt = PROMPTS["TASK_REPLAN_PROMPT"].format(
                overall_task=recorder.overall_task,
                executor_agents_info=recorder.executor_agents_info,
                failure_info=recorder.failure_info,
            )

        plan_recorder = await self.llm.run(plan_prompt)
        recorder.add_run_result(plan_recorder.get_run_result(), "planner")  # add planner trajectory

        # parse tasks
        pattern = r"<task>(.*?)</task>"
        tasks_content: list[str] = re.findall(pattern, plan_recorder.final_output, re.DOTALL)
        tasks_content = [task.strip() for task in tasks_content if task.strip()]
        tasks = [Subtask(task_id=i + 1, task_name=task) for i, task in enumerate(tasks_content)]
        recorder.plan_init(tasks)

        # parse experience from failure info
        if recorder.failure_info is not None:
            pattern = r"<helpful_experience_or_fact>(.*?)</helpful_experience_or_fact>"
            exp_from_failure_match = re.search(pattern, plan_recorder.final_output, re.DOTALL)
            exp_from_failure_content: str | None = (
                exp_from_failure_match.group(1).strip()
                if exp_from_failure_match
                else None
            )
            if exp_from_failure_content:
                recorder.update_experience_from_failure(exp_from_failure_content)

    async def plan_update(self, recorder: WorkforceTaskRecorder, task: Subtask) -> str:
        """Update the task plan based on completed tasks."""
        # check budgets
        if self._modify_plan_budgets <= 0:
            logger.warning("Modify plan budgets exhausted. Continuing with existing plan.")
            return "continue"

        task_plan_list = recorder.formatted_task_plan_list_with_task_results
        last_task_id = task.task_id
        previous_task_plan = "\n".join(f"{task}" for task in task_plan_list[: last_task_id + 1])
        unfinished_task_plan = "\n".join(f"{task}" for task in task_plan_list[last_task_id + 1 :])

        task_update_plan_prompt = (
            PROMPTS["TASK_UPDATE_PLAN_PROMPT"]
            .strip()
            .format(
                overall_task=recorder.overall_task,
                previous_task_plan=previous_task_plan,
                unfinished_task_plan=unfinished_task_plan,
            )
        )
        plan_update_recorder = await self.llm.run(task_update_plan_prompt)
        recorder.add_run_result(plan_update_recorder.get_run_result(), "planner")  # add planner trajectory
        choice, updated_plan = self._parse_update_response(plan_update_recorder.final_output)
        # choice: continue, update, early_completion
        if choice == "continue":
            pass # do nothing here
        elif choice == "update":
            self._modify_plan_budgets -= 1
            if updated_plan is not None and len(updated_plan) > 0:
                recorder.plan_update(task, updated_plan)
            else:
                choice = "continue"  # fallback to continue if no valid updated plan
                updated_plan = None
        elif choice == "early_completion":
            pass # do nothing here
        else:
            raise ValueError(f"Unexpected choice value for plan update: {choice}")

        return choice

    def _parse_update_response(self, response: str) -> tuple[str, list[str] | None]:
        # Parse choice
        pattern_choice = r"<choice>(.*?)</choice>"
        match_choice = re.search(pattern_choice, response, re.DOTALL)
        if match_choice:
            choice = match_choice.group(1).strip().lower()
        else:
            logger.warning("No choice found in response. Defaulting to 'continue'.")
            choice = "continue"

        # Parse updated plan if choice is "update"
        updated_tasks = None
        if choice == "update":
            pattern_updated_plan = r"<updated_unfinished_task_plan>(.*?)</updated_unfinished_task_plan>"
            match_updated_plan = re.search(pattern_updated_plan, response, re.DOTALL)
            if match_updated_plan:
                # Match the task content
                updated_plan_content = match_updated_plan.group(1).strip()
                combined_pattern = r"<task(?:_id:\d+[^>]*)?>([^<]*?)</task(?:_id:\d+[^>]*)?"
                task_matches = re.findall(combined_pattern, updated_plan_content, re.DOTALL)

                updated_tasks = [task.strip() for task in task_matches if task.strip()]
                if not updated_tasks:
                    logger.warning("No tasks found in updated plan. Defaulting to None.")
                    updated_tasks = None
            else:
                logger.warning("No updated plan found in response. Defaulting to None.")
                updated_tasks = None

        return choice, updated_tasks

    async def plan_check(self, recorder: WorkforceTaskRecorder, task: Subtask) -> None:
        task_check_prompt = (
            PROMPTS["TASK_CHECK_PROMPT"]
            .strip()
            .format(
                overall_task=recorder.overall_task,
                task_plan=recorder.formatted_task_plan,
                last_completed_task=task.task_name,
                last_completed_task_id=task.task_id,
                last_completed_task_description=task.task_description,
                last_completed_task_result=task.task_result,
            )
        )
        res = await self.llm.run(task_check_prompt)
        recorder.add_run_result(res.get_run_result(), "planner")  # add planner trajectory
        # parse and update task status
        task_check_result = self._parse_check_response(res.final_output)
        task.task_status = task_check_result

    def _parse_check_response(self, response: str) -> str:
        pattern = r"<task_status>(.*?)</task_status>"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            task_status = match.group(1).strip().lower()
            if "partial" in task_status:  # in case that models output "partial_success"
                return "partial success"
            if task_status in ["success", "failed", "partial success"]:
                return task_status
            else:
                logger.warning(f"Unexpected task status value: {task_status}. Defaulting to 'partial success'.")
                return "partial success"
        else:
            logger.warning("No task status found in response. Defaulting to 'partial success'.")
            return "partial success"

    async def reflect_on_failure(
        self, recorder: WorkforceTaskRecorder, additional_context: str = ""
    ) -> None:
        """Reflect on the failure of the overall task and provide analysis."""
        reflection_prompt = (
            PROMPTS["TASK_REFLECTION_PROMPT"]
            .strip()
            .format(
                question=recorder.overall_task,
                task_results="\n\n".join(
                    recorder.formatted_task_plan_list_with_task_results
                ),
                additional_context=(
                    additional_context
                    if additional_context == ""
                    else f"\n\n{additional_context}\n\n"
                ),
            )
        )
        reflection_recorder = await self.llm.run(reflection_prompt)
        recorder.add_run_result(
            reflection_recorder.get_run_result(), "planner_reflect_on_failure"
        )  # add trajectory
        reflection_result = reflection_recorder.final_output
        if additional_context:
            reflection_result += f"\n\n{additional_context}"
        recorder.update_failure_info(reflection_result)
