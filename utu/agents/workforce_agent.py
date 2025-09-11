"""
- [x] setup tracing
- [x] purify logging
- [ ] support stream?
- [x] support self-reflection (planner_max_reflection)
"""

from agents import trace

from ..config import AgentConfig, ConfigLoader
from ..utils import AgentsUtils, FileUtils, get_logger
from .base_agent import BaseAgent
from .workforce import AnswererAgent, AssignerAgent, ExecutorAgent, PlannerAgent, WorkspaceTaskRecorder

logger = get_logger(__name__)
PLANNER_PROMPTS = FileUtils.load_prompts("agents/workforce/planner.yaml")


class WorkforceAgent(BaseAgent):
    name = "workforce_agent"

    def __init__(self, config: AgentConfig | str):
        """Initialize the workforce agent"""
        if isinstance(config, str):
            config = ConfigLoader.load_agent_config(config)
        self.config = config
        logger.info(f"Workforce agent config: {self.config.workforce_config}")
        self.planner_max_reflection = config.workforce_config.get("planner_max_reflection", 1)

    async def run(self, input: str, trace_id: str = None) -> WorkspaceTaskRecorder:
        trace_id = trace_id or AgentsUtils.gen_trace_id()

        logger.info("Initializing agents...")
        planner_agent = PlannerAgent(config=self.config)
        assigner_agent = AssignerAgent(config=self.config)
        answerer_agent = AnswererAgent(config=self.config)
        executor_agent_group: dict[str, ExecutorAgent] = {}
        for name, config in self.config.workforce_executor_agents.items():
            executor_agent_group[name] = ExecutorAgent(config=config, workforce_config=self.config)

        recorder = WorkspaceTaskRecorder(
            overall_task=input, executor_agent_kwargs_list=self.config.workforce_executor_infos
        )

        with trace(workflow_name=self.name, trace_id=trace_id):
            for i in range(self.planner_max_reflection):
                # * 1. generate plan
                logger.info("Generating plan...")
                await planner_agent.plan_task(recorder)
                logger.info(f"Plan: {recorder.task_plan}")

                # DISCUSS: merge .get_next_task and .has_uncompleted_tasks? (while True)
                while recorder.has_uncompleted_tasks:
                    # * 2. assign tasks
                    next_task = await assigner_agent.assign_task(recorder)
                    logger.info(f"Assign task: {next_task.task_id} assigned to {next_task.assigned_agent}")

                    # * 3. execute task
                    logger.info(f"Executing task: {next_task.task_id}")
                    await executor_agent_group[next_task.assigned_agent].execute_task(recorder=recorder, task=next_task)
                    logger.info(f"Task {next_task.task_id} result: {next_task.task_result}")
                    await planner_agent.plan_check(recorder, next_task)
                    logger.info(f"Task {next_task.task_id} checked: {next_task.task_status}")

                    # * 4. update plan
                    if not recorder.has_uncompleted_tasks:  # early stop
                        break
                    plan_update_choice = await planner_agent.plan_update(recorder, next_task)
                    logger.info(f"Plan update choice: {plan_update_choice}")
                    if plan_update_choice == "stop":
                        logger.info("Planner determined overall task is complete, stopping execution")
                        break
                    elif plan_update_choice == "update":
                        logger.info(f"Task plan updated: {recorder.task_plan}")

                # * 5. self-reflection
                logger.info(f"Starting reflection process, {i + 1} / {self.planner_max_reflection}")

                # Check 0: failed tasks
                if recorder.has_failed_task:
                    await planner_agent.reflect_on_failure(recorder)
                    continue

                # Check 1: answer quality
                await answerer_agent.extract_final_answer(recorder)  # NOTE: the recorder.tentative_answer is updated!
                answer_quality_acceptable, failure_reason = recorder.check_tentative_answer_quality()
                if not answer_quality_acceptable:
                    additional_context = PLANNER_PROMPTS["REFLECTION_FAILURE_PROMPT_1"].format(
                        failure_reason=failure_reason,
                    )
                    await planner_agent.reflect_on_failure(recorder, additional_context)
                    continue

                # Check 2: self-check
                self_check_passed = await answerer_agent.answer_self_check(recorder)
                if not self_check_passed:
                    logger.warning(f"Task {next_task.task_id} self-check failed, reflecting and replanning!")
                    additional_context = PLANNER_PROMPTS["REFLECTION_FAILURE_PROMPT_2"].format(
                        tentative_answer=recorder.tentative_answer,
                    )
                    await planner_agent.reflect_on_failure(recorder, additional_context)
                    logger.info(f"Self-check failure analysis: {recorder.failure_info}")
                else:
                    # self-check passed, set the final answer
                    recorder.set_final_output(recorder.tentative_answer)
                    break

            else:  # if not success, use the tentative answer as final answer
                if not recorder.final_output:
                    recorder.set_final_output(recorder.tentative_answer)

            # Finally: check the answer?
            # success = await answerer_agent.answer_check()
        return recorder
