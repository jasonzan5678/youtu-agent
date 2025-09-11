import re

from ...config import AgentConfig
from ...utils import FileUtils, get_logger
from ..llm_agent import LLMAgent
from .data import WorkspaceTaskRecorder

logger = get_logger(__name__)
PROMPTS: dict[str, str] = FileUtils.load_prompts("agents/workforce/answerer.yaml")


class AnswererAgent:
    """Answer extractor that handles final answer generation from task execution results."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.llm = LLMAgent(config.workforce_answerer_model)

    async def extract_final_answer(self, recorder: WorkspaceTaskRecorder) -> None:
        """Extract the final answer from formatted task execution results."""
        # Generate final answer prompt
        final_prompt = (
            PROMPTS["FINAL_ANSWER_PROMPT"]
            .strip()
            .format(
                question=recorder.overall_task,
                task_results="\n\n".join(recorder.formatted_task_plan_list_with_task_results),
            )
        )
        final_recorder = await self.llm.run(final_prompt)
        recorder.add_run_result(final_recorder.get_run_result(), "answerer_extract_final_answer")  # add trajectory
        final_answer, confidence, uniqueness_assessment = self._parse_final_response(final_recorder.final_output)
        recorder.update_tentative_answer(final_answer, confidence, uniqueness_assessment)

    def _parse_final_response(self, response: str) -> tuple[str, str, str]:
        answer_pattern = r"<answer>(.*?)</answer>"
        answer_match = re.search(answer_pattern, response, re.DOTALL)
        final_answer = answer_match.group(1).strip() if answer_match else response.strip()

        def _check(s: str, p: str) -> bool:
            return s.startswith(p) or " " + p + " " in s or s.endswith(p)

        confidence_pattern = r"<confidence>(.*?)</confidence>"
        confidence_match = re.search(confidence_pattern, response, re.DOTALL)
        confidence = "low"
        confidence_text = confidence_match.group(1).strip().lower() if confidence_match else "low"
        if _check(confidence_text, "high"):
            confidence = "high"
        elif _check(confidence_text, "medium"):
            confidence = "medium"

        uniqueness_pattern = r"<answer_uniqueness>(.*?)</answer_uniqueness>"
        uniqueness_match = re.search(uniqueness_pattern, response, re.DOTALL)
        uniqueness_assessment = "unclear"  # default
        uniqueness_text = uniqueness_match.group(1).strip().lower() if uniqueness_match else "unclear"
        if _check(uniqueness_text, "unique"):
            uniqueness_assessment = "unique"
        elif _check(uniqueness_text, "non-unique"):
            uniqueness_assessment = "non-unique"
        return final_answer, confidence, uniqueness_assessment

    async def answer_check(self, question: str, model_answer: str, ground_truth: str) -> bool:
        """Check if model answer and ground truth are semantically equivalent using LLM."""
        raise NotImplementedError

    async def answer_self_check(self, recorder: WorkspaceTaskRecorder) -> bool:
        """Self-check if the attempted answer follows correct format and process."""
        self_check_prompt = PROMPTS["ANSWER_SELF_CHECK_PROMPT"].format(
            question=recorder.overall_task,
            task_results="\n\n".join(recorder.formatted_task_plan_list_with_task_results),
            attempt_answer=recorder.tentative_answer,
        )
        self_check_recorder = await self.llm.run(self_check_prompt)
        recorder.add_run_result(self_check_recorder.get_run_result(), "answerer_self_check")  # add trajectory
        correct_pattern = r"<correct>(.*?)</correct>"
        correct_match = re.search(correct_pattern, self_check_recorder.final_output, re.DOTALL)
        if correct_match:
            return correct_match.group(1).strip().lower() == "yes"
        return False
