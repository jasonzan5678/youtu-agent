import asyncio
import subprocess
from dataclasses import dataclass, field

from agents import RunResultStreaming, trace
from agents._run_impl import QueueCompleteSentinel

from ..agents import SimpleAgent
from ..agents.common import DataClassWithStreamEvents
from ..utils import DIR_ROOT, FileUtils, LLMOutputParser, PrintUtils, get_logger

logger = get_logger(__name__)


@dataclass
class TaskRecorder(DataClassWithStreamEvents):
    requirements: str = field(default_factory=str)  # structured, but not used for now
    implementation: str = field(default_factory=str)
    manifest: dict = field(default_factory=dict)
    # class_name, requirements, methods
    output_file: str = field(default_factory=str)


class ToolGenerator:
    def __init__(self):
        self.prompts = FileUtils.load_prompts("meta/tool_generator_mcp.yaml")
        self.llm = SimpleAgent(name="tool_generator", instructions="You are a Python software engineer assistant.")
        self.output_dir = DIR_ROOT / "configs/tools/generated"
        self.output_dir.mkdir(exist_ok=True)

        self._initialized = False

    async def build(self):
        if self._initialized:
            return

    async def run(self, user_input: str):
        await self.build()
        self.llm.clear_input_items()
        with trace("tool_generator"):
            task_recorder = TaskRecorder()
            # step 1: generate requirements
            await self.step1(task_recorder, user_input)
            # step 2: generate implementation
            await self.step2(task_recorder)
            # step 3: generate manifest
            await self.step3(task_recorder)
            # postprocess
            self.postprocess(task_recorder)
        print(f"Generated tool config saved to {task_recorder.output_file}")

    def run_streamed(self, user_input: str) -> TaskRecorder:
        task_recorder = TaskRecorder()
        task_recorder._run_impl_task = asyncio.create_task(self._start_streaming(task_recorder, user_input))
        return task_recorder

    async def _start_streaming(self, task_recorder: TaskRecorder, user_input: str):
        await self.build()
        with trace("tool_generator"):
            await self.step1(task_recorder, user_input)
            await self.step2(task_recorder)
            await self.step3(task_recorder)
            self.postprocess(task_recorder)

        task_recorder._event_queue.put_nowait(QueueCompleteSentinel())
        task_recorder._is_complete = True

    async def step1(self, task_recorder: TaskRecorder, user_input: str) -> None:
        async with self.llm as agent:
            query = FileUtils.get_jinja_template_str(self.prompts["SETP_1_REQUIREMENT"]).render(user_request=user_input)
            res = agent.run_streamed(query)
            await self._process_streamed(res, task_recorder)
            task_recorder.requirements = res.final_output  # DISCUSS: parse requirements

    async def step2(self, task_recorder: TaskRecorder) -> None:
        async with self.llm as agent:
            query = FileUtils.get_jinja_template_str(self.prompts["SETP_2_IMPLEMENTATION"]).render()
            res = agent.run_streamed(query)
            await self._process_streamed(res, task_recorder)
            task_recorder.implementation = LLMOutputParser.extract_code_python(res.final_output)

    async def step3(self, task_recorder: TaskRecorder) -> None:
        async with self.llm as agent:
            query = FileUtils.get_jinja_template_str(self.prompts["SETP_3_MANIFEST"]).render()
            res = agent.run_streamed(query)
            await self._process_streamed(res, task_recorder)
            task_recorder.manifest = LLMOutputParser.extract_code_json(res.final_output)

    def postprocess(self, task_recorder: TaskRecorder) -> None:
        name = LLMOutputParser.camel_to_snake(task_recorder.manifest["class_name"])
        odir = self.output_dir / name
        odir.mkdir(exist_ok=True)
        with open(odir / "runner.py", "w") as f:
            f.write(task_recorder.implementation)
        with open(odir / "main.py", "w") as f:
            f.write(self.prompts["TEMPLATE_MAIN"])
        with open(odir / "manifest.json", "w") as f:
            f.write(
                FileUtils.get_jinja_template_str(self.prompts["TEMPLATE_MANIFEST"]).render(
                    name=name,
                    class_name=task_recorder.manifest["class_name"],
                    requirements=PrintUtils.format_json(task_recorder.manifest["requirements"]),
                    methods=PrintUtils.format_json(task_recorder.manifest["methods"]),
                )
            )
        with open(odir / "requirements.txt", "w") as f:
            f.write("\n".join(task_recorder.manifest["requirements"] + ["mcp"]))  # mcp is required
        config_fn = self.output_dir / f"{name}.yaml"
        with config_fn.open("w") as f:
            f.write(
                FileUtils.get_jinja_template_str(self.prompts["TEMPLATE_CONFIG"]).render(
                    name=name,
                )
            )
        task_recorder.output_file = str(config_fn)

        # init the environment
        self._init_environment(odir)

    def _init_environment(self, odir):
        """Initialize the virtual environment and install requirements."""
        subprocess.run(
            f"cd {odir} && uv venv && . .venv/bin/activate && uv pip install -r requirements.txt",
            shell=True,
            check=True,
        )

    async def _process_streamed(self, run_result_streaming: RunResultStreaming, task_recorder: TaskRecorder):
        async for event in run_result_streaming.stream_events():
            task_recorder._event_queue.put_nowait(event)
        self.llm.input_items = run_result_streaming.to_input_list()
