import asyncio

from utu.meta import ToolGenerator
from utu.utils import AgentsUtils, PrintUtils


async def main():
    generator = ToolGenerator()
    task = await PrintUtils.async_print_input("Enter your tool requirements: ")
    task_recorder = generator.run_streamed(task)
    await AgentsUtils.print_stream_events(task_recorder.stream_events())
    print(f"Generated tool config saved to {task_recorder.output_file}")


if __name__ == "__main__":
    asyncio.run(main())
