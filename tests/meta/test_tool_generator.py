from agents.mcp import MCPServerStdio

from utu.config import ConfigLoader
from utu.utils import DIR_ROOT


async def test_generated_tool():
    config_name = "download_bilibili_video"
    action = (
        "download_video",
        {
            "url": "https://www.bilibili.com/video/BV1LFemztE3W/",
            "output_path": str(DIR_ROOT / "data" / "youtu-agent.mp4"),
        },
    )
    # config_name = "convert_video_to_gif"
    # action = ("convert_video_to_gif", {"input_path": str(DIR_ROOT / "data" / "youtu-agent.mp4")})

    config = ConfigLoader.load_toolkit_config(f"generated/{config_name}")
    async with MCPServerStdio(
        params=config.config,
    ) as server:
        await server.connect()
        tools = await server.list_tools()
        print(tools)
        res = await server.call_tool(*action)
        print(res)
