import os
import json
import re

# 抑制来自 gRPC 的低级别日志，让输出更干净
os.environ['GRPC_VERBOSITY'] = 'ERROR'

import google.generativeai as genai
from googleapiclient.discovery import build
from dotenv import load_dotenv

# --- 1. 配置与初始化 (来自 config.py) ---

# 加载环境变量
load_dotenv()

# 配置 Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# 配置 Google Search API
CUSTOM_SEARCH_API_KEY = os.getenv("CUSTOM_SEARCH_API_KEY")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")

# 初始化模型
# Planner: 使用 Pro 模型，因为它需要强大的推理和规划能力
planner_model = genai.GenerativeModel('gemini-1.5-pro-latest')

# Worker: 使用 Flash 模型，因为它快速且经济，适合执行具体任务
worker_model = genai.GenerativeModel('gemini-1.5-flash-latest')

print("✅ Models and APIs configured.")


# --- 2. 工具定义 (来自 Google Search.ini) ---

def google_search(query: str, num_results: int = 5, date_range: dict = None) -> list | str:
    """
    使用 Google Custom Search API 执行搜索，支持按日期范围过滤。
    每个结果是一个包含 title, link, 和 snippet 的字典。
    """
    print(f"🔍 Performing search for: '{query}'")
    try:
        search_params = {
            'q': query,
            'cx': SEARCH_ENGINE_ID,
            'num': num_results
        }
        if date_range and 'start' in date_range and 'end' in date_range:
            # Google Search API 的日期格式是 YYYYMMDD
            sort_string = f"date:r:{date_range['start']}:{date_range['end']}"
            search_params['sort'] = sort_string
            print(f"   (Date range: {date_range['start']} to {date_range['end']})")

        service = build("customsearch", "v1", developerKey=CUSTOM_SEARCH_API_KEY)
        result = (
            service.cse()
            .list(**search_params)
            .execute()
        )
        # 提取并格式化我们关心的信息
        formatted_results = [
            {
                "title": item.get("title"),
                "link": item.get("link"),
                "snippet": item.get("snippet"),
            }
            for item in result.get("items", [])
        ]
        return formatted_results
    except Exception as e:
        return f"Error during Google Search: {e}"

def write_file(report_dir: str, filename: str, content: str) -> str:
    """
    将内容写入指定报告目录下的文件。
    如果目录不存在，则创建它。
    """
    os.makedirs(report_dir, exist_ok=True)
    filepath = os.path.join(report_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"📄 File created: {filepath}")
    return f"Successfully created {filename}"

# --- 3. Planner 定义 (来自 Planner (Gemini 1.5 Pro)) ---

def create_research_plan(topic: str) -> str:
    """
    使用 Planner (Gemini 1.5 Pro) 为给定课题创建一个 JSON 格式的研究计划。
    """
    prompt = f"""
    你是一个世界级的AI研究助理和规划师。你的任务是为一个复杂的研究课题制定一个详细、分步的行动计划。
    最终目标是生成一份全面的课题报告。

    **重要**: 你的输出必须是一个合法的、可以被 Python 的 `json.loads()` 解析的 JSON 对象。这个对象包含两个键: "plan" 和 "report_structure"。
    - "plan" 是一个步骤数组。
    - "report_structure" 是一个描述最终报告结构的对象，例如 `{"type": "html_multi_page"}`。

    每个步骤都是一个 JSON 对象，包含 `step_number`, `title`, `description`, 和一个 `action` 对象。
    `action` 对象必须包含 `tool` (例如 "google_search", "write_file") 和 `parameters`。
    - `google_search` 的 `parameters` 可以包含 `date_range`，其 `start` 和 `end` 值的格式为 YYYYMMDD。
    - `write_file` 的 `parameters` 应该包含 `filename` 和 `content_description` (描述需要生成的内容)。

    这是一个输出格式的例子:
    ```json
    {
      "plan": [
        {
          "step_number": 1,
          "title": "在指定日期范围内搜索核心动态",
          "description": "搜索2025年9月的固态电池相关新闻、技术突破和市场报告。",
          "action": {
            "tool": "google_search",
            "parameters": { 
              "query": "固态电池 最新动态 技术突破 市场分析",
              "date_range": { "start": "20250901", "end": "20250930" }
            }
          }
        },
        {
          "step_number": 2,
          "title": "创建报告主页 (index.html)",
          "description": "基于收集到的信息，生成HTML格式的报告主页，包含标题、摘要和导航链接。",
          "action": {
            "tool": "write_file",
            "parameters": {
              "filename": "index.html",
              "content_description": "一个包含标题、摘要和导航链接到其他页面的HTML页面。使用现代化的CSS样式，确保页面美观、响应式。"
            }
          }
        }
      ],
      "report_structure": {
        "type": "html_multi_page"
      }
    }
    ```

    现在，请为以下课题生成一个 JSON 格式的研究计划:
    课题: "{topic}"
    """
    
    print("🧠 Planner is thinking... This may take up to 30 seconds. Please wait.")
    response = planner_model.generate_content(
        prompt,
        request_options={"timeout": 60}  # 设置60秒超时
    )
    
    # 使用正则表达式更稳健地提取 JSON 内容
    match = re.search(r"```json\s*([\s\S]*?)\s*```", response.text)
    if match:
        return match.group(1).strip()
    else:
        # 如果没有找到 markdown 块，假设模型直接返回了 JSON
        return response.text.strip()


# --- 4. Worker 定义 (来自 Worker .ini) ---

def generate_file_content(prompt: str) -> str:
    """使用Planner/Coder(Pro模型)生成文件内容。"""
    print(f"✍️ Generating file content...")
    # 使用Pro模型进行代码生成，因为它更强大
    return planner_model.generate_content(prompt, request_options={"timeout": 120}).text

def execute_task(instruction: str, context: str = "") -> str:
    """
    使用 Worker (Gemini 1.5 Flash) 执行一个具体的任务。
    """
    prompt = f"""
    你是一个高效的AI助手。请严格按照以下指令执行任务。保持回答简洁、直接。

    指令: {instruction}

    上下文:
    ---
    {context}
    ---
    """
    
    print(f"🛠️ Worker is executing: {instruction[:50]}...")
    response = worker_model.generate_content(
        prompt,
        request_options={"timeout": 60}  # 设置60秒超时
    )
    return response.text


# --- 5. Orchestrator 定义 (来自 Orchestrator.ini, 已升级) ---

def run_research_agent(topic: str):
    """
    运行整个研究代理流程。
    """
    print(f"🚀 Starting research on topic: '{topic}'")
    
    try:
        # 1. Planner 创建 JSON 计划和报告结构
        research_plan_json_str = create_research_plan(topic)
        print("\n📝 Research Plan Created (JSON):")
        print(research_plan_json_str)
        
        # 解析整个计划对象
        full_plan = json.loads(research_plan_json_str)
        plan_steps = full_plan.get("plan", [])
        report_structure = full_plan.get("report_structure", {"type": "text"})

        # 为报告创建一个唯一的目录
        report_dir = "report_" + re.sub(r'\W+', '_', topic)[:50]

    except json.JSONDecodeError:
        print("❌ Error: Failed to parse the research plan. The Planner did not return valid JSON.")
        print("--- Model's Raw Output ---")
        print(research_plan_json_str)
        print("--------------------------")
        return
    except Exception as e: # 捕获所有其他可能的错误, 如网络问题、API密钥无效等
        print(f"❌ Error: An error occurred during the planning phase: {e}")
        return

    # 存储研究结果
    # 这是一个简单的内存，用于在步骤之间传递信息
    research_findings = {}
    
    # 2. Orchestrator 循环执行计划
    for step in plan_steps:
        step_num = step.get('step_number', 'N/A')
        title = step.get('title', 'Untitled')
        action = step.get('action', {})
        tool = action.get('tool')
        
        print(f"\n▶️ Executing Step {step_num}: {title}")
        
        if tool == "google_search":
            query = action.get('parameters', {}).get('query')
            if not query:
                print("⚠️ Warning: Search step is missing a query.")
                continue
            date_range = action.get('parameters', {}).get('date_range')
            search_results = google_search(query, date_range=date_range)
            research_findings[f"step_{step_num}_search"] = search_results
            
            summary_instruction = f"根据以下搜索结果，总结关于 '{query}' 的要点。"
            summary = execute_task(summary_instruction, str(search_results))
            
            research_findings[f"step_{step_num}_summary"] = summary
            print("\n✅ Worker Summary:")
            print(summary)

        elif tool == "write_file":
            filename = action.get('parameters', {}).get('filename')
            content_desc = action.get('parameters', {}).get('content_description')
            if not filename or not content_desc:
                print("⚠️ Warning: Write file step is missing filename or content description.")
                continue
            
            # 让 Planner/Coder (Pro模型) 生成文件内容
            generation_prompt = f"""
            你是一个专业的Web前端开发者和内容作者。
            根据以下指令和上下文信息，生成文件 `{filename}` 的完整内容。
            
            指令: {content_desc}
            
            上下文信息 (来自之前的研究步骤):
            {json.dumps(research_findings, indent=2, ensure_ascii=False)}
            
            请只输出文件的完整代码，不要包含任何额外的解释或markdown标记。
            """
            print(f"✍️ Generating content for {filename}...")
            file_content = generate_file_content(generation_prompt)
            write_file(report_dir, filename, file_content)

        elif tool == "think":
            print("🤔 This is a thinking step. No action taken by tools.")
        else:
            print(f"⚠️ Warning: Unknown tool '{tool}' in step {step_num}.")

    # 3. 最终完成提示
    if report_structure.get("type") == "html_multi_page":
        print("\n\n🎉 === HTML REPORT GENERATED === 🎉\n")
        print(f"报告已生成在目录: ./{report_dir}")
        try:
            # 尝试打开主页
            main_page = os.path.join(report_dir, 'index.html')
            print(f"你可以在浏览器中打开文件: {os.path.abspath(main_page)}")
        except Exception:
            pass
    else:
        # 对于旧的文本报告模式（如果需要）
        print("\n\n🎉 === RESEARCH COMPLETE === 🎉\n")
        print("所有研究步骤已完成。最终产出请查看上方日志。")

# --- 6. 运行代理 ---
if __name__ == "__main__":
    # 将一次性任务改为交互式循环
    while True:
        print("\n" + "="*50)
        # 提示用户输入
        research_topic = input("请输入您想研究的课题 (输入 'exit' 或 'quit' 退出): ")
        
        if research_topic.lower() in ['exit', 'quit']:
            print("👋 感谢使用，再见！")
            break
        
        run_research_agent(research_topic)