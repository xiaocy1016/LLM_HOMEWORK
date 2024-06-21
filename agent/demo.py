import pprint
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool


# 步骤 1：配置您所使用的 LLM。
llm_cfg = {
    # 使用与 OpenAI API 兼容的模型服务，例如 vLLM 或 Ollama：
    'model': 'Qwen1.5-14B',
    'model_server': 'http://10.58.0.2:8000/v1', 

    'api_key': 'None',

    # （可选） LLM 的超参数：
    # 'generate_cfg': {
    #     'top_p': 0.8
    # }
}

# 步骤 2：创建一个智能体。这里我们以 `Assistant` 智能体为例，它能够使用工具并读取文件。
system_instruction = '''你是一个乐于助人的AI助手，可以通过用户命令来控制选课系统。
你总是用中文回复用户。'''
tools = ['code_interpreter']  # `code_interpreter` 是框架自带的工具，用于执行代码。
files = ['./tutorial.md']  # 给智能体一个 md 文件阅读。
bot = Assistant(llm=llm_cfg,
                system_message=system_instruction,
                function_list=tools,
                files=files)

# 步骤 3：作为聊天机器人运行智能体。
messages = []  # 这里储存聊天历史。
while True:
    # 例如，输入请求 "绘制一只狗并将其旋转 90 度"。
    query = input('用户请求: ')
    # 将用户请求添加到聊天历史。
    messages.append({'role': 'user', 'content': query})
    response = []

    last = []

    for response in bot.run(messages=messages):
        # 流式输出，存储最终响应
        last = response
    # 将机器人的回应添加到聊天历史。
    messages.extend(last)
    print('机器人回应:')
    pprint.pprint(last[-1], indent=1)