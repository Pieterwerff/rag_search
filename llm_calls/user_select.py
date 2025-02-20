def user_input() -> str:
    user_input = input("""
        Select the LLM to use:
        1) meta-llama/Llama-3.3-70B-Instruct
        2) microsoft/phi-4
        3) deepseek-ai/DeepSeek-V3
        4) NousResearch/Hermes-3-Llama-3.1-405B
        5) Qwen/QwQ-32B-Preview
        6) nvidia/Llama-3.1-Nemotron-70B-Instruct
        7) Qwen/Qwen2.5-72B-Instruct
        8) 01-ai/Yi-34B-Chat
        9) databricks/dbrx-instruct
    """)

    if int(user_input) == 1:
        llm = "meta-llama/Llama-3.3-70B-Instruct"
    elif int(user_input) == 2:
        llm = "microsoft/phi-4"
    elif int(user_input) == 3:
        llm = "deepseek-ai/DeepSeek-V3"
    elif int(user_input) == 4:
        llm = "NousResearch/Hermes-3-Llama-3.1-405B"
    elif int(user_input) == 5:
        llm = "Qwen/QwQ-32B-Preview"
    elif int(user_input) == 6:
        llm = "nvidia/Llama-3.1-Nemotron-70B-Instruct"
    elif int(user_input) == 7:
        llm = "Qwen/Qwen2.5-72B-Instruct"
    elif int(user_input) == 8:
        llm = "01-ai/Yi-34B-Chat"
    elif int(user_input) == 9:
        llm = "databricks/dbrx-instruct"

    return llm