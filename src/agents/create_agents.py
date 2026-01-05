from __future__ import annotations

import yaml
from autogen import AssistantAgent, UserProxyAgent
from autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent

def get_system_message(agent: str, system_message_path: str = "config/system_message.yaml") -> str:
    with open(system_message_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg[agent]["system_message"]

def termination_msg(x):
    return isinstance(x, dict) and str(x.get("content", "")).strip().upper().endswith("TERMINATE")

def create_mainpath_agents(llm_config: dict, llm_config_4a: dict | None = None, system_message_path: str = "config/system_message.yaml"):
    user = UserProxyAgent(
        name="User",
        is_termination_msg=termination_msg,
        human_input_mode="NEVER",
        max_consecutive_auto_reply=50,
        code_execution_config=False,
        llm_config=False,
        system_message="You are the user proxy for running the pipeline. Provide inputs and accept outputs.",
    )

    a1 = AssistantAgent(
        name="Agent1_Planner",
        llm_config=llm_config,
        system_message=get_system_message("Agent1_Planner", system_message_path),
    )
    a2 = AssistantAgent(
        name="Agent2_CADWriter",
        llm_config=llm_config,
        system_message=get_system_message("Agent2_CADWriter", system_message_path),
    )
    a3 = AssistantAgent(
        name="Agent3_Executor",
        llm_config=llm_config,
        system_message=get_system_message("Agent3_Executor", system_message_path),
    )

    # 4A：用MULTIMODEL（如 gpt-4o / llama-3.2-vision 等）
    a4 = MultimodalConversableAgent(
        name="Agent4A_Verifier",
        llm_config=llm_config_4a or llm_config,
        system_message=get_system_message("Agent4A_Verifier", system_message_path),
    )

    a5 = AssistantAgent(
        name="Agent5_Optimizer",
        llm_config=llm_config,
        system_message=get_system_message("Agent5_Optimizer", system_message_path),
    )

    a6 = AssistantAgent(
        name="Agent6_Memory",
        llm_config=llm_config,
        system_message=get_system_message("Agent6_Memory", system_message_path),
    )

    return [user, a1, a2, a3, a4, a5, a6]

