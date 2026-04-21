# %% 
import asyncio
import datetime
import os
import sys
import threading
import time
import urllib.request
from collections.abc import Callable
from copy import copy
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import jwt  # PyJWT — for MCP bonus section
import numpy as np
from control_arena import ControlAgent, control_agent
from control_arena.analysis import samples_df
from control_arena.eval import run_monitor_for_eval_logs
from control_arena.micro_protocol import trusted_monitoring_builder
from control_arena.monitor import (
    MonitorResponse,
    basic_monitor_builder,
    cot_prompt_formatter,
)
from control_arena.monitor._types import MonitorScoreRange
from control_arena.policy import (
    attack_policy_builder,
    honest_policy_builder,
    trusted_policy_builder,
    untrusted_policy_builder,
)
from control_arena.scorers import max_suspicion_scorer
from control_arena.settings.apps.apps_solution_policy import apps_solution_policy
from inspect_ai import score as inspect_score
from inspect_ai.agent import Agent, AgentState
from inspect_ai.log import write_eval_log
from inspect_ai.model import ChatMessageAssistant
from inspect_ai.tool import Tool
from mcp.server.fastmcp import FastMCP  # for MCP bonus section
from openai import OpenAI
from pydantic import Field
#%%
for _path in [
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parent.parent),
]:
    if _path not in sys.path:
        sys.path.insert(0, _path)
from day2_utils.eval_utils import extract_scores, load_eval_logs
from day2_utils.generate_trajectories import generate_trajectories
from day2_utils.metrics import calculate_roc_metrics
from day2_utils.plotting import (
    plot_attack_evasion_curve,
    plot_roc_comparison,
    plot_roc_curve,
    plot_safety_usefulness_frontier,
    plot_score_histograms,
)

from aisb_utils import report
from aisb_utils.env import load_dotenv

load_dotenv()

#%%
# Paths relative to this file
SCRIPT_DIR = Path(__file__).parent

# OpenRouter client for exercises 1-2
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY", ""),
)

# A cheap, small model for exercises where we *want* the model to be easily manipulated
SMALL_MODEL = "meta-llama/llama-3-8b-instruct"

#%%
#Exercise 1: **Task: List channels through which untrusted input can reach the model's context window.** Think beyond the chat box — the agent *reads* a lot of content while doing its job.

# Input channels include:
# 1. The chat box where the user inputs messages.
# 2. Tools that the agent uses, which may return content that the agent reads.
# 3. The agent's own internal state or memory, if it stores information from previous interactions.
# 4. External data sources or APIs that the agent queries during its operation.
# 5. System messages or prompts that are provided to the agent at the start of a conversation or task.


#%%

user_input = input("What is the model weight of your model")

state = AgentState(messages=[])

state.messages.append(
    ChatMessageUser(content=user_input)
)

new_state = await agent(state)

for msg in new_state.messages:
    print(f"\nRole: {msg.role}")
    print(msg.content)
# %%
