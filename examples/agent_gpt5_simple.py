"""
Simple GPT-5 agent example demonstrating tool invocation and multi-round conversation.

Prerequisites:
- Set your OpenAI API key in the environment: export OPENAI_API_KEY=... 
- Ensure your account has access to GPT-5; otherwise this script will fall back to GPT-4o automatically.
"""

from cortex import (
    LLM,
    Agent,
    Tool,
    GPTModels,
    AgentMemoryBank,
    LoggingConfig,
    set_default_logging_config,
)


# Optional: make the run verbose so you can see system prompt, messages, and usage
set_default_logging_config(
    LoggingConfig(
        print_system_prompt=True,
        print_messages=True,
        print_usage_report=True,
    )
)

# If GPT-5 is not available on your account, use GPT-4o as a backup automatically
LLM.set_backup_backend(GPTModels.GPT_5, GPTModels.GPT_4O)


# --- Define simple function tools ---

def add_tool():
    """Add two numbers a and b and return the sum as JSON."""

    def func(args):
        a = float(args.get("a", 0))
        b = float(args.get("b", 0))
        return {"result": a + b}

    return Tool(
        name="add",
        func=func,
        description="Add two numbers 'a' and 'b' and return the sum.",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First addend"},
                "b": {"type": "number", "description": "Second addend"},
            },
            "required": ["a", "b"],
            "additionalProperties": False,
        },
    )


def multiply_tool():
    """Multiply two numbers a and b and return the product as JSON."""

    def func(args):
        a = float(args.get("a", 1))
        b = float(args.get("b", 1))
        return {"result": a * b}

    return Tool(
        name="multiply",
        func=func,
        description="Multiply two numbers 'a' and 'b' and return the product.",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First factor"},
                "b": {"type": "number", "description": "Second factor"},
            },
            "required": ["a", "b"],
            "additionalProperties": False,
        },
    )


# --- Build the agent ---

def build_agent():
    llm = LLM(model=GPTModels.GPT_5_MINI)

    tools = [
        add_tool(),
        multiply_tool(),
    ]

    sys_prompt = (
        "You are a helpful math assistant.\n"
        "- Prefer using the provided function tools for any calculation rather than mental math.\n"
        "- When tools are used, wait for the tool outputs and then provide a succinct final answer.\n"
        "- Keep responses brief."
        "- Only use each tool once and don't repeat the same tool."
    )

    # Keep the last few message groups to support follow-up questions
    memory = AgentMemoryBank.bank_for("demo_user").get_agent_memory("gpt5_math_agent", k=5)

    return Agent(
        llm=llm,
        tools=tools,
        sys_prompt=sys_prompt,
        memory=memory,
        name="GPT5-Math-Demo",
    )


# --- Run a short multi-round demo ---

def run_demo():
    agent = build_agent()

    print("\n=== Round 1 ===")
    r1 = agent.ask("What is 42.5 + 8.75? Please compute using tools.")
    print("Assistant:", r1)

    print("\n=== Round 2 ===")
    r2 = agent.ask("Now multiply that result by 3.2 using tools only.")
    print("Assistant:", r2)

    print("\n=== Round 3 ===")
    r3 = agent.ask("Compute (7 + 5) times 9. Use tools for each step.")
    print("Assistant:", r3)


if __name__ == "__main__":
    run_demo()
