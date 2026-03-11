import asyncio

from cortex import (
    AgentSystemContext,
    AsyncAgentMemoryBank,
    FunctionStep,
    GPTModels,
    LLM,
    TaskCoordinatorBuilder,
    TaskCoordinatorSystem,
    TaskWorkerBuilder,
    Tool,
    WorkflowAgent,
)


def gross_margin_tool(args, context):
    revenue = float(args["revenue"])
    cost = float(args["cost"])
    margin = 0.0 if revenue == 0 else (revenue - cost) / revenue
    return {
        "gross_margin": margin,
        "gross_profit": revenue - cost,
    }


finance_tool = Tool(
    name="calculate_gross_margin",
    func=gross_margin_tool,
    description="Calculate gross margin and gross profit from revenue and cost inputs.",
    parameters={
        "type": "object",
        "properties": {
            "revenue": {"type": "number"},
            "cost": {"type": "number"},
        },
        "required": ["revenue", "cost"],
        "additionalProperties": False,
    },
)


async def summarize_risk_review(state, context, workflow):
    desc = state.input
    company = desc.get("metadata", {}).get("company", "the vendor")
    return {
        "conversation_id": desc["conversation_id"],
        "task_id": desc["task_id"],
        "parent_task_id": desc.get("parent_task_id"),
        "from_node": desc["to_node"],
        "to_node": desc["from_node"],
        "role": "worker",
        "status": "completed",
        "summary": f"Risk workflow completed a light vendor review for {company}.",
        "output": {
            "risk_review": {
                "company": company,
                "status": "moderate_risk",
                "recommendation": "Proceed with standard controls and contract review.",
            }
        },
        "confidence": 0.83,
        "assumptions": [],
        "blockers": [],
        "child_summaries": [],
        "escalation_reason": None,
        "metadata": {"runtime": "workflow"},
    }


risk_review_workflow = WorkflowAgent(
    name="Risk Review Workflow",
    steps=[
        FunctionStep.final(
            name="summarize_risk_review",
            func=summarize_risk_review,
        ),
    ],
)


def finance_prompt(context):
    return (
        "You are the finance worker. Read the TaskDesc carefully, use tools when useful, "
        "and return a TaskResult-shaped JSON object."
    )


def coordinator_prompt(context):
    return (
        "You coordinate finance and vendor-risk specialists for procurement decisions. "
        "Rewrite the user's request into clear worker-facing TaskDesc handoffs and synthesize "
        "their TaskResults into the final answer."
    )


async def main():
    context = AgentSystemContext(memory_bank=AsyncAgentMemoryBank())

    finance_worker = TaskWorkerBuilder.create_agent(
        name="Finance Analyst",
        llm=LLM(model=GPTModels.GPT_4O_MINI),
        prompt_builder=finance_prompt,
        tools=[finance_tool],
        description="Analyzes unit economics, pricing, and margin implications.",
    )

    risk_worker = TaskWorkerBuilder.create_workflow(
        name="Vendor Risk Reviewer",
        workflow=risk_review_workflow,
        description="Performs a workflow-based vendor risk review.",
    )

    coordinator = TaskCoordinatorBuilder.create_agent(
        name="Task Coordinator",
        llm=LLM(model=GPTModels.GPT_4O_MINI),
        prompt_builder=coordinator_prompt,
        description="Delegates procurement questions to structured task workers.",
    )

    system = TaskCoordinatorSystem(
        coordinator=coordinator,
        workers=[finance_worker, risk_worker],
        context=context,
    )

    response = await system.async_ask(
        "Should we approve Acme Power Services for onboarding if revenue is 120000 and cost is 85000?",
        metadata={"company": "Acme Power Services"},
    )
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
