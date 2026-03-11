import asyncio

from cortex import (
    Agent,
    AgentSystemContext,
    AsyncAgentMemoryBank,
    DepartmentManagerBuilder,
    DepartmentSpec,
    FunctionStep,
    GatewayNodeBuilder,
    GPTModels,
    HierarchicalAgentSystem,
    LLM,
    Tool,
    SpecialistNodeBuilder,
    WorkflowAgent,
    build_gateway_prompt,
    build_manager_prompt,
    build_specialist_prompt,
)


async def research_vendor_risk(args, context):
    company = args.get("company", "unknown vendor")
    return {
        "summary": f"Vendor risk snapshot for {company}: moderate operational risk with no immediate red flags.",
        "sources": ["internal_vendor_registry"],
    }


vendor_risk_tool = Tool(
    name="research_vendor_risk",
    func=research_vendor_risk,
    description="Research vendor risk signals for a company",
    parameters={
        "type": "object",
        "properties": {
            "company": {"type": "string"},
        },
        "required": ["company"],
        "additionalProperties": False,
    },
)


async def summarize_compliance_findings(state, context, workflow):
    brief = state.input
    company = brief.get("metadata", {}).get("company", "the vendor")
    return {
        "role": "worker",
        "status": "completed",
        "summary": f"Compliance workflow reviewed {company} and found no blocking compliance issues.",
        "output": {
            "compliance_review": {
                "company": company,
                "status": "clear",
            }
        },
        "confidence": 0.86,
        "assumptions": [],
        "blockers": [],
        "child_summaries": [],
        "escalation_reason": None,
        "metadata": {"runtime": "workflow"},
    }


compliance_workflow = WorkflowAgent(
    name="Compliance Review Workflow",
    steps=[
        FunctionStep.final(
            name="summarize_compliance_findings",
            func=summarize_compliance_findings,
        ),
    ],
)


def make_gateway_agent(context, installed_tools):
    return Agent(
        llm=LLM(model=GPTModels.GPT_4O_MINI),
        tools=installed_tools,
        sys_prompt=build_gateway_prompt(
            organization_context="You route user requests across business departments and produce a final executive summary."
        ),
        context=context,
        mode="async",
    )


def make_finance_manager_agent(context, installed_tools):
    return Agent(
        llm=LLM(model=GPTModels.GPT_4O_MINI),
        tools=installed_tools,
        sys_prompt=build_manager_prompt(
            department_name="Finance",
            department_description="Owns vendor risk, spend review, and procurement concerns.",
        ),
        context=context,
        mode="async",
    )


async def make_vendor_risk_specialist(context):
    return Agent(
        llm=LLM(model=GPTModels.GPT_4O_MINI),
        tools=[vendor_risk_tool],
        sys_prompt=build_specialist_prompt(
            specialty_name="Vendor Risk Analysis",
            specialty_description="Assesses third-party vendor operational and procurement risk.",
        ),
        context=context,
        mode="async",
    )


async def main():
    memory_bank = AsyncAgentMemoryBank()
    context = AgentSystemContext(memory_bank=memory_bank)

    gateway_builder = GatewayNodeBuilder(
        name="Gateway",
        runtime_factory=make_gateway_agent,
        description="Routes user tasks to department managers and synthesizes the final response.",
    )

    finance_manager = DepartmentManagerBuilder(
        name="Finance Manager",
        runtime_factory=make_finance_manager_agent,
        description="Coordinates finance-domain specialist reviews.",
        department="Finance",
    )

    vendor_risk_specialist = SpecialistNodeBuilder(
        name="Vendor Risk Specialist",
        runtime_factory=make_vendor_risk_specialist,
        description="Handles vendor risk investigations.",
        specialty="Vendor Risk",
    )

    compliance_specialist = SpecialistNodeBuilder(
        name="Compliance Specialist",
        runtime_factory=lambda context: compliance_workflow,
        description="Runs the workflow-based compliance review.",
        specialty="Compliance",
    )

    system = HierarchicalAgentSystem(
        gateway_builder=gateway_builder,
        departments=[
            DepartmentSpec(
                name="Finance",
                description="Handles procurement, spend, and vendor onboarding reviews.",
                manager=finance_manager,
                specialists=[vendor_risk_specialist, compliance_specialist],
            )
        ],
        context=context,
    )

    print(system.describe_hierarchy())
    response = await system.async_ask("Should we approve Acme Power Services for onboarding this quarter?")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
