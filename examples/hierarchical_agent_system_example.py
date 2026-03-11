import asyncio

from cortex import (
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
async def main():
    memory_bank = AsyncAgentMemoryBank()
    context = AgentSystemContext(memory_bank=memory_bank)

    gateway_builder = GatewayNodeBuilder.create_agent(
        name="Gateway",
        llm=LLM(model=GPTModels.GPT_4O_MINI),
        prompt=build_gateway_prompt(
            organization_context="You route user requests across business departments and produce a final executive summary."
        ),
        description="Routes user tasks to department managers and synthesizes the final response.",
    )

    finance_manager = DepartmentManagerBuilder.create_agent(
        name="Finance Manager",
        department="Finance",
        llm=LLM(model=GPTModels.GPT_4O_MINI),
        prompt=build_manager_prompt(
            department_name="Finance",
            department_description="Owns vendor risk, spend review, and procurement concerns.",
        ),
        description="Coordinates finance-domain specialist reviews.",
    )

    vendor_risk_specialist = SpecialistNodeBuilder.create_agent(
        name="Vendor Risk Specialist",
        specialty="Vendor Risk",
        llm=LLM(model=GPTModels.GPT_4O_MINI),
        prompt=build_specialist_prompt(
            specialty_name="Vendor Risk Analysis",
            specialty_description="Assesses third-party vendor operational and procurement risk.",
        ),
        tools=[vendor_risk_tool],
        description="Handles vendor risk investigations.",
    )

    compliance_specialist = SpecialistNodeBuilder.create_workflow(
        name="Compliance Specialist",
        specialty="Compliance",
        workflow=compliance_workflow,
        description="Runs the workflow-based compliance review.",
    )

    finance_department = DepartmentSpec.create(
        name="Finance",
        description="Handles procurement, spend, and vendor onboarding reviews.",
        manager=finance_manager,
    ).add_specialists(vendor_risk_specialist, compliance_specialist)

    system = HierarchicalAgentSystem.create(
        gateway_builder=gateway_builder,
        departments=[finance_department],
        context=context,
    )

    print(system.describe_hierarchy())
    response = await system.async_ask("Should we approve Acme Power Services for onboarding this quarter?")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
