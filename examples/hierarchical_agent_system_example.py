import asyncio

from cortex import (
    AgentSystemContext,
    AsyncAgentMemoryBank,
    DepartmentManagerBuilder,
    DelegationBrief,
    DepartmentSpec,
    FunctionStep,
    GatewayNodeBuilder,
    GPTModels,
    HierarchicalAgentSystem,
    LLM,
    NodeResult,
    Tool,
    SpecialistNodeBuilder,
    WorkflowAgent,
    build_specialist_brief,
    build_gateway_prompt,
    build_specialist_prompt,
    synthesize_results,
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


def make_finance_manager_workflow(context, installed_tools):
    async def synthesize_finance_review(state, context, workflow):
        brief = DelegationBrief.from_dict(state.input)
        child_results = []

        for tool in installed_tools:
            specialist_name = tool.name.removesuffix("_node").replace("_", " ").title()
            specialist_brief = build_specialist_brief(
                parent_brief=brief,
                specialist_name=specialist_name,
                scoped_task=f"Review the finance request from the perspective of {specialist_name}.",
                caller_understanding=f"The finance manager wants {specialist_name} analysis for: {brief.original_request_summary}",
                expected_output={"type": "specialist_result"},
            )
            raw_result = await tool.async_run({"brief": specialist_brief.to_dict()}, context, None)
            child_results.append(NodeResult.from_dict(raw_result))

        return synthesize_results(
            brief=brief,
            role="manager",
            from_node="Finance Manager",
            child_results=child_results,
            summary="Finance workflow synthesized specialist reviews into a department recommendation.",
            metadata={"runtime": "workflow"},
        ).to_dict()

    return WorkflowAgent(
        name="Finance Manager Workflow",
        steps=[
            FunctionStep.final(
                name="synthesize_finance_review",
                func=synthesize_finance_review,
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

    finance_manager = DepartmentManagerBuilder.create_workflow(
        name="Finance Manager",
        department="Finance",
        workflow=make_finance_manager_workflow,
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
