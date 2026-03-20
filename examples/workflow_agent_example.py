from cortex import (
    GPTModels,
    LLM,
    WorkflowNodeResult,
    edge,
    function_node,
    llm_node,
    parallel_node,
    router_node,
    workflow,
)


def route_request(state, context):
    text = str(state.data.get("input") or "").lower()
    if "refund" in text or "return" in text:
        return WorkflowNodeResult.next("extract_refund_request")
    return WorkflowNodeResult.next("compose_direct_answer")


def build_refund_prompt(state, context):
    return (
        "Extract the refund-related information from the user's message. "
        "Return a JSON object with keys: intent, order_reference, issue_summary."
    )


def build_direct_answer_prompt(state, context):
    return (
        "Answer the user's request directly and concisely. "
        "If the request is ambiguous, ask a clarifying question."
    )


def build_refund_response_prompt(state, context):
    extracted = state.require("refund_request")
    return (
        "You are preparing a support response for a refund workflow.\n"
        f"Structured request data: {extracted}\n"
        "Write a concise, user-facing response that confirms what information was captured "
        "and states the next step."
    )


def summarize_refund_issue(data, context):
    extracted = data.get("refund_request", {})
    return {"summary": f"Refund summary: {extracted.get('issue_summary', '')}"}


def detect_order_reference(data, context):
    extracted = data.get("refund_request", {})
    return {"order_ref": extracted.get("order_reference")}


def compose_refund_subworkflow(data, context):
    return {"refund_response": f"Processing refund for: {data.get('refund_request', {})}"}


def finalize_refund_answer(data, context):
    analysis = data.get("refund_parallel_analysis", {})
    return analysis.get("compose_refund_subworkflow", {})


support_workflow = workflow(
    name="Support Workflow",
    nodes=[
        router_node(
            name="route_request",
            func=route_request,
            possible_next_nodes=["extract_refund_request", "compose_direct_answer"],
        ),
        llm_node(
            name="extract_refund_request",
            llm=LLM(model=GPTModels.GPT_4O_MINI),
            prompt=build_refund_prompt,
            result_shape={
                "type": "object",
                "properties": {
                    "intent": {"type": "string"},
                    "order_reference": {"type": ["string", "null"]},
                    "issue_summary": {"type": "string"},
                },
                "required": ["intent", "order_reference", "issue_summary"],
                "additionalProperties": False,
            },
            output_key="refund_request",
        ),
        parallel_node(
            name="analyze_refund_request",
            branches={
                "compose_refund_subworkflow": compose_refund_subworkflow,
                "summarize_refund_issue": summarize_refund_issue,
                "detect_order_reference": detect_order_reference,
            },
            output_key="refund_parallel_analysis",
        ),
        function_node(
            name="finalize_refund_answer",
            func=finalize_refund_answer,
            is_final=True,
        ),
        llm_node(
            name="compose_direct_answer",
            llm=LLM(model=GPTModels.GPT_4O_MINI),
            prompt=build_direct_answer_prompt,
            is_final=True,
        ),
    ],
    edges=[
        edge("extract_refund_request", "analyze_refund_request"),
        edge("analyze_refund_request", "finalize_refund_answer"),
    ],
    start_node="route_request",
)


async def main():
    response = await support_workflow.async_ask(
        "I want a refund for order 12345 because the product arrived damaged."
    )
    print(response)
