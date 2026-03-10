from cortex import GPTModels, LLM, LLMStep, RouterStep, StepResult, WorkflowAgent


def route_request(state, context, workflow):
    text = str(state.input or "").lower()
    if "refund" in text or "return" in text:
        return StepResult.next("extract_refund_request")
    return StepResult.next("compose_direct_answer")


def build_refund_prompt(state, context, workflow):
    return (
        "Extract the refund-related information from the user's message. "
        "Return a JSON object with keys: intent, order_reference, issue_summary."
    )


def build_direct_answer_prompt(state, context, workflow):
    return (
        "Answer the user's request directly and concisely. "
        "If the request is ambiguous, ask a clarifying question."
    )


def build_refund_response_prompt(state, context, workflow):
    extracted = state.require("refund_request")
    return (
        "You are preparing a support response for a refund workflow.\n"
        f"Structured request data: {extracted}\n"
        "Write a concise, user-facing response that confirms what information was captured "
        "and states the next step."
    )


workflow = WorkflowAgent(
    name="Support Workflow",
    steps=[
        RouterStep(
            name="route_request",
            func=route_request,
            possible_next_steps=["extract_refund_request", "compose_direct_answer"],
        ),
        LLMStep(
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
            next_step="compose_refund_response",
        ),
        LLMStep.final(
            name="compose_refund_response",
            llm=LLM(model=GPTModels.GPT_4O_MINI),
            prompt=build_refund_response_prompt,
            input_builder=lambda state, context, workflow: state.get("refund_request", {}),
        ),
        LLMStep.final(
            name="compose_direct_answer",
            llm=LLM(model=GPTModels.GPT_4O_MINI),
            prompt=build_direct_answer_prompt,
        ),
    ],
    start_step="route_request",
)


async def main():
    response = await workflow.async_ask("I want a refund for order 12345 because the product arrived damaged.")
    print(response)
