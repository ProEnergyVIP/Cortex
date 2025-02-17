from dataclasses import dataclass
import json
from typing import Any
from intellifun.message import SystemMessage, UserMessage, UserVisionMessage, print_message
from intellifun.debug import is_debug

@dataclass
class CheckResult:
    success: bool
    value: Any = None
    message: str = None

    @classmethod
    def ok(cls, value):
        '''Return a success CheckResult and the given value'''
        return cls(success=True, value=value)

    @classmethod
    def fail(cls, message):
        '''Return a failure CheckResult with the given message'''
        return cls(success=False, message=message)


JSON_FORMAT = '''
Your answer MUST CONFORM to this JSON format:

{shape}


Ensure your answer is a valid JSON object that can be parsed with Python's 
json.loads() function successfully.
DO NOT include the JSON schema itself in the output, only the JSON object conforming to the schema.
DO NOT include the `json` tag in your answer.
'''

def llmfunc(llm, prompt, result_shape=None, check_func=None, max_attempts=3, llm_args={}):
    '''Create a new LLMFunc, which is a LLM based intelligent function
    
    This function is used to create a new LLMFunc, which is a function that uses an LLM to
    resolve some task. A LLM model and system prompt is needed to create the function. 
    Optionally, a result shape can be provided to ensure the result is in the correct format.

    The result function will take a string query as input and return the result of the LLM call.
    The final result will be just a string, or a JSON object if a result shape is provided.

    If a check function is provided, the result will be checked against the check function.
    The check function should return a CheckResult object, which can be used to determine if the
    result is correct or not. If the check failed, the LLMFunc will run again with the error
    message provided by the check function. Once check passes, the result of the check function
    will be returned as the final result. It will retry up to max_attempts times.

    Args:
    llm (LLM): The LLM instance to use
    prompt (str): The system prompt to send to the LLM
    result_shape (dict): The expected shape of the result, as a JSON parameter object
    check_func (function): A function to check the result, should return a CheckResult
    max_attempts (int): The maximum number of attempts to make, default is 3

    Returns:
    function: The LLMFunc function
    '''
    if result_shape:
        prompt += JSON_FORMAT.format(shape=json.dumps(result_shape))

    sys_msg = SystemMessage(content=prompt)

    if is_debug:
        print_message(sys_msg)

    def func(msg, image_urls=None):
        if isinstance(msg, UserMessage):
            user_msg = msg
        elif image_urls:
            user_msg = UserVisionMessage(content=msg, image_urls=image_urls)
        else:
            user_msg = UserMessage(content=msg)
        
        msgs = [user_msg]

        history = []

        for i in range(max_attempts):
            if check_func:
                msgs.extend(history)

            if is_debug:
                for msg in msgs:
                    print_message(msg)

            ai_msg = llm.call(sys_msg, msgs, **llm_args)

            if is_debug:
                print_message(ai_msg)
            
            result = check_result(ai_msg, result_shape, check_func)

            if result.success:
                return result.value

            if not check_func or i == max_attempts - 1:
                return result.message

            history.append(ai_msg)
            msg = UserMessage(content=result.message)
            history.append(msg)
    return func


def check_result(ai_msg, result_shape=None, check_func=None):
    try:
        answer = json.loads(ai_msg.content) if result_shape else ai_msg.content
    except json.JSONDecodeError as e:
        return CheckResult.fail(str(e))
    
    if not check_func:
        return CheckResult.ok(answer)
    
    try:
        return check_func(answer)
    except Exception as e:
        return CheckResult.fail(str(e))
