from dataclasses import dataclass
import json
import logging
from typing import Any, List
from cortex.message import InputFile, InputImage, SystemMessage, UserMessage

logger = logging.getLogger(__name__)

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

def _prepare_messages(msg, image_urls=None, file_urls=None):
    '''Prepare the user message for the LLM call'''
    if isinstance(msg, List):
        return msg
    elif isinstance(msg, UserMessage):
        return [msg]
    
    if image_urls:
        imgs = [InputImage(image_url=url) for url in image_urls]
        return [UserMessage(content=msg, images=imgs)]
    elif file_urls:
        files = [InputFile(file_url=url) for url in file_urls]
        return [UserMessage(content=msg, files=files)]
    else:
        return [UserMessage(content=msg)]

def _handle_logging(msgs, ai_msg, usage):
    '''Handle logging of messages and usage'''
    for msg in msgs:
        logger.info(msg.decorate())
    logger.info(ai_msg.decorate())
    
    logger.info(ai_msg.usage.format())
    
    if usage and ai_msg.model and ai_msg.usage:
        usage.add_usage(ai_msg.model, ai_msg.usage)

def llmfunc(llm, prompt, result_shape=None, check_func=None, max_attempts=3, llm_args=None, logging_config=None, async_mode=False):
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
    logging_config (LoggingConfig): Configuration for message logging
    async_mode (bool): If True, returns an async function that uses the LLM's async_call method

    Returns:
    function: The LLMFunc function (async if async_mode is True)
    '''
    if result_shape:
        prompt += JSON_FORMAT.format(shape=json.dumps(result_shape))

    sys_msg = SystemMessage(content=prompt)

    logging_config = logging_config

    logger.debug(sys_msg.decorate())
    
    llm_args = llm_args or {}

    if async_mode:
        async def func(msg, image_urls=None, file_urls=None, usage=None):
            msgs = _prepare_messages(msg, image_urls, file_urls)
            history = []

            for i in range(max_attempts):
                if check_func:
                    msgs.extend(history)

                for msg in msgs:
                    logger.info(msg.decorate())

                ai_msg = await llm.async_call(sys_msg, msgs, **llm_args)
                
                _handle_logging(msgs, ai_msg, usage)
                
                result = check_result(ai_msg, result_shape, check_func)

                if result.success:
                    return result.value

                if not check_func or i == max_attempts - 1:
                    return result.message

                history.append(ai_msg)
                msg = UserMessage(content=result.message)
                history.append(msg)
        return func
    else:
        def func(msg, image_urls=None, file_urls=None, usage=None):
            msgs = _prepare_messages(msg, image_urls, file_urls)
            history = []

            for i in range(max_attempts):
                if check_func:
                    msgs.extend(history)

                for msg in msgs:
                    logger.info(msg.decorate())

                ai_msg = llm.call(sys_msg, msgs, **llm_args)
                
                _handle_logging(msgs, ai_msg, usage)
                
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
