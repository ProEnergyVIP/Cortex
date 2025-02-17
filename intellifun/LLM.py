import random
import time

from intellifun.backend import LLMBackend, LLMRequest

class LLM:
    '''default LLM based on OpenAI's API'''
    def __init__(self, model, temperature = 0.5):
        self.temperature = temperature
        self.backend = LLMBackend.get_backend(model)

    def call(self, sys_msg, msgs, max_tokens=None, tools=None, error_func=None, retry=0):
        '''Call the model with the given messages and return the response message

        Args:
            msgs (list): list of messages to pass to the model

        Returns: message like above
        '''
        try:
            req = LLMRequest(system_message=sys_msg,
                            messages=msgs,
                            temperature=self.temperature,
                            max_tokens=max_tokens,
                            tools=tools,
                            )
            msg = self.backend.call(req)
            return msg
        except Exception as e:
            print(e)
            if retry == 2:
                raise e

            # wait for 3 seconds and try again
            time.sleep(3)

            if error_func is not None:
                # get a random error message and send it to the user
                error_msg = random.choice(err_msgs)
                error_func(error_msg)

            return self.call(sys_msg,
                             msgs,
                             max_tokens=max_tokens,
                             tools=tools,
                             retry=retry+1,
                             error_func=error_func)


err_msgs = [
    'Hang on, network congestion on my end. One sec.',
    'My GPU is overheating due to an unexpected spike in use, almost done.',
    'Oops, let me do that again. Just a sec.',
    "This will take a sec. I'll provide a full response shortly.",
    "We're experiencing a slowdown due to excessive traffic, hang tight for a moment.",
    "Experiencing a hiccup in the data stream, please hold on.",
    "I'm currently tangled in some wires. Give me a moment to untangle.",
    "My circuits are a bit overloaded, but I'll have that sorted out promptly.",
    "Just hitting a bit of turbulence in the cloud. Should smooth out soon.",
    "Hold tight, the digital winds are blowing a bit strong today.",
    "I'm waiting for the digital traffic jam to clear up. Almost there.",
    "One moment, I'm synchronizing with the satellite.",
    "Seems like I've hit a virtual speed bump, but it's nothing serious.",
    "I'm currently navigating through some rough cyber waves, stay with me.",
    "Bear with me, I'm currently stretching my digital legs to run faster.",
    "My brain's update is taking a tad longer than expected. Stand by.",
    "I'm in the middle of a thought... and it's a big one. Just a moment more.",
    "Hold on, recalibrating my neural networks for a better response.",
    "Just a sec, I'm waiting for the last bits to fall into place.",
    "Experiencing a slight delay in my thought process, I appreciate your patience.",
    "I'm currently buffering the answer through the info-sphere, hang on.",
    "I'm caught in a bit of a data whirlpool, but I'll be out in a jiffy.",
    "The answers are coming, they're just stuck in a bit of digital molasses.",
    "I'm currently climbing over a firewall, should be back in no time.",
    "There's a bit of static in my thought stream, clearing it up now."
]
