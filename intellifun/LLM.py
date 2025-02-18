import random
from threading import Lock

from intellifun.backend import LLMBackend, LLMRequest

class LLM:
    '''default LLM based on OpenAI's API'''
    # Class-level state for backup backends and failed models
    _backup_backends = {}  # Maps model -> backup_model
    _failed_models = set()  # Set of models that have failed
    _runtime_lock = Lock()  # Lock for protecting runtime state changes

    @classmethod
    def _detect_cycle(cls, start_model: str, backup_model: str) -> bool:
        '''Check if adding this backup would create a cycle in the backup chain'''
        # First check if this would create an immediate cycle
        if start_model == backup_model:
            return True
            
        # Then check for indirect cycles
        current = backup_model
        visited = {start_model}
        
        # Follow the backup chain
        while current in cls._backup_backends:
            if current in visited:  # We've seen this model before - cycle detected
                return True
            visited.add(current)
            current = cls._backup_backends[current]
            
            # Also check if this would complete a cycle back to our start
            if current == start_model:
                return True
        
        return False

    @classmethod
    def set_backup_backend(cls, model: str, backup_model: str) -> None:
        '''Set a backup backend model to use if the primary model fails.
        
        Args:
            model: The primary model identifier
            backup_model: The backup model identifier to use if primary fails
            
        Raises:
            ValueError: If setting this backup would create a cycle in the backup chain
        '''
        if cls._detect_cycle(model, backup_model):
            raise ValueError(
                f"Cannot set {backup_model} as backup for {model} as it would create a cycle in the backup chain"
            )
        
        cls._backup_backends[model] = backup_model

    @classmethod
    def clear_backup_backends(cls) -> None:
        '''Clear all backup backend configurations and reset failed models'''
        with cls._runtime_lock:  # Need lock here as it affects runtime state
            cls._backup_backends.clear()
            cls._failed_models.clear()

    @classmethod
    def _get_effective_model(cls, model: str) -> str:
        '''Get the effective model to use, considering failures and backups.
        Note: This method can be called without a lock for initialization,
        but should be called with a lock when switching models during runtime.
        '''
        if model in cls._failed_models and model in cls._backup_backends:
            return cls._backup_backends[model]
        return model

    def __init__(self, model, temperature = 0.5):
        self.model = model
        self.temperature = temperature
        self._initialize_backend()

    def _initialize_backend(self):
        '''Initialize the backend using the effective model.
        Note: This method assumes the caller holds _runtime_lock
        '''
        effective_model = self._get_effective_model(self.model)
        self.backend = LLMBackend.get_backend(effective_model)

    def call(self, sys_msg, msgs, max_tokens=None, tools=None):
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
                            tools=tools or [],
                            )
            return self.backend.call(req)
        except Exception as e:
            # Check if we can switch to backup
            should_retry = False
            with self._runtime_lock:  # Minimize lock holding time
                if self.model not in self._failed_models and self.model in self._backup_backends:
                    # Mark this model as failed and switch backend
                    self._failed_models.add(self.model)
                    self._initialize_backend()
                    should_retry = True

            # Retry with new backend if we switched
            if should_retry:
                return self.call(sys_msg, msgs, max_tokens, tools)
            
            raise  # Re-raise if no backup available or already using backup

def get_random_error_message():
    '''Get a random error message'''
    return random.choice(err_msgs)

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
