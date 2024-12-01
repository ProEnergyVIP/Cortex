import os

is_debug = bool(os.environ.get('DEBUG', False))
is_production = not is_debug
