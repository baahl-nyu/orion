from .orion import scheme
from .mlflow_logger import MLflowLogger, get_logger, set_logger

init_scheme = scheme.init_scheme
delete_scheme = scheme.delete_scheme
encode = scheme.encode
decode = scheme.decode
encrypt = scheme.encrypt
decrypt = scheme.decrypt
fit = scheme.fit
compile = scheme.compile