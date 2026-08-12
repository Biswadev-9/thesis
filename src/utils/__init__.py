from src.utils.instantiators import instantiate_callbacks, instantiate_loggers
from src.utils.logging_utils import log_hyperparameters
from src.utils.pylogger import RankedLogger
from src.utils.rich_utils import enforce_tags, print_config_tree
from src.utils.serialization import allow_project_checkpoint_globals
from src.utils.utils import extras, get_metric_value, task_wrapper

# Applied on import so every entry point (train, eval, analyze, extract_features) can
# reload this project's checkpoints under torch >= 2.6's weights_only default.
allow_project_checkpoint_globals()
