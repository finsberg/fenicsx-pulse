import warnings

# Print deprecation warning if the user imports pulse directly
warnings.warn(
    "Importing fenicsx_pulse directly is deprecated. Please use 'import pulse' instead",
    category=DeprecationWarning,
    stacklevel=2,
)
from pulse import *  # noqa: F403
