To see the logs from the `neuroalign.modeling` modules in your Jupyter notebook (or any interactive Python session), you need to configure the logging system *before* you run any code that uses the loggers.

I've created a utility function `configure_logging` for this purpose.

Please add the following code to the very first cell of your notebook and run it:

```python
from neuroalign.utils import configure_logging
import logging

# Configure logging to display INFO level messages (or DEBUG for more verbose output)
configure_logging(level="INFO")

# You can also set it to "DEBUG" to see all debug messages from the modules.
# configure_logging(level="DEBUG")

# Example: If you have a specific logger you want to adjust, you can do:
# logging.getLogger("neuroalign.modeling.univariate.estimator").setLevel(logging.DEBUG)

print("Logging has been configured!")
```

After running this cell, when you execute `estimator.fit_predict()`, you should start seeing the log messages in the notebook's output.

If you still don't see logs after doing this, please let me know.