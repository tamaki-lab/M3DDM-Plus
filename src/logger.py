from datetime import datetime
from pytorch_lightning.loggers import CometLogger
from comet_ml import Experiment


def configure_logger(
    logged_params: dict,
    model_name: str,
    disable_logging: bool = False,
    lightning: bool = False,
):
    """Factory for Comet logging.

    Returns CometLogger for Lightning or Experiment for scripts.
    """
    # Experiment name prefixed with model_name
    ts = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    exp_name = f"{model_name}_{ts}"
    if lightning:
        logger = CometLogger(
            name=exp_name,
            disabled=disable_logging,
        )
        logger.experiment.add_tag(model_name)
        logger.experiment.log_parameters(logged_params)
        return logger

    # Not using Comet
    if disable_logging:
        class Dummy:
            def __getattr__(self, name):
                return lambda *args, **kwargs: None

        return Dummy()

    exp = Experiment()
    exp.set_name(exp_name)
    exp.add_tag(model_name)
    exp.log_parameters(logged_params)
    return exp
