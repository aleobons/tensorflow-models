from pathlib import Path
import re
import os
from configparser import ConfigParser
from typing import List, Dict

from pydantic import BaseModel, constr, validator
from strictyaml import load, YAML

from object_detection.utils import config_util


# instantiate
config = ConfigParser()

losses_allowed = [
    "Loss/total_loss",
    "Loss/localization_loss",
    "Loss/regularization_loss",
    "Loss/classification_loss",
]

metrics_allowed = [
    "DetectionBoxes_Precision/mAP",
    "DetectionBoxes_Precision/mAP (large)",
    "DetectionBoxes_Precision/mAP (medium)",
    "DetectionBoxes_Precision/mAP (small)",
    "DetectionBoxes_Precision/mAP@.50IOU",
    "DetectionBoxes_Precision/mAP@.75IOU",
    "DetectionBoxes_Recall/AR@1",
    "DetectionBoxes_Recall/AR@10",
    "DetectionBoxes_Recall/AR@100",
    "DetectionBoxes_Recall/AR@100 (large)",
    "DetectionBoxes_Recall/AR@100 (medium)",
    "DetectionBoxes_Recall/AR@100 (small)",
]


def path_pipeline_config_validator(path_pipeline_config: str) -> str:

    if re.match(
        "^\./([a-z0-9\-\_]+/)*[a-zA-Z0-9\-\_]+\.config$",
        path_pipeline_config,
    ):
        return path_pipeline_config

    raise ValueError(f"Invalid path pipeline config: {path_pipeline_config}")


class TrackingConfig(BaseModel):
    """
    Tracking-level config.
    """

    # the order is necessary for validation
    losses_allowed: List[str]
    metrics_allowed: List[str]

    path_pipeline_config: str
    _validator_path_pipeline_config = validator(
        "path_pipeline_config", pre=True, allow_reuse=True
    )(path_pipeline_config_validator)

    parametros: List[
        constr(regex="^[a-z][a-z0-9\-\_]*(\.[a-z0-9\-\_]*)+([a-z0-9\-\_])+$")
    ]
    metricas: Dict[constr(regex="^[/\w.\- ]*$"), str]
    losses: Dict[str, str]

    # TODO - checar se a métrica existe
    metrica_para_avaliacao: str

    @validator("parametros")
    def validate_parametros(cls, params, values):
        def _validate_message_has_field(message, field):
            if not message.HasField(field):
                raise ValueError("Expecting message to have field %s" % field)

        path_pipeline_config = Path(values.get("path_pipeline_config"))
        if not os.path.exists(path_pipeline_config):
            raise ValueError(f"Path pipeline config not found: {path_pipeline_config}")

        pipeline_config = config_util.get_configs_from_pipeline_file(
            path_pipeline_config
        )

        for param in params:
            fields = param.split(".")
            first_field = fields.pop(0)
            message = pipeline_config[first_field]

            for field in fields:
                _validate_message_has_field(message, field)
                message = getattr(message, field)

        return params

    @validator("losses")
    def allowed_losses(cls, losses, values):

        allowed_loss = values.get("losses_allowed")

        for loss in losses.values():
            if loss not in allowed_loss:
                raise ValueError(
                    f"the loss parameter specified: {loss}, "
                    f"is not in the allowed set: {allowed_loss}"
                )
        return losses

    @validator("metricas")
    def allowed_metrics(cls, metrics, values):

        allowed_metrics = values.get("metrics_allowed")

        for metric in metrics.values():
            if metric not in allowed_metrics:
                raise ValueError(
                    f"the metric parameter specified: {metric}, "
                    f"is not in the allowed set: {allowed_metrics}"
                )
        return metrics


class Config(BaseModel):
    """Master config object."""

    tracking_config: TrackingConfig


def fetch_config_from_yaml(cfg_path: Path) -> YAML:
    """Parse YAML containing the package configuration."""

    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")


def create_and_validate_config(
    parsed_config_path: Path, parsed_config: YAML = None
) -> Config:
    """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml(parsed_config_path)

    parsed_config_dict = parsed_config.data
    parsed_config_dict.update(
        {"losses_allowed": losses_allowed, "metrics_allowed": metrics_allowed}
    )

    # specify the data attribute from the strictyaml YAML type.
    _config = Config(
        tracking_config=TrackingConfig(**parsed_config_dict),
    )

    return _config
