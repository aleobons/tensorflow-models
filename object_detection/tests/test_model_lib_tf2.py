"""Tests for object detection model library."""

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import os

# import tempfile
# import unittest
import numpy as np
import mlflow
from pathlib import Path

# import six
import tensorflow.compat.v1 as tf
import tensorflow.compat.v2 as tf2

# from object_detection import exporter_lib_v2
# from object_detection import inputs
from object_detection import model_lib_v2

from object_detection.core import model

# from object_detection.protos import train_pb2
from object_detection.utils import config_util

# from object_detection.utils import tf_version

# if six.PY2:
#     import mock  # pylint: disable=g-importing-member,g-import-not-at-top
# else:
#     from unittest import mock  # pylint: disable=g-importing-member,g-import-not-at-top

# Model for test. Current options are:
# 'ssd_mobilenet_v2_pets_keras'
MODEL_NAME_FOR_TEST = "ssd_mobilenet_v2_pets_keras"
ROOT_PACKAGE = "./object_detection"


def _get_data_path(val_suffix=""):
    """Returns an absolute path to TFRecord file."""
    return os.path.join(ROOT_PACKAGE, "test_data", f"pets_examples{val_suffix}.record")


def get_pipeline_config_path(model_name):
    """Returns path to the local pipeline config file."""
    return os.path.join(
        ROOT_PACKAGE,
        "samples",
        "configs",
        model_name + ".config",
    )


def _get_labelmap_path():
    """Returns an absolute path to label map file."""
    return os.path.join(ROOT_PACKAGE, "data", "pet_label_map.pbtxt")


def _get_config_kwarg_overrides():
    """Returns overrides to the configs that insert the correct local paths."""
    data_path_train = _get_data_path()
    data_path_val = _get_data_path(val_suffix="_val")
    label_map_path = _get_labelmap_path()
    return {
        "train_input_path": data_path_train,
        "eval_input_path": data_path_val,
        "label_map_path": label_map_path,
        "train_input_reader": {"batch_size": 1},
    }


def test_train_loop(sample_tracking_config):
    """Tests that Estimator and input function are constructed correctly."""
    tf.keras.backend.clear_session()

    # Given
    model_dir = tf.test.get_temp_dir()
    pipeline_config_path = get_pipeline_config_path(MODEL_NAME_FOR_TEST)
    new_pipeline_config_path = os.path.join(model_dir, "new_pipeline.config")
    config_util.clear_fine_tune_checkpoint(
        pipeline_config_path, new_pipeline_config_path
    )
    config_kwarg_overrides = _get_config_kwarg_overrides()
    model_output = "./tests/test_outputs/model_output"
    metrics_json = "./tests/test_outputs/metrics.json"
    summary_file_path = "./tests/test_outputs/tensorboard"

    train_steps = 2
    # strategy = tf2.distribute.MirroredStrategy(["/cpu:0", "/cpu:1"])
    strategy = tf2.compat.v2.distribute.MirroredStrategy()

    # When
    with strategy.scope():
        model_lib_v2.train_loop(
            pipeline_config_path=new_pipeline_config_path,
            model_dir=model_dir,
            input_model_type="encoded_image_string_tensor",
            model_output=model_output,
            train_steps=train_steps,
            checkpoint_every_n=1,
            output_metrics=metrics_json,
            summary_file_path=summary_file_path,
            num_steps_per_iteration=1,
            tracking_config=sample_tracking_config,
            input_train=None,  # TODO criar teste para input_train e input_val
            input_val=None,
            **config_kwarg_overrides,
        )

    # Then
    client = mlflow.tracking.MlflowClient()
    artifact_uri = client.list_run_infos(experiment_id="0")[-1].artifact_uri

    model_pb = os.path.join(model_output, "saved_model.pb")
    # model_artifact_pb = os.path.join(artifact_uri, "saved_model", "saved_model.pb")
    pipeline_artifact = os.path.join(artifact_uri, "new_pipeline.config")

    tensorboard_train = os.path.join(summary_file_path, "train")
    tensorboard_eval = os.path.join(summary_file_path, "eval")

    assert os.path.exists(model_pb)
    assert os.path.exists(Path(pipeline_artifact))
    assert os.path.exists(metrics_json)
    assert os.path.exists(tensorboard_train)
    assert os.path.exists(tensorboard_eval)


class SimpleModel(model.DetectionModel):
    """A model with a single weight vector."""

    def __init__(self, num_classes=1):
        super(SimpleModel, self).__init__(num_classes)
        self.weight = tf.keras.backend.variable(np.ones(10), name="weight")

    def postprocess(self, prediction_dict, true_image_shapes):
        return {}

    def updates(self):
        return []

    def restore_map(self, *args, **kwargs):
        pass

    def restore_from_objects(self, fine_tune_checkpoint_type):
        return {"model": self}

    def preprocess(self, _):
        return tf.zeros((1, 128, 128, 3)), tf.constant([[128, 128, 3]])

    def provide_groundtruth(self, *args, **kwargs):
        pass

    def predict(self, pred_inputs, true_image_shapes):
        return {
            "prediction": tf.abs(
                tf.reduce_sum(self.weight) * tf.reduce_sum(pred_inputs)
            )
        }

    def loss(self, prediction_dict, _):
        return {"loss": tf.reduce_sum(prediction_dict["prediction"])}

    def regularization_losses(self):
        return []


def fake_model_builder(*_, **__):
    return SimpleModel()


# FAKE_BUILDER_MAP = {"detection_model_fn_base": fake_model_builder}


# @unittest.skipIf(tf_version.is_tf1(), "Skipping TF2.X only test.")
# class ModelCheckpointTest(tf.test.TestCase):
#     """Test for model checkpoint related functionality."""


# def test_checkpoint_max_to_keep():
#     """Test that only the most recent checkpoints are kept."""

#     strategy = tf2.distribute.OneDeviceStrategy(device="/cpu:0")
#     with mock.patch.dict(model_lib_v2.MODEL_BUILD_UTIL_MAP, FAKE_BUILDER_MAP):

#         model_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
#         pipeline_config_path = get_pipeline_config_path(MODEL_NAME_FOR_TEST)
#         new_pipeline_config_path = os.path.join(model_dir, "new_pipeline.config")
#         config_util.clear_fine_tune_checkpoint(
#             pipeline_config_path, new_pipeline_config_path
#         )
#         config_kwarg_overrides = _get_config_kwarg_overrides()

#         with strategy.scope():
#             model_lib_v2.train_loop(
#                 new_pipeline_config_path,
#                 model_dir=model_dir,
#                 train_steps=5,
#                 checkpoint_every_n=2,
#                 checkpoint_max_to_keep=3,
#                 num_steps_per_iteration=1,
#                 input_model_type="encoded_image_string_tensor",
#                 model_output="model",
#                 **config_kwarg_overrides
#             )
#         ckpt_files = tf.io.gfile.glob(os.path.join(model_dir, "ckpt-*.index"))
#         self.assertEqual(len(ckpt_files), 3, "{} not of length 3.".format(ckpt_files))


# class IncompatibleModel(SimpleModel):
#     def restore_from_objects(self, *args, **kwargs):
#         return {"weight": self.weight}


# @unittest.skipIf(tf_version.is_tf1(), "Skipping TF2.X only test.")
# class CheckpointV2Test(tf.test.TestCase):
#     def setUp(self):
#         super(CheckpointV2Test, self).setUp()

#         self._model = SimpleModel()
#         tf.keras.backend.set_value(self._model.weight, np.ones(10) * 42)
#         ckpt = tf.train.Checkpoint(model=self._model)

#         self._test_dir = tf.test.get_temp_dir()
#         self._ckpt_path = ckpt.save(os.path.join(self._test_dir, "ckpt"))
#         tf.keras.backend.set_value(self._model.weight, np.ones(10))

#         pipeline_config_path = get_pipeline_config_path(MODEL_NAME_FOR_TEST)
#         configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
#         configs = config_util.merge_external_params_with_configs(
#             configs, kwargs_dict=_get_config_kwarg_overrides()
#         )
#         self._train_input_fn = inputs.create_train_input_fn(
#             configs["train_config"], configs["train_input_config"], configs["model"]
#         )

#     def test_restore_v2(self):
#         """Test that restoring a v2 style checkpoint works."""

#         model_lib_v2.load_fine_tune_checkpoint(
#             self._model,
#             self._ckpt_path,
#             checkpoint_type="",
#             checkpoint_version=train_pb2.CheckpointVersion.V2,
#             run_model_on_dummy_input=True,
#             input_dataset=self._train_input_fn(),
#             unpad_groundtruth_tensors=True,
#         )
#         np.testing.assert_allclose(self._model.weight.numpy(), 42)

#     def test_restore_map_incompatible_error(self):
#         """Test that restoring an incompatible restore map causes an error."""

#         with self.assertRaisesRegex(
#             TypeError, r".*received a \(str -> ResourceVariable\).*"
#         ):
#             model_lib_v2.load_fine_tune_checkpoint(
#                 IncompatibleModel(),
#                 self._ckpt_path,
#                 checkpoint_type="",
#                 checkpoint_version=train_pb2.CheckpointVersion.V2,
#                 run_model_on_dummy_input=True,
#                 input_dataset=self._train_input_fn(),
#                 unpad_groundtruth_tensors=True,
#             )


# if __name__ == "__main__":
#     tf.test.main()
