# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Creates and runs TF2 object detection models.

For local training/evaluation run:
PIPELINE_CONFIG_PATH=path/to/pipeline.config
MODEL_DIR=/tmp/model_outputs
NUM_TRAIN_STEPS=10000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
python model_main_tf2.py -- \
  --model_dir=$MODEL_DIR --num_train_steps=$NUM_TRAIN_STEPS \
  --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
  --pipeline_config_path=$PIPELINE_CONFIG_PATH \
  --alsologtostderr
"""
import json
import os
from absl import flags
import tensorflow.compat.v2 as tf
import tensorflow.compat.v1 as tf1

from object_detection import model_lib_v2
from object_detection import core as config


tf.get_logger().setLevel("ERROR")

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

flags.DEFINE_string("pipeline_config_path", None, "Path to pipeline config " "file.")
flags.DEFINE_integer("num_train_steps", None, "Number of train steps.")
flags.DEFINE_bool(
    "eval_on_train_data",
    False,
    "Enable evaluating on train " "data (only supported in distributed training).",
)
flags.DEFINE_integer(
    "sample_1_of_n_eval_examples",
    None,
    "Will sample one of " "every n eval input examples, where n is provided.",
)
flags.DEFINE_integer(
    "sample_1_of_n_eval_on_train_examples",
    5,
    "Will sample "
    "one of every n train input examples for evaluation, "
    "where n is provided. This is only used if "
    "`eval_training_data` is True.",
)
flags.DEFINE_string(
    "model_dir",
    None,
    "Path to output model directory "
    "where event and checkpoint files will be written.",
)
flags.DEFINE_string(
    "checkpoint_dir",
    None,
    "Path to directory holding a checkpoint.  If "
    "`checkpoint_dir` is provided, this binary operates in eval-only mode, "
    "writing resulting metrics to `model_dir`.",
)

flags.DEFINE_integer(
    "eval_timeout",
    3600,
    "Number of seconds to wait for an" "evaluation checkpoint before exiting.",
)

flags.DEFINE_bool("use_tpu", False, "Whether the job is executing on a TPU.")
flags.DEFINE_string(
    "tpu_name", default=None, help="Name of the Cloud TPU for Cluster Resolvers."
)
flags.DEFINE_integer(
    "num_workers",
    1,
    "When num_workers > 1, training uses "
    "MultiWorkerMirroredStrategy. When num_workers = 1 it uses "
    "MirroredStrategy.",
)
flags.DEFINE_integer(
    "checkpoint_every_n", 1000, "Integer defining how often we checkpoint."
)
flags.DEFINE_boolean(
    "record_summaries",
    True,
    (
        "Whether or not to record summaries defined by the model"
        " or the training pipeline. This does not impact the"
        " summaries of the loss values which are always"
        " recorded."
    ),
)
flags.DEFINE_integer(
    "checkpoint_max_to_keep",
    7,
    ("the number of most recent checkpoints to keep in the model directory"),
)

flags.DEFINE_string("summary_file_path", None, ("tensorboard logs location"))
flags.DEFINE_string(
    "pipeline_config_override_path", None, "Path to pipeline config override file"
)

flags.DEFINE_string("input_train", None, "Path to input train data")
flags.DEFINE_string("input_train_pattern", "", "Pattern for input train files")

flags.DEFINE_string("input_val", None, "Path to input val data")
flags.DEFINE_string("input_val_pattern", "", "Pattern for input train files")

flags.DEFINE_string(
    "input_model_type",
    "encoded_image_string_tensor",
    "Type of input data in score model",
)
flags.DEFINE_string("model_output", "model", "Path to model output")

flags.DEFINE_string("tracking_config_path", None, "Path to tracking config")


FLAGS = flags.FLAGS


def main(unused_argv):
    flags.mark_flag_as_required("model_dir")
    flags.mark_flag_as_required("pipeline_config_path")
    flags.mark_flag_as_required("input_train")
    flags.mark_flag_as_required("input_val")
    flags.mark_flag_as_required("model_output")

    tf.config.set_soft_device_placement(True)

    tracking_config = None
    if FLAGS.tracking_config_path is not None:
        tracking_config = config.create_and_validate_config(
            parsed_config_path=FLAGS.tracking_config_path
        )

    if FLAGS.checkpoint_dir:
        if FLAGS.pipeline_config_override_path:
            with tf1.gfile.GFile(FLAGS.pipeline_config_override_path, "r") as f:
                config_override = f.read()
        else:
            config_override = None

        model_lib_v2.eval_continuously(
            pipeline_config_path=FLAGS.pipeline_config_path,
            model_dir=FLAGS.model_dir,
            train_steps=FLAGS.num_train_steps,
            sample_1_of_n_eval_examples=FLAGS.sample_1_of_n_eval_examples,
            sample_1_of_n_eval_on_train_examples=(
                FLAGS.sample_1_of_n_eval_on_train_examples
            ),
            checkpoint_dir=FLAGS.checkpoint_dir,
            wait_interval=300,
            timeout=FLAGS.eval_timeout,
            use_tpu=FLAGS.use_tpu,
            summary_file_path=FLAGS.summary_file_path,
            config_override=config_override,
        )
    else:
        if FLAGS.use_tpu:
            # TPU is automatically inferred if tpu_name is None and
            # we are running under cloud ai-platform.
            print("[INFO] using TPU...")
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver(FLAGS.tpu_name)
            tf.config.experimental_connect_to_cluster(resolver)
            tf.tpu.experimental.initialize_tpu_system(resolver)
            strategy = tf.distribute.TPUStrategy(resolver)
        elif FLAGS.num_workers > 1:
            print("[INFO] using MultiWorkerMirroredStrategy...")
            strategy = tf.distribute.MultiWorkerMirroredStrategy()
        else:
            strategy = tf.compat.v2.distribute.MirroredStrategy()

        def export(data, _):
            with open("outputs/metrics.json", "w") as f:
                json.dump(data, f)

        input_train = (
            os.path.join(FLAGS.input_train, FLAGS.input_train_pattern)
            if os.path.isdir(FLAGS.input_train)
            else FLAGS.input_train
        )

        input_val = (
            os.path.join(FLAGS.input_val, FLAGS.input_val_pattern)
            if os.path.isdir(FLAGS.input_val)
            else FLAGS.input_val
        )

        with strategy.scope():
            model_lib_v2.train_loop(
                pipeline_config_path=FLAGS.pipeline_config_path,
                model_dir=FLAGS.model_dir,
                input_model_type=FLAGS.input_model_type,
                model_output=FLAGS.model_output,
                train_steps=FLAGS.num_train_steps,
                use_tpu=FLAGS.use_tpu,
                checkpoint_every_n=FLAGS.checkpoint_every_n,
                record_summaries=FLAGS.record_summaries,
                checkpoint_max_to_keep=FLAGS.checkpoint_max_to_keep,
                summary_file_path=FLAGS.summary_file_path,
                performance_summary_exporter=export,
                input_train=input_train,
                input_val=input_val,
                pipeline_config_override_path=FLAGS.pipeline_config_override_path,
                tracking_config=tracking_config,
            )


if __name__ == "__main__":
    tf.compat.v1.app.run()
