#  Copyright 2021 ssoudan. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

# Lint as: python2, python3
# Copyright 2019 Google LLC. All Rights Reserved.
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
"""MNIST handwritten digit classification example using TFX."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import List, Text, Any, Dict

import absl
import tensorflow_model_analysis as tfma
from tfx.components import Evaluator
from tfx.components import ExampleValidator
from tfx.components import ImportExampleGen
from tfx.components import Pusher
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Trainer
from tfx.components import Transform
from tfx.components.trainer.executor import GenericExecutor
from tfx.dsl.components.base import executor_spec
from tfx.orchestration import metadata
from tfx.orchestration import pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2

from tfx_x.components import Filter
from tfx_x.components.configuration.converter.component import FromCustomConfig
from tfx_x.components.examples.stratified_sampler.component import StratifiedSampler

_pipeline_name = 'mnist_native_keras'

# This example assumes that MNIST data is stored in ~/mnist/data and the utility
# function is in ~/mnist. Feel free to customize as needed.
_mnist_root = os.path.join(os.environ['HOME'], 'mnist')
_data_root = os.path.join(_mnist_root, '../../data/data')
# Python module files to inject customized logic into the TFX components. The
# Transform and Trainer both require user-defined functions to run successfully.
_module_file = os.path.join(_mnist_root, 'mnist_utils_native_keras.py')

# Path which can be listened to by the model server. Pusher will output the
# trained model here.
_serving_model_dir = os.path.join(_mnist_root, 'serving_model', _pipeline_name)
_serving_model_dir_lite = os.path.join(
  _mnist_root, 'serving_model_lite', _pipeline_name)

# Directory and data locations.  This example assumes all of the images,
# example code, and metadata library is relative to $HOME, but you can store
# these files anywhere on your local filesystem.
_tfx_root = os.path.join(os.environ['HOME'], 'tfx')
_pipeline_root = os.path.join(_tfx_root, 'pipelines', _pipeline_name)
# Sqlite ML-metadata db path.
_metadata_path = os.path.join(_tfx_root, 'metadata', _pipeline_name,
                              'metadata.db')

# Pipeline arguments for Beam powered Components.
_beam_pipeline_args = [
  '--direct_running_mode=multi_processing',
  # 0 means auto-detect based on on the number of CPUs available
  # during execution time.
  '--direct_num_workers=0',
]


def _create_pipeline(pipeline_name: Text, pipeline_root: Text, data_root: Text,
                     custom_config: Dict[Text, Any],
                     module_file: Text,
                     serving_model_dir: Text,
                     metadata_path: Text,
                     beam_pipeline_args: List[Text]) -> pipeline.Pipeline:
  """Implements the handwritten digit classification example using TFX."""
  # Store the configuration along with the pipeline run so results can be reproduced
  pipeline_configuration = FromCustomConfig(custom_config=custom_config)

  # Brings data into the pipeline.
  example_gen = ImportExampleGen(input_base=data_root)

  # Computes statistics over data for visualization and example validation.
  statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])

  # Generates schema based on statistics files.
  schema_gen = SchemaGen(
    statistics=statistics_gen.outputs['statistics'], infer_feature_shape=True)

  # Performs anomaly detection based on statistics and data schema.
  example_validator = ExampleValidator(
    statistics=statistics_gen.outputs['statistics'],
    schema=schema_gen.outputs['schema'])

  # Create a filtered dataset - today we only want a model for small digits
  filter = Filter(examples=example_gen.outputs['examples'],
                  pipeline_configuration=pipeline_configuration.outputs[
                    'pipeline_configuration'],
                  splits_to_transform=['train', 'eval'],
                  splits_to_copy=[])

  # Create a stratified dataset for evaluation
  stratified_examples = StratifiedSampler(examples=filter.outputs['filtered_examples'],
                                          pipeline_configuration=pipeline_configuration.outputs[
                                            'pipeline_configuration'],
                                          samples_per_key=1200,
                                          splits_to_transform=['eval'],
                                          splits_to_copy=['train'])

  # Performs transformations and feature engineering in training and serving.
  transform = Transform(
    examples=filter.outputs['filtered_examples'],
    schema=schema_gen.outputs['schema'],
    module_file=module_file)

  # Uses user-provided Python function that trains a Keras model.
  trainer = Trainer(
    module_file=module_file,
    custom_executor_spec=executor_spec.ExecutorClassSpec(GenericExecutor),
    custom_config=custom_config,
    examples=transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],
    schema=schema_gen.outputs['schema'],
    train_args=trainer_pb2.TrainArgs(num_steps=5000),
    eval_args=trainer_pb2.EvalArgs(num_steps=100),
    instance_name='mnist')

  # Uses TFMA to compute evaluation statistics over features of a model and
  # performs quality validation of a candidate model.
  eval_config = tfma.EvalConfig(
    model_specs=[tfma.ModelSpec(label_key='image_class')],
    slicing_specs=[tfma.SlicingSpec()],
    metrics_specs=[
      tfma.MetricsSpec(metrics=[
        tfma.MetricConfig(
          class_name='SparseCategoricalAccuracy',
          threshold=tfma.config.MetricThreshold(
            value_threshold=tfma.GenericValueThreshold(
              lower_bound={'value': 0.8})))
      ])
    ])

  # Uses TFMA to compute the evaluation statistics over features of a model.
  evaluator = Evaluator(
    examples=stratified_examples.outputs['stratified_examples'],
    model=trainer.outputs['model'],
    eval_config=eval_config,
    instance_name='mnist')

  # Checks whether the model passed the validation steps and pushes the model
  # to a file destination if check passed.
  pusher = Pusher(
    model=trainer.outputs['model'],
    model_blessing=evaluator.outputs['blessing'],
    push_destination=pusher_pb2.PushDestination(
      filesystem=pusher_pb2.PushDestination.Filesystem(
        base_directory=serving_model_dir)),
    instance_name='mnist')

  return pipeline.Pipeline(
    pipeline_name=pipeline_name,
    pipeline_root=pipeline_root,
    components=[
      pipeline_configuration,
      example_gen,
      filter,
      stratified_examples,
      statistics_gen,
      schema_gen,
      example_validator,
      transform,
      trainer,
      evaluator,
      pusher,
    ],
    enable_cache=True,
    metadata_connection_config=metadata.sqlite_metadata_connection_config(
      metadata_path),
    beam_pipeline_args=beam_pipeline_args)


# To run this pipeline from the python CLI:
#   $python mnist_pipeline_native_keras.py
if __name__ == '__main__':
  absl.logging.set_verbosity(absl.logging.INFO)

  to_key_fn = """
def to_key(m):
  return m.features.feature['trip_miles'].float_list.value[0] > 21.
"""

  predicate_fn = """
def predicate(m):  
  return m.features.feature['trip_miles'].float_list.value[0] < 42.
"""

  _custom_config = {'layer_count': 3,
                    'to_key_fn': to_key_fn,
                    'predicate_fn': predicate_fn}

  BeamDagRunner().run(
    _create_pipeline(
      pipeline_name=_pipeline_name,
      pipeline_root=_pipeline_root,
      data_root=_data_root,
      custom_config=_custom_config,
      module_file=_module_file,
      serving_model_dir=_serving_model_dir,
      metadata_path=_metadata_path,
      beam_pipeline_args=_beam_pipeline_args))
