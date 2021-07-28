# Lint as: python3
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
import importlib
import json
import os
from typing import Any, Dict, List, Text, Optional

import tensorflow as tf
from tfx import types
from tfx.components.pusher import executor as tfx_pusher_executor
from tfx.types import artifact_utils, standard_component_specs
from tfx.utils import io_utils

OUTPUT_KEY = 'output'
MODEL_KEY = 'model'
FUNCTION_NAME_KEY = 'function_name'
PIPELINE_CONFIGURATION_KEY = 'pipeline_configuration'


def noop(_model: tf.keras.Model, _pipeline_configuration: Dict[Text, Any], _output_dir: Text,
         _model_pushed_dir: Optional[Text],
         _model_pushed_artifact: Optional[types.Artifact],
         _transform_graph_artifact: Optional[types.Artifact]):
  pass


class Executor(tfx_pusher_executor.Executor):
  """Executor for Export."""

  def Do(self,
         input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    """Export a model with the provided function.

    ...

    Args:
      input_dict: Input dict from input key to a list of artifacts, including:
        - model: A list of type `standard_artifacts.Model`
        - pipeline_configuration: optional PipelineConfiguration artifact.
        - model_blessing: optional model blessing artifact.
        - infra_blessing: optional infra blessing artifact.
        - pushed_model: optional pushed model artifact.
        - transform_graph: optional transform graph artifact.
      output_dict: Output dict from key to a list of artifacts, including:
        - output: model export artifact.
      exec_properties: A dict of execution properties, including:
        - function_name: The name of the function to apply on the model - noop function is used if not specified.
        - instance_name: Optional unique instance_name. Necessary iff multiple Hello components
          are declared in the same pipeline.

    Returns:
      None

    Raises:
      OSError and its subclasses
      ValueError
    """
    self._log_startup(input_dict, output_dict, exec_properties)

    if not self.CheckBlessing(input_dict):
      return

    model = artifact_utils.get_single_instance(
      input_dict[MODEL_KEY])

    output = artifact_utils.get_single_instance(
      output_dict[OUTPUT_KEY])

    model_push_artifact = None
    if standard_component_specs.PUSHED_MODEL_KEY in input_dict:
      model_push_artifact = artifact_utils.get_single_instance(
        input_dict[standard_component_specs.PUSHED_MODEL_KEY])

    transform_graph_artifact = None
    if standard_component_specs.TRANSFORM_GRAPH_KEY in input_dict:
      transform_graph_artifact = artifact_utils.get_single_instance(
        input_dict[standard_component_specs.TRANSFORM_GRAPH_KEY])

    function_name = exec_properties.get(FUNCTION_NAME_KEY, 'tfx_x.components.model.export.executor.noop')

    pipeline_configuration = {}
    if PIPELINE_CONFIGURATION_KEY in input_dict:
      pipeline_configuration_dir = artifact_utils.get_single_uri(input_dict[PIPELINE_CONFIGURATION_KEY])
      pipeline_configuration_file = os.path.join(pipeline_configuration_dir, 'custom_config.json')
      pipeline_configuration_str = io_utils.read_string_file(pipeline_configuration_file)
      pipeline_configuration = json.loads(pipeline_configuration_str)

    # check if function_name can be found
    function_name_split = function_name.split('.')
    module_name = '.'.join(function_name_split[0:-1])
    module = importlib.import_module(module_name)

    fn = getattr(module, function_name_split[-1])

    if fn is None:
      raise ValueError('`function_name` not found')

    input_dir = artifact_utils.get_single_uri([model])
    output_dir = artifact_utils.get_single_uri([output])

    model_push_dir = None
    if model_push_artifact is not None:
      model_push_dir = artifact_utils.get_single_uri([model_push_artifact])

    # load the model
    model = tf.keras.models.load_model(os.path.join(input_dir, 'Format-Serving'))

    # export
    fn(model, pipeline_configuration, output_dir, model_push_dir, model_push_artifact, transform_graph_artifact)
