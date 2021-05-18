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
from tensorflow.python.saved_model.save_options import SaveOptions
from tfx import types
from tfx.dsl.components.base import base_executor
from tfx.types import artifact_utils
from tfx.utils import io_utils

OUTPUT_MODEL_KEY = 'output_model'
INPUT_MODEL_KEY = 'input_model'
FUNCTION_NAME_KEY = 'function_name'
PIPELINE_CONFIGURATION_KEY = 'pipeline_configuration'


def identity(model: tf.keras.Model, pipeline_configuration: Dict[Text, Any]) -> (
    tf.keras.Model, Dict[Text, Any], Optional[SaveOptions]):
  return model, None, None


class Executor(base_executor.BaseExecutor):
  """Executor for Transform."""

  def Do(self,
         input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    """Transform a model with the provided function.

    ...

    Args:
      input_dict: Input dict from input key to a list of artifacts, including:
        - input_model: A list of type `standard_artifacts.Model`
        - pipeline_configuration: optional PipelineConfiguration artifact.
      output_dict: Output dict from key to a list of artifacts, including:
        - output_model: A list of type `standard_artifacts.Model`
      exec_properties: A dict of execution properties, including:
        - function_name: The name of the function to apply on the model - identity function is used if not specified.
        - instance_name: Optional unique instance_name. Necessary iff multiple Hello components
          are declared in the same pipeline.

    Returns:
      None

    Raises:
      OSError and its subclasses
      ValueError
    """
    self._log_startup(input_dict, output_dict, exec_properties)

    input_model = artifact_utils.get_single_instance(
      input_dict[INPUT_MODEL_KEY])
    output_model = artifact_utils.get_single_instance(
      output_dict[OUTPUT_MODEL_KEY])
    function_name = exec_properties.get(FUNCTION_NAME_KEY, 'tfx_x.components.model.transform.executor.identity')

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

    input_dir = artifact_utils.get_single_uri([input_model])
    output_dir = artifact_utils.get_single_uri([output_model])

    # load the model
    model = tf.keras.models.load_model(os.path.join(input_dir, 'Format-Serving'))

    # transform
    new_model, signatures, options = fn(model, pipeline_configuration)

    # save the model
    tf.saved_model.save(model, os.path.join(output_dir, 'Format-Serving'), signatures, options)
