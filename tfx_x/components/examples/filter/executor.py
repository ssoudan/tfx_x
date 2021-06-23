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

"""TFX stratified_sampler executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
from typing import Any, Dict, Mapping, List, Text

import apache_beam as beam
import tensorflow as tf
from absl import logging
from tfx import types
from tfx.dsl.components.base import base_beam_executor
from tfx.types import artifact_utils, Artifact
from tfx.utils import io_utils, json_utils

from tfx_x.components import utils

FILTERED_EXAMPLES_KEY = 'filtered_examples'
EXAMPLES_KEY = 'examples'
PREDICATE_FN_KEY = 'predicate_fn'
SPLITS_TO_COPY_KEY = 'splits_to_copy'
SPLITS_TO_TRANSFORM_KEY = 'splits_to_transform'
PIPELINE_CONFIGURATION_KEY = 'pipeline_configuration'
PREDICATE_FN_KEY_KEY = 'predicate_fn_key'

_FILTERED_EXAMPLES_FILE_PREFIX = 'filtered_examples'
_FILTERED_EXAMPLES_DIR_NAME = 'filtered_examples'


class Executor(base_beam_executor.BaseBeamExecutor):
  """TFX stratified sampler executor."""

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    """Runs stratified sampling on given input examples.
    Args:
      input_dict: Input dict from input key to a list of Artifacts.
        - examples: examples for inference.
        - pipeline_configuration: optional PipelineConfiguration artifact.
      output_dict: Output dict from output key to a list of Artifacts.
        - filtered_examples: the stratified examples.
      exec_properties: A dict of execution properties.
        - splits_to_transform: list of splits to transform.
        - splits_to_copy: list of splits to copy as is.
        - predicate_fn: the function defines if a sample must be kept - must be 'predicate: Example -> bool
        - predicate_fn_key: alternate name for the key containing the def of `predicate()`
    Returns:
      None
    """
    self._log_startup(input_dict, output_dict, exec_properties)

    examples = input_dict[EXAMPLES_KEY]

    # Priority is as follow:
    # 1. default value
    # 2. from PipelineConfiguration
    # 3. from exec_properties

    splits_to_transform = []
    predicate_fn = None

    predicate_fn_key = exec_properties[
      PREDICATE_FN_KEY_KEY] if PREDICATE_FN_KEY_KEY in exec_properties else PREDICATE_FN_KEY

    splits_to_copy = artifact_utils.decode_split_names(
      artifact_utils.get_single_instance(examples).split_names)

    if PIPELINE_CONFIGURATION_KEY in input_dict:
      pipeline_configuration_dir = artifact_utils.get_single_uri(input_dict[PIPELINE_CONFIGURATION_KEY])
      pipeline_configuration_file = os.path.join(pipeline_configuration_dir, 'custom_config.json')
      pipeline_configuration_str = io_utils.read_string_file(pipeline_configuration_file)
      pipeline_configuration = json.loads(pipeline_configuration_str)

      if SPLITS_TO_TRANSFORM_KEY in pipeline_configuration:
        splits_to_transform = pipeline_configuration[SPLITS_TO_TRANSFORM_KEY]
      else:
        splits_to_transform = []

      if SPLITS_TO_COPY_KEY in pipeline_configuration:
        splits_to_copy = pipeline_configuration[SPLITS_TO_COPY_KEY]

      if predicate_fn_key in pipeline_configuration:
        predicate_fn = pipeline_configuration[predicate_fn_key]

    # Now looking at the exec_properties
    if SPLITS_TO_TRANSFORM_KEY in exec_properties and exec_properties[SPLITS_TO_TRANSFORM_KEY] is not None:
      splits_to_transform = json_utils.loads(exec_properties[SPLITS_TO_TRANSFORM_KEY])

    if SPLITS_TO_COPY_KEY in exec_properties and exec_properties[SPLITS_TO_COPY_KEY] is not None:
      splits_to_copy = json_utils.loads(exec_properties[SPLITS_TO_COPY_KEY])

    if PREDICATE_FN_KEY in exec_properties and exec_properties[PREDICATE_FN_KEY] is not None:
      predicate_fn = exec_properties[PREDICATE_FN_KEY]

    if predicate_fn_key in exec_properties and exec_properties[predicate_fn_key] is not None:
      predicate_fn = exec_properties[predicate_fn_key]

    # Validate we have all we need
    if predicate_fn is None:
      raise ValueError('\'predicate_fn\' is missing in exec dict.')

    if EXAMPLES_KEY not in input_dict:
      raise ValueError('\'examples\' is missing in input dict.')

    if FILTERED_EXAMPLES_KEY not in output_dict:
      raise ValueError('\'filtered_examples\' is missing in output dict.')

    output_artifact = artifact_utils.get_single_instance(output_dict[FILTERED_EXAMPLES_KEY])
    output_artifact.split_names = artifact_utils.encode_split_names(splits_to_transform + splits_to_copy)

    example_uris = {}

    for split in splits_to_transform:
      data_uri = artifact_utils.get_split_uri(examples, split)
      example_uris[split] = data_uri

    # do something with the splits we dont want to transform ('splits_to_copy')
    utils.copy_over(examples, output_artifact, splits_to_copy)

    self._run_filtering(example_uris,
                        output_artifact=output_artifact,
                        predicate_fn=predicate_fn)

    logging.info('Filter generates filtered examples to %s', output_artifact.uri)

  def _run_filtering(self,
                     example_uris: Mapping[Text, Text],
                     predicate_fn: Text,
                     output_artifact: Artifact) -> None:
    """Runs stratified sampling on given example data.
    Args:
      example_uris: Mapping of example split name to example uri.
      predicate_fn: function to decide if a example must be kept.
      output_artifact: Output artifact.
    Returns:
      None
    """

    d = {}
    exec(predicate_fn, globals(), d)  # how ugly is that?
    predicate = d['predicate']

    with self._make_beam_pipeline() as pipeline:
      for split_name, example_uri in example_uris.items():
        data_list = [(
            pipeline | 'ReadData[{}]'.format(split_name) >> beam.io.ReadFromTFRecord(
          file_pattern=io_utils.all_files_pattern(example_uri)))]

        dest_path = os.path.join(artifact_utils.get_split_uri([output_artifact], split_name),
                                 _FILTERED_EXAMPLES_FILE_PREFIX)

        _ = (
            [data for data in data_list]
            | 'FlattenExamples ({})'.format(split_name) >> beam.Flatten(pipeline=pipeline)
            | 'ParseExamples ({})'.format(split_name) >> beam.Map(tf.train.Example.FromString)
            | 'Filter ({})'.format(split_name) >> beam.Filter(predicate)
            | 'WriteStratifiedSamples ({})'.format(split_name) >> beam.io.WriteToTFRecord(
          dest_path,
          file_name_suffix='.gz',
          coder=beam.coders.ProtoCoder(tf.train.Example)))
        logging.info('Sampling result written to %s.', dest_path)
