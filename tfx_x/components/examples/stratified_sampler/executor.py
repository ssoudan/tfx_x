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

import os
from typing import Any, Dict, Mapping, List, Text

import apache_beam as beam
import tensorflow as tf
from absl import logging
from tfx import types
from tfx.dsl.components.base import base_executor
from tfx.types import artifact_utils, Artifact
from tfx.utils import io_utils

from tfx_x.components import utils

SAMPLING_RESULT_KEY = 'sampling_result'
EXAMPLES_KEY = 'examples'
COUNT_PER_KEY_KEY = 'count_per_key'
THRESHOLD_KEY = 'threshold'
KEY_KEY = 'key'
SPLITS_TO_COPY_KEY = 'splits_to_copy'
SPLITS_TO_TRANSFORM_KEY = 'splits_to_transform'

_STRATIFIED_EXAMPLES_FILE_PREFIX = 'stratified_examples'
_STRATIFIED_EXAMPLES_DIR_NAME = 'stratified_examples'


class Executor(base_executor.BaseExecutor):
  """TFX stratified sampler executor."""

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    """Runs stratified sampling on given input examples.
    Args:
      input_dict: Input dict from input key to a list of Artifacts.
        - examples: examples for inference.
      output_dict: Output dict from output key to a list of Artifacts.
        - output: the stratified examples.
      exec_properties: A dict of execution properties.
        - splits_to_transform: list of splits to transform.
        - splits_to_copy: list of splits to copy as is.
        - key: the feature to use as key (must be a float)
        - threshold: the threshold to use
        - count_per_key: the number samples per classes
    Returns:
      None
    """
    self._log_startup(input_dict, output_dict, exec_properties)

    examples = input_dict[EXAMPLES_KEY]

    if SPLITS_TO_TRANSFORM_KEY in exec_properties:
      splits_to_transform = exec_properties[SPLITS_TO_TRANSFORM_KEY]
    else:
      splits_to_transform = []

    if SPLITS_TO_COPY_KEY in exec_properties:
      splits_to_copy = exec_properties[SPLITS_TO_COPY_KEY]
    else:
      splits_to_copy = artifact_utils.decode_split_names(
        artifact_utils.get_single_instance(examples).split_names)

    if KEY_KEY in exec_properties:
      key = exec_properties[KEY_KEY]
    else:
      raise ValueError('\'key\' is missing in input dict.')

    if THRESHOLD_KEY in exec_properties:
      threshold = exec_properties[THRESHOLD_KEY]
    else:
      threshold = 0.5

    if COUNT_PER_KEY_KEY in exec_properties:
      count_per_key = exec_properties[COUNT_PER_KEY_KEY]
    else:
      raise ValueError('\'count_per_key\' is missing in input dict.')

    if EXAMPLES_KEY not in input_dict:
      raise ValueError('\'examples\' is missing in input dict.')
    if SAMPLING_RESULT_KEY not in output_dict:
      raise ValueError('\'sampling_result\' is missing in output dict.')
    output_artifact = artifact_utils.get_single_instance(output_dict[SAMPLING_RESULT_KEY])
    output_artifact.split_names = artifact_utils.encode_split_names(splits_to_transform + splits_to_copy)

    example_uris = {}

    for split in splits_to_transform:
      data_uri = artifact_utils.get_split_uri(examples, split)
      example_uris[split] = data_uri

    # do something with the splits we dont want to transform ('splits_to_copy')
    utils.copy_over(examples, output_artifact, splits_to_copy)

    self._run_sampling(example_uris, key=key, output_artifact=output_artifact, count_per_key=count_per_key,
                       threshold=threshold)

    logging.info('StratifiedSampler generates stratified examples to %s', output_artifact.uri)

  def _run_sampling(self,
                    example_uris: Mapping[Text, Text],
                    key: Text,
                    output_artifact: Artifact,
                    count_per_key: int,
                    threshold: float = 0.5) -> None:
    """Runs stratified sampling on given example data.
    Args:
      example_uris: Mapping of example split name to example uri.
      key: feature used for the stratification.
      output_artifact: Output artifact.
      count_per_key: number of examples to keep per value of the key.
      threshold: threshold to convert the feature in a bool.
    Returns:
      None
    """
    with self._make_beam_pipeline() as pipeline:
      for split_name, example_uri in example_uris.items():
        data_list = [(
            pipeline | 'ReadData[{}]'.format(split_name) >> beam.io.ReadFromTFRecord(
          file_pattern=io_utils.all_files_pattern(example_uri)))]

        dest_path = os.path.join(artifact_utils.get_split_uri([output_artifact], split_name),
                                 _STRATIFIED_EXAMPLES_FILE_PREFIX)

        _ = (
            [data for data in data_list]
            | 'FlattenExamples ({})'.format(split_name) >> beam.Flatten(pipeline=pipeline)
            | 'ParseExamples ({})'.format(split_name) >> beam.Map(tf.train.Example.FromString)
            | 'Key ({})'.format(split_name) >> beam.Map(
          lambda m: (m.features.feature[key].float_list.value[0] > threshold, m))
            | 'Sample per key ({})'.format(split_name) >> beam.combiners.Sample.FixedSizePerKey(count_per_key)
            | 'Values ({})'.format(split_name) >> beam.Values()
            | 'Flatten lists ({})'.format(split_name) >> beam.FlatMap(lambda elements: elements)
            | 'WriteStratifiedSamples ({})'.format(split_name) >> beam.io.WriteToTFRecord(
          dest_path,
          file_name_suffix='.gz',
          coder=beam.coders.ProtoCoder(tf.train.Example)))
    logging.info('Sampling result written to %s.', dest_path)
