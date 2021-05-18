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
import json
import os

import tensorflow as tf
from absl import logging
from tfx.dsl.io import fileio
from tfx.types import artifact_utils
from tfx.types import standard_artifacts

from tfx_x.components.examples.stratified_sampler import executor
from tfx_x.components.examples.stratified_sampler.executor import STRATIFIED_EXAMPLES_KEY, EXAMPLES_KEY, \
  SAMPLES_PER_KEY_KEY, TO_KEY_FN_KEY, SPLITS_TO_TRANSFORM_KEY, SPLITS_TO_COPY_KEY


class ExecutorTest(tf.test.TestCase):

  def setUp(self):
    super(ExecutorTest, self).setUp()
    self._source_data_dir = os.path.join(
      os.path.dirname(os.path.dirname(__file__)), 'testdata')
    self._output_data_dir = os.path.join(
      os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
      self._testMethodName)
    self.component_id = 'test_component'

    # Create input dict.
    self._examples = standard_artifacts.Examples()
    self._examples.uri = os.path.join(self._source_data_dir, 'csv_example_gen')

    self._examples.split_names = artifact_utils.encode_split_names(
      ['train', 'eval', 'unlabelled'])

    self._input_dict = {
      EXAMPLES_KEY: [self._examples],
    }

    # Create output dict.
    self._sampling_result = standard_artifacts.Examples()
    self._stratified_examples_dir = os.path.join(self._output_data_dir, "something")
    self._sampling_result.uri = self._stratified_examples_dir

    self._output_dict_sr = {
      STRATIFIED_EXAMPLES_KEY: [self._sampling_result],
    }

    # Create exe properties.
    self._exec_properties = {
      'component_id': self.component_id,
      SPLITS_TO_TRANSFORM_KEY: json.dumps(['eval']),
      SPLITS_TO_COPY_KEY: json.dumps([]),
      TO_KEY_FN_KEY: """
def to_key(m):  
  return m.features.feature['trip_miles'].float_list.value[0] > 42.
""",
      SAMPLES_PER_KEY_KEY: 1000,
    }

    # Create context
    self._tmp_dir = os.path.join(self._output_data_dir, '.temp')
    self._context = executor.Executor.Context(
      tmp_dir=self._tmp_dir, unique_id='2')

  def _get_results(self, path, file_name, proto_type):
    results = []
    filepattern = os.path.join(path, file_name) + '-?????-of-?????.gz'
    for f in fileio.glob(filepattern):
      record_iterator = tf.compat.v1.python_io.tf_record_iterator(
        path=f,
        options=tf.compat.v1.python_io.TFRecordOptions(
          tf.compat.v1.python_io.TFRecordCompressionType.GZIP))
      for record_string in record_iterator:
        stratified_examples = proto_type()
        stratified_examples.MergeFromString(record_string)
        results.append(stratified_examples)
    return results

  def _verify_stratified_example_split(self, split_name):
    dir_path = os.path.join(self._stratified_examples_dir, 'Split-' + split_name)
    logging.info("Looking for examples split in %s", dir_path)

    self.assertTrue(
      fileio.exists(dir_path))
    results = self._get_results(
      dir_path,
      executor._STRATIFIED_EXAMPLES_FILE_PREFIX, tf.train.Example)
    self.assertTrue(results)

  def _verify_copied_example_split(self, split_name):
    dir_path = os.path.join(self._stratified_examples_dir,  'Split-' + split_name)
    logging.info("Looking for examples split in %s", dir_path)

    self.assertTrue(
      fileio.exists(dir_path))

  def testDoWithOutputExamplesEvalSplit(self):
    self._exec_properties['splits_to_transform'] = json.dumps(['eval'])

    # Run executor.
    stratified_sampler = executor.Executor(self._context)
    stratified_sampler.Do(self._input_dict, self._output_dict_sr,
                          self._exec_properties)

    # Check outputs.
    self.assertTrue(fileio.exists(self._stratified_examples_dir))
    # self._verify_example_split('train')
    self.assertNotIn('train', artifact_utils.decode_split_names(self._sampling_result.split_names))
    self.assertIn('eval', artifact_utils.decode_split_names(self._sampling_result.split_names))
    self.assertLen(artifact_utils.decode_split_names(self._sampling_result.split_names), 1)
    self._verify_stratified_example_split('eval')

  def testDoWithOutputExamplesAllSplits(self):
    self._exec_properties[SPLITS_TO_TRANSFORM_KEY] = json.dumps(['eval', 'train'])

    # Run executor.
    stratified_sampler = executor.Executor(self._context)
    stratified_sampler.Do(self._input_dict, self._output_dict_sr,
                          self._exec_properties)

    # Check outputs.
    self.assertTrue(fileio.exists(self._stratified_examples_dir))
    self.assertIn('train', artifact_utils.decode_split_names(self._sampling_result.split_names))
    self.assertIn('eval', artifact_utils.decode_split_names(self._sampling_result.split_names))
    self.assertLen(artifact_utils.decode_split_names(self._sampling_result.split_names), 2)
    self._verify_stratified_example_split('train')
    self._verify_stratified_example_split('eval')

  def testDoWithOutputExamplesOneSplitSampledOneSplitCopied(self):
    self._exec_properties[SPLITS_TO_TRANSFORM_KEY] = json.dumps(['eval'])
    self._exec_properties[SPLITS_TO_COPY_KEY] = json.dumps(['train', 'unlabelled'])

    # Run executor.
    stratified_sampler = executor.Executor(self._context)
    stratified_sampler.Do(self._input_dict, self._output_dict_sr,
                          self._exec_properties)

    # Check outputs.
    self.assertTrue(fileio.exists(self._stratified_examples_dir))
    self.assertIn('train', artifact_utils.decode_split_names(self._sampling_result.split_names))
    self.assertIn('eval', artifact_utils.decode_split_names(self._sampling_result.split_names))
    self.assertIn('unlabelled', artifact_utils.decode_split_names(self._sampling_result.split_names))
    self.assertLen(artifact_utils.decode_split_names(self._sampling_result.split_names), 3)
    self._verify_copied_example_split('train')
    self._verify_copied_example_split('unlabelled')
    self._verify_stratified_example_split('eval')

  def testDoWithOutputExamplesTwoSplitsSampledOneSplitCopied(self):
    self._exec_properties[SPLITS_TO_TRANSFORM_KEY] = json.dumps(['train', 'eval'])
    self._exec_properties[SPLITS_TO_COPY_KEY] = json.dumps(['unlabelled'])

    # Run executor.
    stratified_sampler = executor.Executor(self._context)
    stratified_sampler.Do(self._input_dict, self._output_dict_sr,
                          self._exec_properties)

    # Check outputs.
    self.assertTrue(fileio.exists(self._stratified_examples_dir))
    self.assertIn('train', artifact_utils.decode_split_names(self._sampling_result.split_names))
    self.assertIn('eval', artifact_utils.decode_split_names(self._sampling_result.split_names))
    self.assertIn('unlabelled', artifact_utils.decode_split_names(self._sampling_result.split_names))
    self.assertLen(artifact_utils.decode_split_names(self._sampling_result.split_names), 3)
    self._verify_stratified_example_split('train')
    self._verify_copied_example_split('unlabelled')
    self._verify_stratified_example_split('eval')


if __name__ == '__main__':
  tf.test.main()
