# Copyright 2020 The T5 Authors.
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

"""Tests for t5.data.feature_converters."""

from typing import Dict, Sequence
from unittest import mock
from t5.data import dataset_providers
from t5.data import feature_converters
from t5.data import test_utils
import tensorflow.compat.v2 as tf

tf.compat.v1.enable_eager_execution()

assert_dataset = test_utils.assert_dataset


def create_default_dataset(
    x: Sequence[Dict[str, int]],
    feature_names: Sequence[str] = ("inputs", "targets")) -> tf.data.Dataset:
  output_types = {feature_name: tf.int64 for feature_name in feature_names}
  output_shapes = {feature_name: [None] for feature_name in feature_names}

  ds = tf.data.Dataset.from_generator(
      lambda: x, output_types=output_types, output_shapes=output_shapes)
  return ds


class FeatureConvertersTest(tf.test.TestCase):

  def setUp(self):
    super(FeatureConvertersTest, self).setUp()
    default_vocab = test_utils.sentencepiece_vocab()
    self.output_features = {
        "inputs": dataset_providers.Feature(vocabulary=default_vocab),
        "targets": dataset_providers.Feature(vocabulary=default_vocab)
    }

  def test_validate_dataset_missing_feature(self):
    x = [{"targets": [3, 9, 4, 5]}]
    ds = create_default_dataset(x, feature_names=["targets"])

    with mock.patch.object(feature_converters.FeatureConverter,
                           "__abstractmethods__", set()):
      converter = feature_converters.FeatureConverter(self.output_features)
      expected_msg = ("Dataset is missing an expected feature during "
                      "initial validation: 'inputs'")
      with self.assertRaisesRegex(ValueError, expected_msg):
        converter.validate_dataset(
            ds,
            expected_features=self.output_features,
            expected_type=tf.int64,
            error_label="initial")

  def test_validate_dataset_incorrect_dtype(self):
    x = [{"inputs": [9, 4, 3, 8, 6], "targets": [3, 9, 4, 5]}]
    ds = tf.data.Dataset.from_generator(
        lambda: x, output_types={"inputs": tf.int32, "targets": tf.int64},
        output_shapes={"inputs": [None], "targets": [None]})

    with mock.patch.object(feature_converters.FeatureConverter,
                           "__abstractmethods__", set()):
      converter = feature_converters.FeatureConverter(self.output_features)
      expected_msg = ("Dataset has incorrect type for feature 'inputs' during "
                      "initial validation: Got int32, expected int64")
      with self.assertRaisesRegex(ValueError, expected_msg):
        converter.validate_dataset(
            ds,
            expected_features=self.output_features,
            expected_type=tf.int64,
            error_label="initial")

  def test_validate_dataset_incorrect_rank(self):
    x = [{"inputs": [[9, 4, 3, 8, 6]], "targets": [[3, 9, 4, 5]]}]
    ds = tf.data.Dataset.from_generator(
        lambda: x, output_types={"inputs": tf.int32, "targets": tf.int64},
        output_shapes={"inputs": [None, 1], "targets": [None, 1]})

    with mock.patch.object(feature_converters.FeatureConverter,
                           "__abstractmethods__", set()):
      converter = feature_converters.FeatureConverter(self.output_features)
      expected_msg = ("Dataset has incorrect type for feature 'inputs' during "
                      "initial validation: Got int32, expected int64")
      with self.assertRaisesRegex(ValueError, expected_msg):
        converter.validate_dataset(
            ds,
            expected_features=self.output_features,
            expected_type=tf.int64,
            expected_rank=1,
            error_label="initial")

  def test_validate_dataset_missing_length(self):
    x = [{"inputs": [9, 4, 3, 8, 6], "targets": [3, 9, 4, 5]}]
    ds = tf.data.Dataset.from_generator(
        lambda: x, output_types={"inputs": tf.int64, "targets": tf.int64},
        output_shapes={"inputs": [5], "targets": [5]})

    with mock.patch.object(feature_converters.FeatureConverter,
                           "__abstractmethods__", set()):
      converter = feature_converters.FeatureConverter(self.output_features)
      expected_msg = ("Sequence length for feature 'targets' is missing "
                      "during final validation")
      with self.assertRaisesRegex(ValueError, expected_msg):
        converter.validate_dataset(
            ds,
            expected_features=self.output_features,
            expected_type=tf.int64,
            check_length=True,
            sequence_length={"inputs": 5},
            error_label="final")

  def test_validate_dataset_incorrect_length(self):
    x = [{"inputs": [9, 4, 3, 8, 6], "targets": [3, 9, 4, 5]}]
    ds = tf.data.Dataset.from_generator(
        lambda: x, output_types={"inputs": tf.int64, "targets": tf.int64},
        output_shapes={"inputs": [5], "targets": [5]})

    with mock.patch.object(feature_converters.FeatureConverter,
                           "__abstractmethods__", set()):
      converter = feature_converters.FeatureConverter(self.output_features)
      expected_msg = (
          "The sequence length of feature 'inputs' does not match the requested"
          " sequence length during final validation: Got 5, expected 7")
      with self.assertRaisesRegex(ValueError, expected_msg):
        converter.validate_dataset(
            ds,
            expected_features=self.output_features,
            expected_type=tf.int64,
            check_length=True,
            sequence_length={"inputs": 7, "targets": 5},
            error_label="final")

  def test_validate_dataset_ensure_no_eos(self):
    x = [{"inputs": [9, 4, 3, 8, 1], "targets": [3, 9, 4, 5]}]
    ds = tf.data.Dataset.from_generator(
        lambda: x, output_types={"inputs": tf.int64, "targets": tf.int64},
        output_shapes={"inputs": [5], "targets": [4]})

    with mock.patch.object(feature_converters.FeatureConverter,
                           "__abstractmethods__", set()):
      expected_msg = (r".*Feature \\'inputs\\' unexpectedly contains EOS=1 "
                      r"token during initial validation.*")
      with self.assertRaisesRegex(
          tf.errors.InvalidArgumentError,
          expected_msg):
        converter = feature_converters.FeatureConverter(self.output_features)
        ds = converter.validate_dataset(
            ds,
            expected_features=self.output_features,
            expected_type=tf.int64,
            ensure_no_eos=True,
            error_label="initial")
        list(ds.as_numpy_iterator())

  def test_validate_dataset_plaintext_field(self):
    x = [{"targets": [3, 9, 4, 5], "targets_plaintext": "some text"}]
    output_types = {"targets": tf.int64, "targets_plaintext": tf.string}
    output_shapes = {"targets": [4], "targets_plaintext": []}
    ds = tf.data.Dataset.from_generator(
        lambda: x, output_types=output_types, output_shapes=output_shapes)

    # ds has targets and targets_plaintext but output_features only has targets
    default_vocab = test_utils.sentencepiece_vocab()
    output_features = {
        "targets": dataset_providers.Feature(vocabulary=default_vocab)
    }

    with mock.patch.object(feature_converters.FeatureConverter,
                           "__abstractmethods__", set()):
      converter = feature_converters.FeatureConverter(output_features)
      ds = converter.validate_dataset(
          ds,
          expected_features=output_features,
          expected_type=tf.int64,
          ensure_no_eos=True,
          error_label="initial")


if __name__ == "__main__":
  tf.test.main()
