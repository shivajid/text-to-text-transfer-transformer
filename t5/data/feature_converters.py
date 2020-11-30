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

"""Feature converters for common architectures.

In short, feature converters are mapping from "task features" to "model
features" where the former refers to the features output by the Task API
(typically "inputs" and "outputs") and the latter refers to the features
specific to the model architecture (typically standard features defined below).

We provide feature converters for the following three architectures:

  - encoder-decoder
  - decoder-only
  - encoder-only

Each of these feature converters inherit the base class FeatureConverter and
override two methods _convert_features and get_model_feature_sequence_length to
define how task features are mapped to the model features including the length
relationships.


Definition: standard_features

Throughout this module, we refer to the following 10 fields as
standard_features. Depending on the model architecture, a subset of them will be
returned by the feature converter.

  - encoder_input_token
  - encoder_target_token
  - encoder_loss_weight
  - encoder_position
  - encoder_segment_id
  - decoder_input_token
  - decoder_target_token
  - decoder_loss_weight
  - decoder_position
  - decoder_segment_id

  *_segment_id and *_position are only relevant for packed dataset.

  *_segment_id is an tf.Tensor of integer which is aligned with
  encoder_input_token. Positive integers represent the sequence membership in
  the packed examples. 0 represents padding. For example, encoder_segment_id =
  [1, 1, 2, 2, 2, 0] means that the first two positions belong to the first
  sequence, the next three to the second sequence and the last position is a
  padding.

  *_loss_weight is used to indicate which positions should be used for the loss
  calculation.


Underlying assumptions

The feature converters implemented in this module assume the following about the
input dataset.

  - EOS tokens are not appended (i.e., the output of the Task API).
  - The input dataset is not batched.

For use-cases not covered by these standard cases, users need to define their
own feature-converter.
"""
import abc
import collections
from typing import Mapping, Union, List
# TODO(hwchung): transformer_dataset does not depend on mesh_tensorflow at all.
# move/modify/adapt the pack_or_pad to t5
from t5.data import dataset_providers
import tensorflow.compat.v2 as tf


class FeatureConverter(abc.ABC):
  """Abstract base class for feature converters.

  Subclasses of FeatureConverter are used to convert from "task features" to
  "model features". The former refers to the features output from the Task API.
  Typically they are "inputs" and "outputs". The model features are more
  descriptive features that are passed to the model implementation. Typically,
  they are standard features, which are defined in the module docstring.

  This conversion is fully specified by

    1. defining the mapping of the features in `_convert_features` method and
    2. defining the relationship between sequence lengths of task features and
      model features in `get_model_feature_sequence_length` which is a function
      of sequence_length (i.e., the sequence length of the task features).

  Therefore, a subclass of FeatureConverter should override _convert_features
  and get_model_feature_sequence_length methods.

  The actual feature conversion is done in the `convert_features` method, which
  wraps around the `_convert_features` method in order to provide useful checks
  and ensure compatibilities. See validate_dataset and convert_features methods
  for more details.

  Other notes:

    The input dataset to the feature converter should not have the EOS tokens
    appended. If EOS tokens are desired, this should be done in the
    `_convert_features` of each derived class. We provide a helper method
    `trim_and_ensure_eos` for that purpose.

    `output_features` is a terminology used in the Task API. This can be
    confusing because output_features is an input to the FeatureConverter.
    Therefore, we interchangeably use the term task features to mean
    output_features.

    This class also provides helper methods such as validating the dataset
    properties as well as useful properties.

    If pack = True, each feature in the output_features should be packable,
    i.e., 1-dimensional.

  Attributes:
    output_features: features of the input dataset to be converted. Also
      corresponds output_features from the Task API.
    pack: whether to pack the dataset.
    input_dtype: input data type, typically int64, for compatiblility with
      tf.Example proto.
    output_dtype: input data type, typically int32, for compatibility with TPUs.
  """

  def __init__(self,
               output_features: Mapping[str, dataset_providers.Feature],
               pack: bool = True,
               input_dtype: tf.dtypes.DType = tf.int64,
               output_dtype: tf.dtypes.DType = tf.int32):
    self._output_features = collections.OrderedDict(
        sorted(list(output_features.items()))
    )
    self._pack = pack
    self._input_dtype = input_dtype
    self._output_dtype = output_dtype

  def validate_dataset(
      self,
      ds: tf.data.Dataset,
      expected_features: Union[Mapping[str, dataset_providers.Feature],
                               List[str]],
      expected_type: tf.DType,
      error_label: str,
      ensure_no_eos: bool = False,
      check_length: bool = False,
      expected_rank: int = 1,
      sequence_length: Mapping[str, int] = None) -> tf.data.Dataset:
    """Generate inputs for an autoregressive model, by shifting the targets.

    Expanded from t5.data.dataset_providers.TaskV3._validate_dataset.

    This method is used to validate whether the input dataset is compatiable
    with the desired specifications. In particular, the following aspects are
    checked.

    Each feature in `expected_features`
      - is also in `ds`
      - has self.output_dtype
      - has rank of 1
      - is also in model_feature_sequence_length
      - has matching length in sequence_length

    The last two are optional, controlled by `check_length` arg. The last one
    only works if the sequence has a length dimension defined. For example, the
    output dataset of the Task API's get_dataset method typically has [None]
    shape, i.e., the length is not defined yet. In this case, the last check is
    skipped.

    Args:
      ds: a tf.data.Dataset to be validated
      expected_features: expected features either in Mapping or List format.
      expected_type: expected data type of each feature
      error_label: a label used to indicate the validation stage
      ensure_no_eos: whether to ensure that each feature does not contain the
        EOS id anywhere.
      check_length: whether to check the length of each feature
      expected_rank: expected rank of each feature
      sequence_length: a mapping from feature to its length

    Returns:
      ds: the same dataset as the inpu as the input. Internally, the TensorFlow
        graph may contain additional nodes if ensure_no_eos is set to True.
    """
    element_spec = ds.element_spec
    for feat in expected_features:
      if feat not in element_spec:
        raise ValueError("Dataset is missing an expected feature during "
                         f"{error_label} validation: '{feat}'")

      if expected_type != element_spec[feat].dtype:
        actual_dtype = element_spec[feat].dtype.name
        raise ValueError(
            f"Dataset has incorrect type for feature '{feat}' during "
            f"{error_label} validation: "
            f"Got {actual_dtype}, expected {expected_type.name}")

      if expected_rank != len(element_spec[feat].shape):
        actual_rank = len(element_spec[feat].shape)
        raise ValueError(
            f"Dataset has incorrect rank for feature '{feat}' during "
            f"{error_label} validation: "
            f"Got {actual_rank}, expected {expected_rank}")

      if check_length:
        if expected_rank == 0:
          raise ValueError(
              "If check_length is True, expected rank should be greater than 0."
          )

        if sequence_length is None:
          raise ValueError(
              "If check_length is True, sequence_length should be specified.")

        if feat not in sequence_length:
          raise ValueError(f"Sequence length for feature '{feat}' is missing "
                           f"during {error_label} validation")

        # At this point, the rank of each feature is strictly greater than 0.
        actual_length = element_spec[feat].shape.as_list()[0]
        # Prior to padding, the length is None. Skip the check.
        if actual_length is not None and sequence_length[feat] != actual_length:
          raise ValueError(
              f"The sequence length of feature '{feat}' does not match the "
              f"requested sequence length during {error_label} validation: "
              f"Got {actual_length}, expected {sequence_length[feat]}")

    def _ensure_no_eos(feat, v):
      if feat not in expected_features:
        return v
      error_message = (f"Feature '{feat}' unexpectedly contains EOS=1 token "
                       f"during {error_label} validation")
      with tf.control_dependencies([
          tf.debugging.assert_none_equal(
              v, tf.constant(1, tf.int64), message=error_message)
      ]):
        return v

    if ensure_no_eos:
      ds = ds.map(
          lambda ex: {k: _ensure_no_eos(k, v) for k, v in ex.items()})

    return ds

  def trim_and_ensure_eos(
      self,
      ds: tf.data.Dataset,
      sequence_length: Mapping[str, int]
    ) -> tf.data.Dataset:
    """Trim and append EOS=1 token to model features."""
    def _trim_and_append_eos(feat, v):
      if feat not in self.output_features:
        return v

      if sequence_length and self.output_features[feat].add_eos:
        v = tf.concat([v[:sequence_length[feat]-1], [1]], axis=0)
      elif sequence_length:
        v = v[:sequence_length[feat]]
      elif self.output_features[feat].add_eos:
        v = tf.concat([v, [1]], axis=0)
      return v

    return ds.map(
        lambda ex: {k: _trim_and_append_eos(k, v) for k, v in ex.items()},
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

  def convert_features(self, ds: tf.data.Dataset,
                       sequence_length: Mapping[str, int]) -> tf.data.Dataset:
    """This method should not be overridden by subclasses.

    In the initial validation
      - Each feature in self.output_features also in `ds`
      - Each feature in self.output_features has self.input_dtype
      - Each feature in self.output_features has rank of 1
      - Each feature in self.output_features also in sequence_length

    In the final validation
      - Each feature in model_features also in (output) ds
      - Each feature in model_features has self.output_dtype
      - Each feature in model_features has rank of 1
      - Each feature in model_features also in model_feature_sequence_length
      - Each feature in model_features has matching length as in
        model_feature_sequence_length.

    Therefore, the input dataset and the output dataset are compatible in terms
    of expected fields and length.

    Args:
      ds: a tf.data.Dataset to be validated
      sequence_length: a mapping from feature to its length

    Returns:
      ds: the converted dataset.
    """
    # Initial validation stage
    ds = self.validate_dataset(
        ds,
        expected_features=self.output_features,
        expected_type=self.input_dtype,
        error_label="initial",
        check_length=True,
        sequence_length=sequence_length,
        ensure_no_eos=True)

    # Main feature conversion, implemented by subclasses
    ds = self._convert_features(ds, sequence_length)

    model_feature_sequence_length = self.get_model_feature_sequence_length(
        sequence_length)
    # Final validation stage
    ds = self.validate_dataset(
        ds,
        expected_features=model_feature_sequence_length.keys(),
        expected_type=self.output_dtype,
        error_label="final",
        check_length=True,
        sequence_length=model_feature_sequence_length)

    return ds

  @abc.abstractmethod
  def _convert_features(self, ds: tf.data.Dataset,
                        sequence_length: Mapping[str, int]) -> tf.data.Dataset:
    """Main feature conversion method to be overridden.."""
    raise NotImplementedError

  @abc.abstractmethod
  def get_model_feature_sequence_length(
      self, sequence_length: Mapping[str, int]) -> Mapping[str, int]:
    """Define the length relationship between task and model features."""
    raise NotImplementedError

  @property
  def output_features(self) -> Mapping[str, dataset_providers.Feature]:
    return self._output_features

  @property
  def pack(self) -> bool:
    return self._pack

  @property
  def input_dtype(self) -> tf.dtypes.DType:
    return self._input_dtype

  @property
  def output_dtype(self) -> tf.dtypes.DType:
    return self._output_dtype
