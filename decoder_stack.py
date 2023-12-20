# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""The decoder stack in inference mode."""

from typing import Any, Tuple

import gin
from transformer import decoder_stack
import transformer_layer as tl


struct = decoder_stack.struct
nn_components = decoder_stack.nn_components
position = decoder_stack.position
jnp = decoder_stack.jnp
attention = decoder_stack.attention

DStackWindowState = decoder_stack.DStackWindowState

Array = Any

TransformerTaskConfig = decoder_stack.TransformerTaskConfig

DStackDecoderState = Tuple[tl.DecoderState, ...]


@gin.configurable
class DecoderStackGenerate(decoder_stack.DecoderStack):
  """Stack of transformer decoder layers."""

  layer_factory = tl.TransformerLayerGenerate

  def init_decoder_state_vanilla(
      self, sequence_length: int, start_of_sequence: Array
  ) -> DStackDecoderState:
    """Return initial state for autoregressive generation."""
    return tuple(
        [
            layer.init_decoder_state_vanilla(sequence_length, start_of_sequence)
            for layer in self.transformer_layers
        ]
    )
