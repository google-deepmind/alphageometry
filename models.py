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

"""Transformer language model generate mode."""

from typing import Any, Tuple
import beam_search
import decoder_stack
import gin
import jax
import jax.numpy as jnp
from transformer import models


@gin.configurable
class DecoderOnlyLanguageModelGenerate(models.DecoderOnlyLanguageModel):
  """Decoder only language modeling in inference mode."""

  decoder_factory = decoder_stack.DecoderStackGenerate

  num_heads: int = gin.REQUIRED
  head_size: int = gin.REQUIRED

  def get_fake_input(self) -> dict[str, Any]:
    fake_input_dict = super().get_fake_input()
    b = self.task_config.batch_size
    n = self.num_heads
    h = self.head_size
    fake_input_dict.update({
        'dstate': tuple(
            [{
                'current_index': jnp.array([0] * b, dtype=jnp.int32),
                'keys': jnp.zeros((b, 2048, n, h), dtype=jnp.bfloat16),
                'values': jnp.zeros((b, 2048, n, h), dtype=jnp.bfloat16),
                'recurrent_kvq': None,
                'relative_position_bias': jnp.zeros(
                    (b, n, 1, 1024), dtype=jnp.bfloat16
                ),
            }]
            * 12
        ),
        'eos': jnp.zeros([1024], dtype=jnp.bfloat16),
        'mask': jnp.ones([1024], dtype=jnp.bfloat16),
        'length': 1,
        'temperature': 1.0,
    })
    return fake_input_dict

  def __call__(self, inputs: ...) -> tuple[Any, dict[str, Any]]:
    # Make sure this code is not used on untested cases.
    if self.mode not in ['init', 'beam_search']:
      raise ValueError(f'{type(self)} cannot do mode {self.mode}')
    if self.decoder.supports_generate():
      raise ValueError(f'{type(self)}.decoder cannot supports_generate()')

    self.decoder(
        input_tokens=inputs['targets'][:, 0:1],
        target_tokens=None,
        start_of_sequence=inputs['start_of_sequence'],
    )

    b = inputs['targets'].shape[0]
    no_start_of_seq = jnp.array([False] * b, dtype=jnp.bool_)

    # This fn is used in both beam_search or topk_sampling.
    def tokens_to_logits_fn(
        input_token: jnp.ndarray, dstate: tuple[dict[str, jnp.ndarray], ...]
    ) -> tuple[jnp.ndarray, tuple[dict[str, jnp.ndarray], ...]]:
      (logits, dstate, _) = self.decoder(
          input_tokens=input_token,
          target_tokens=None,
          start_of_sequence=no_start_of_seq,
          decoder_state=dstate,
      )
      return logits[:, -1, :], dstate

    last_token = jax.lax.dynamic_slice_in_dim(
        inputs['targets'], inputs['length'] - 1, 1, axis=1
    )

    # last token is used to seed beam_search
    inputs['targets'] = inputs['targets'][:, 0:-1]
    dstate = jax.lax.cond(
        inputs['start_of_sequence'][0],
        lambda: self.generate(inputs)[0],
        lambda: inputs['dstate'],
    )

    # Then we run beam search, init with last_token & dstate.
    finished_seqs, finished_scores, dstate = beam_search.beam_search_flat(
        last_token,
        dstate,
        tokens_to_logits_fn,
        max_decode_len=512,
        eos=inputs['eos'].reshape((1, 1, -1)),
        mask=inputs['mask'].reshape((1, 1, -1)),
    )

    return 0.0, {
        'finished_seqs': finished_seqs,
        'finished_scores': finished_scores,
        'dstate': dstate,
    }

  def generate(
      self, inputs: ...
  ) -> tuple[tuple[dict[str, jnp.ndarray, ...], ...], jnp.ndarray]:
    """Generate an output sequence.

    Args:
      inputs: the same as argument to _call_.

    Returns:
      An array of generated tokens of shape (batch_size, sequence_length).
    """
    input_tokens = inputs['targets']  # [b,seq_len]
    start_of_sequence = inputs['start_of_sequence']  # [b]
    target_tokens = jnp.pad(input_tokens[:, 1:], [(0, 0), (0, 1)])
    batch_size = target_tokens.shape[0]

    # Assuming all sequences start at the same time.
    start0 = inputs['start_of_sequence'][0]
    dstate = jax.lax.cond(
        start0,
        lambda: self.decoder.init_decoder_state_vanilla(  # pylint: disable=g-long-lambda
            1024, start_of_sequence
        ),
        lambda: inputs['dstate'],
    )

    first_token = input_tokens[:, 0:1]
    no_start_of_seq = jnp.array([False] * batch_size, dtype=jnp.bool_)
    temperature = 1
    if 'temperature' in inputs:
      temperature = inputs['temperature']

    num_steps = inputs['length']
    if self.mode == 'beam_search':
      num_steps -= 1

    def cond_fn(scan_state) -> jnp.bool_:
      _, _, i, _ = scan_state
      return i < num_steps

    def loop_fn(scan_state: Any) -> Tuple[Any, Any, Any, Any]:
      (dstate, input_token, i, _) = scan_state

      (logits, dstate, _) = self.decoder(
          input_tokens=input_token,
          target_tokens=None,
          start_of_sequence=no_start_of_seq,
          decoder_state=dstate,
      )

      logits = logits / temperature
      output_token = jax.lax.dynamic_slice_in_dim(target_tokens, i, 1, axis=1)

      return (dstate, output_token, i + 1, logits)

    # Scan over the sequence length.
    dummy_logits = jnp.zeros((batch_size, 1, 1024))
    initial_scan_state = (dstate, first_token, 0, dummy_logits)
    dstate, _, _, logits = jax.lax.while_loop(
        cond_fn, loop_fn, initial_scan_state
    )
    return dstate, logits
