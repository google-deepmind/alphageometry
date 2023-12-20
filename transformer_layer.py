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

"""A single transformer layer in inference mode.

Modified
https://github.com/google-research/meliad/blob/main/transformer/transformer_layer.py
To accommodate sequence packing + kv cache + relative position during test time.
"""

from typing import Callable, Mapping, NewType, Optional, Tuple

from absl import logging
import gin
import jax
import jax.numpy as jnp
from transformer import attention
from transformer import nn_components
from transformer import position
from transformer import transformer_layer


Array = jnp.ndarray
DecoderState = NewType("DecoderState", Mapping[str, Array])
WindowState = Optional[Tuple[attention.KVITuple, Array]]


@jax.vmap
def update_slice_in_dim_1(array: Array, update: Array, idx: Array) -> Array:
  """Update a stored keys/values slice for different-lengthed seqs in batch."""
  return jax.lax.dynamic_update_slice_in_dim(array, update, idx, axis=0)


def slice_in_dim_1(window_length: int) -> Callable[[Array, Array], Array]:
  @jax.vmap
  def fn(array: Array, idx: Array) -> Array:
    return jax.lax.dynamic_slice_in_dim(array, idx, window_length, axis=0)

  return fn


@gin.configurable
class TransformerLayerGenerate(transformer_layer.TransformerLayer):
  """Full transformer layer, with attention."""

  def _next_decoder_state(
      self, decoder_state: DecoderState, keys: Array, values: Array
  ) -> Tuple[DecoderState, Array, Array]:
    """Compute the next decoder state, and return keys,values to attend to.

    The keys,values returned from this function are drawn from the prior
    decoding state, and comprise a full window of local context.

    Args:
      decoder_state: The current decoder state, initially created using
        init_decoder_state().
      keys: The key for the current token, of shape (batch_size, 1, dim)
      values: The value for the current token of shape (batch_size, 1, dim)

    Returns:
      (next_decoder_state,
       window of keys of shape (batch_size, window_length, dim),
       window of values of shape (batch_size, window_length, dim))
    """

    assert keys.shape[1] == 1  # single-token autoregressive decoding.

    # Unpack decoder_state
    stored_keys = decoder_state["keys"]
    stored_values = decoder_state["values"]
    curr_index = decoder_state["current_index"]

    # Slice to get window_length-sized chunk of previous keys,values.
    out_decoder_state = {}
    curr_win_index = curr_index - self.window_length

    # out_keys = jax.lax.dynamic_slice_in_dim(
    #     stored_keys, curr_win_index, self.window_length, axis=1)
    out_keys = slice_in_dim_1(self.window_length)(stored_keys, curr_win_index)

    # out_values = jax.lax.dynamic_slice_in_dim(
    #     stored_values, curr_win_index, self.window_length, axis=1)
    out_values = slice_in_dim_1(self.window_length)(
        stored_values, curr_win_index
    )

    # Write current keys,values to stored keys, values.
    # stored_keys = jax.lax.dynamic_update_slice_in_dim(
    #     stored_keys, keys, curr_index, axis=1)
    stored_keys = update_slice_in_dim_1(stored_keys, keys, curr_index)
    # stored_values = jax.lax.dynamic_update_slice_in_dim(
    #     stored_values, values, curr_index, axis=1)
    stored_values = update_slice_in_dim_1(stored_values, values, curr_index)
    curr_index = curr_index + 1

    # Pack a new decoder_state object.
    out_decoder_state["keys"] = stored_keys
    out_decoder_state["values"] = stored_values
    out_decoder_state["current_index"] = curr_index
    out_decoder_state["relative_position_bias"] = decoder_state[
        "relative_position_bias"
    ]
    out_decoder_state["recurrent_kvq"] = decoder_state["recurrent_kvq"]

    return (DecoderState(out_decoder_state), out_keys, out_values)

  def __call__(
      self,
      xs: Array,
      start_of_sequence: Array,
      *,
      importance: Optional[Array] = None,
      cross_attention_kv: Optional[Tuple[Array, Array]] = None,
      window_state: Optional[WindowState] = None,
      decoder_state: Optional[DecoderState] = None,
  ):
    """Computes attention over a sequence of inputs.

    Args:
      xs: input sequence of shape (batch_size, sequence_length, num_hidden)
      start_of_sequence: An input array of shape (batch_size)  --- The following
        must be passed by keyword only. ---
      importance: Array of shape (batch_size, sequence_length). An importance
        bias for attention.
      cross_attention_kv: Keys and values from encoder for cross-attention.
      window_state: State object which contains context from the prior window
        when using a transformer-XL or sliding window. Initially created with
        load_window_state().
      decoder_state: State object for autoregressive decoding, initially created
        with from init_decoder_state().

    Returns:
      (ys: outputs of shape (batch_size, sequence_length, num_hidden),
       importance_score: importance score for the next layer,
       next_window_state: state to pass to the next window,
       next_decoder_state: next decoder state for autoregressive decoding,
       viz_dict: dictionary of visualizations
      )
    """

    xs = jnp.asarray(xs, dtype=self.dtype)
    logging.info("tlayer: recurrent = %r", self.recurrent_attention)
    logging.info("tlayer: compute_importance = %r", self.compute_importance)

    is_training = self.mode == "train"

    # Compute keys, values and queries.
    # ---------------------------------
    logging.info("tlayer: compute keys,values,queries.")
    (keys, values, queries, queries2) = self.tbase.kvq(xs)
    attention_scale_factors = self.tbase.attention_scale_factors()
    (_, sequence_length, num_heads, _) = queries.shape  # (b, k, h, d)

    # Get biases and masks that are shared across windows.
    # ----------------------------------------------------
    if decoder_state is not None:
      logging.info("tlayer: using autoregressive decoder.")
      # When decoding, prior keys,values are loaded from the decoder state.
      # Other values are precomputed, and loaded from the decoder state.
      # The decoder state will be updated with the current token.
      assert window_state is None

      prev_kvi = None
      recurrent_state = None  # Use precomputed recurrent_kvq.
      cross_attention_kv = None
      rel_position_bias = decoder_state["relative_position_bias"]
      causal_mask = None
      dropout_multiplier = None

      # Reuse cached recurrent keys,values for each token.
      cached_recurrent_kvq = decoder_state["recurrent_kvq"]
      if cached_recurrent_kvq is not None:
        assert cross_attention_kv is None
        cross_attention_kv = (cached_recurrent_kvq[0], cached_recurrent_kvq[1])
      del cached_recurrent_kvq

      # Get a full window of keys,values and update decoder state.
      (decoder_state, keys, values) = self._next_decoder_state(
          decoder_state, keys, values
      )

      # Each query attends to window_length prior keys.
      assert keys.shape[1] == self.window_length
      kq_relative_offset = self.window_length

      if not self.use_long_xl_architecture:
        kqpos = position.relative_positions(
            1, self.window_length, offset=0
        )  # 2D mask
        current_idx = decoder_state["current_index"]

        # add (batch, heads) dims for kqpos
        kqpos = jnp.expand_dims(kqpos, axis=(0, 1))
        kqpos = jnp.tile(kqpos, (1, self.num_heads, 1, 1))

        # add (_, heads, _) dim for current_idx
        current_idx = jnp.expand_dims(current_idx, axis=(1, 2, 3))

        causal_mask = kqpos > self.window_length * 2 - current_idx
    else:
      logging.info("tlayer: windowed attention.")
      # When training, attention is done using windows or chunks, and prior
      # context (e.g. keys,values from the previous window) is stored in the
      # window_state object.
      (prev_kvi, recurrent_state) = (
          window_state  # pytype: disable=attribute-error
      )

      # Get the size of the sliding window for pos bias, dropout, & causal mask.
      (num_queries, num_keys) = attention.sliding_attention_window_shape(
          (keys, values, importance),
          prev_kvi,
          queries,
          window_length=self.window_length,
      )
      kq_relative_offset = num_keys - num_queries

      # Get the relative position bias.
      # The bias doesn't depend on the query content, and so can be precomputed.
      if self.relative_positions is not None:
        rel_position_bias = self.relative_positions(
            num_queries, num_keys, bidirectional=False
        )
      else:
        rel_position_bias = None

      # Get causal mask.
      if self.use_causal_mask:
        causal_mask = position.causal_mask(
            num_queries, num_keys, window_length=self.window_length
        )
      else:
        causal_mask = None

      # Apply dropout to the attention matrix.
      # The mask will be broadcast across batches and windows.
      if self.attn_dropout_rate > 0.0 and is_training:
        dropout_rng = self.make_rng("dropout")
        attn_shape = (self.num_heads, num_queries, num_keys)
        dropout_multiplier = nn_components.dropout_multiplier_mask(
            dropout_rng, self.attn_dropout_rate, attn_shape, self.dtype
        )
      else:
        dropout_multiplier = None

    # Load and store values into external memory, if memory is not None.
    # ------------------------------------------------------------------
    (mode, _, update_memory) = self._get_cache_name_from_mode(self.mode)
    external_kv = self._query_external_memory(
        keys,
        values,
        queries,
        start_of_sequence=start_of_sequence,
        mode=mode,
        update_memory=decoder_state is None and update_memory,
    )

    if (
        self.memory is not None
        and self.memory_combine_with_local == "TRAINABLE_WEIGHTED_MEAN"
    ):
      external_memory_bias = jnp.asarray(self.memory_bias, dtype=self.dtype)
      external_memory_bias = jnp.reshape(
          external_memory_bias, (1, 1, num_heads, 1)
      )
      external_memory_bias = jax.nn.sigmoid(external_memory_bias)
    else:
      external_memory_bias = None

    # Compute the number of windows.
    # ------------------------------
    if sequence_length < self.window_length:
      num_windows = 1  # Happens with autoregressive decoding.
    elif sequence_length == self.window_length:
      num_windows = 1
      if self.use_long_xl_architecture:
        assert prev_kvi is not None
    else:
      if not self.use_long_xl_architecture:
        raise ValueError("Can only use sliding window with Transformer XL.")
      num_windows = sequence_length // self.window_length
      if (num_windows * self.window_length) != sequence_length:
        raise ValueError(
            f"Window length {self.window_length} must be a "
            + f"multiple of sequence length {sequence_length}"
        )
    logging.info("tlayer: num_windows = %d.", num_windows)

    # Define the function to do attention within a single window.
    # ---------------------------------------------------------
    def single_window_attention(
        carry: tuple[Array, Array], inputs_w: tuple[Array, Array]
    ) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
      # This function uses the following variables from the outer scope.
      # They are listed here for clarity.
      nonlocal rel_position_bias
      nonlocal causal_mask
      nonlocal kq_relative_offset
      nonlocal dropout_multiplier
      nonlocal attention_scale_factors
      nonlocal external_memory_bias
      nonlocal cross_attention_kv  # externally supplied.

      # keys,values,queries over the whole sequence will be split into chunks.
      # xs_w, kvqi_w, etc. are the chunk for the current window.
      (prev_kvi_w, rec_state) = carry  # carried from one window to the next.
      (kvqi_w, external_kv_w) = inputs_w  # inputs to the current window.
      # (keys_curr_w, values_curr_w, _, _, importance_curr_w) = kvqi_w

      # Concatenate keys,values from the previous window with the current
      # window to implement sliding window attention.
      (kvqi_w, next_kvi_w) = attention.concat_kvqi(kvqi_w, prev_kvi_w)
      (keys_w, values_w, queries_w, queries2_w, importance_w) = kvqi_w

      # Perform recurrent attention within the current window to get the next
      # recurrent state, and set up cross attention.
      if rec_state is not None:
        logging.info("tlayer: recurrent attention.")

        # NOTE -- recurrent states and input tokens are handled separately,
        # because they have separate learned positional embeddings.  Due to
        # the way TransformerBase does cross-attention, this means that we use
        # separate key,value layers for rec_state and tokens_w.

        # Keys, values, queries from recurrent state.
        logging.info("tlayer: recurrent kvq.")
        rec_kvq = self.recurrent_tbase.kvq(rec_state)
        r_scale_factors = self.recurrent_tbase.attention_scale_factors()
        (r_keys, r_values, r_queries, r_queries2) = rec_kvq

        # Joint attention over both recurrent states and input tokens.
        logging.info("tlayer: recurrent self-attention.")
        r_attn_ys = attention.simple_attention(
            r_keys,
            r_values,
            r_queries,
            None,
            scale_factor=r_scale_factors[0],
            dtype=self.dtype,
        )

        logging.info("tlayer: recurrent cross-attention.")
        r_cross_attn_ys = attention.simple_attention(
            keys_w,
            values_w,
            r_queries2,
            importance_w,
            scale_factor=r_scale_factors[1],
            dtype=self.dtype,
        )

        # Recurrent post-attention FFN.
        logging.info("tlayer: recurrent ffn.")
        next_rec_state = self.recurrent_tbase.post_attn_ffn(
            rec_state, r_attn_ys, r_cross_attn_ys
        )

        # Get keys and values for cross-attention from recurrent state.
        assert cross_attention_kv is None
        local_cross_attention_kv = (r_keys, r_values)
      else:
        # Get keys and values for cross-attention from external argument.
        next_rec_state = None
        local_cross_attention_kv = cross_attention_kv

      # If using RoPE, keys and queries are rotated before self-attention.
      if self.relative_position_type == "rotary":
        logging.info(
            "Using rotary position encodings (RoPE), offset = %d",
            kq_relative_offset,
        )
        (keys_w, queries_w) = position.rotate_kq(
            keys_w, queries_w, max_wavelength=10_000, offset=kq_relative_offset
        )

      # Self-attention over input tokens.
      logging.info("tlayer: self-attention.")
      attn_ys_w = attention.simple_attention(
          keys_w,
          values_w,
          queries_w,
          importance_w,
          relative_position_bias=rel_position_bias,
          scale_factor=attention_scale_factors[0],
          causal_mask=causal_mask,
          dropout_multiplier=dropout_multiplier,
          dtype=self.dtype,
      )

      # Attention over external memory.
      if external_kv_w is not None:
        (external_keys_w, external_values_w) = external_kv_w
        y_ext = attention.external_attention(
            external_keys_w,
            external_values_w,
            queries_w,
            scale_factor=attention_scale_factors[0],
        )
        if external_memory_bias is not None:
          ebias = external_memory_bias
          attn_ys_w = (attn_ys_w * (1 - ebias)) + (y_ext * ebias)
        elif self.memory_combine_with_local == "ADD":
          attn_ys_w += y_ext
        elif self.memory_combine_with_local == "STOP_FORWARD":
          attn_ys_w = y_ext + (attn_ys_w - jax.lax.stop_gradient(attn_ys_w))
        else:
          raise ValueError(
              f"Unexpected setting: {self.memory_combine_with_local = }"
          )

      # Cross attention from input tokens to encoder or recurrent state.
      if local_cross_attention_kv is not None:
        logging.info("tlayer: cross-attention.")
        (c_keys, c_values) = local_cross_attention_kv

        # Cross-attention using queries2.
        cross_attn_ys_w = attention.simple_attention(
            c_keys,
            c_values,
            queries2_w,
            None,
            scale_factor=attention_scale_factors[1],
            dtype=self.dtype,
        )
      else:
        cross_attn_ys_w = None

      # End function single_window_attention(...)
      return ((next_kvi_w, next_rec_state), (attn_ys_w, cross_attn_ys_w))

    # Initialize recurrent_tbase before calling jax.lax.scan.
    # Otherwise flax will throw a tantrum.
    if (
        self.recurrent_attention
        and 0 <= self.max_unrolled_windows
        and self.max_unrolled_windows < num_windows
    ):
      logging.info("tlayer: force initialization of recurrent_tbase.")
      self.recurrent_tbase.force_init(recurrent_state)

    # Perform sliding window attention over all keys,values,queries.
    # --------------------------------------------------------------
    initial_carry = (prev_kvi, recurrent_state)  # window state.
    kvqi = (keys, values, queries, queries2, importance)
    attn_inputs = (kvqi, external_kv)
    (next_carry, attn_outputs) = attention.split_and_scan(
        single_window_attention,
        initial_carry,
        attn_inputs,
        sections=num_windows,
        axis=1,
        max_unrolled_windows=self.max_unrolled_windows,
    )
    (attn_ys, cross_attn_ys) = attn_outputs

    logging.info("tlayer: End windows.")

    # Post-attention MLP, resnet, and FFN.
    # ------------------------------------
    logging.info("tlayer: final FFN.")
    ys = self.tbase.post_attn_ffn(xs, attn_ys, cross_attn_ys)

    # Compute importance scores for each token if requested.
    if self.compute_importance:
      (batch_size, sequence_length, _) = ys.shape
      importance_score = self.importance_layer(ys)
      importance_score = importance_score.reshape((batch_size, sequence_length))
    else:
      importance_score = None

    next_window_state = next_carry if window_state is not None else None
    viz_dict = {}  # Visualizations, not currently enabled.
    return (ys, importance_score, next_window_state, decoder_state, viz_dict)

  def init_decoder_state_vanilla(
      self, sequence_length: int, start_of_sequence: Array
  ) -> DecoderState:
    """Initialize decoder state for autoregressive generation.

    Args:
      sequence_length: The maximum length of the sequence to generate.
      start_of_sequence: Array of boolean of shape (batch_size,) True if
        starting a new sequence (with no prefix).

    Returns:
      A state object that can be passed to __call__.
    """

    if not self.use_causal_mask:
      raise ValueError("Generator must have been trained with a causal mask.")

    # Get relative position bias.
    rel_position_bias = self.relative_positions(
        1, self.window_length, offset=self.window_length, bidirectional=False
    )
    rel_position_bias = jnp.tile(rel_position_bias, (self.batch_size, 1, 1, 1))

    # Initialize autoregressive storage for (key, value) pairs.
    # Include space for a prefix of window_length tokens.
    num_keys = sequence_length + self.window_length
    stored_shape = (self.batch_size, num_keys, self.num_heads, self.head_size)
    stored_keys = jnp.zeros(stored_shape, dtype=self.dtype)
    stored_values = jnp.zeros(stored_shape, dtype=self.dtype)

    recurrent_kvq = None
    current_index = jnp.array([self.window_length] * self.batch_size)

    decoder_state_dict = {
        "keys": stored_keys,
        "values": stored_values,
        "current_index": current_index,
        "relative_position_bias": rel_position_bias,
        "recurrent_kvq": recurrent_kvq,
    }
    return DecoderState(decoder_state_dict)
