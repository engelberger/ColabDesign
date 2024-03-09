import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np

from .utils import cat_neighbors_nodes, get_ar_mask

class mpnn_sample:
  def sample(self, I):
    """
    I = {
         [[required]]
         'X' = (L,4,3) 
         'mask' = (L,)
         'residue_index' = (L,)
         'chain_idx' = (L,)
         'decoding_order' = (L,)
         
         [[optional]]
         'ar_mask' = (L,L)
         'bias' = (L,21)
         'temperature' = 1.0
        }
    """

    key = hk.next_rng_key()
    L = I["X"].shape[0]
    temperature = I.get("temperature",1.0)

    # prepare node and edge embeddings
    E, E_idx = self.features(I)
    h_V = jnp.zeros((E.shape[0], E.shape[-1]))
    h_E = self.W_e(E)

    ##############
    # encoder
    ##############
    mask_attend = jnp.take_along_axis(I["mask"][:,None] * I["mask"][None,:], E_idx, 1)
    for layer in self.encoder_layers:
      h_V, h_E = layer(h_V, h_E, E_idx, I["mask"], mask_attend)

    # get autoregressive mask  
    ar_mask = I.get("ar_mask",get_ar_mask(I["decoding_order"]))
    
    mask_attend = jnp.take_along_axis(ar_mask, E_idx, 1)
    mask_1D = I["mask"][:,None]
    mask_bw = mask_1D * mask_attend
    mask_fw = mask_1D * (1 - mask_attend)
    
    h_EX_encoder = cat_neighbors_nodes(jnp.zeros_like(h_V), h_E, E_idx)
    h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
    h_EXV_encoder = mask_fw[...,None] * h_EXV_encoder

    def fwd(x, t, key):
      h_EXV_encoder_t = h_EXV_encoder[t] 
      E_idx_t         = E_idx[t]
      mask_t          = I["mask"][t]
      mask_bw_t       = mask_bw[t]      
      h_ES_t          = cat_neighbors_nodes(x["h_S"], h_E[t], E_idx_t)

      ##############
      # decoder
      ##############
      for l,layer in enumerate(self.decoder_layers):
        h_V = x["h_V"][l]
        h_ESV_decoder_t = cat_neighbors_nodes(h_V, h_ES_t, E_idx_t)
        h_ESV_t = mask_bw_t[...,None] * h_ESV_decoder_t + h_EXV_encoder_t
        h_V_t = layer(h_V[t], h_ESV_t, mask_V=mask_t)
        # update
        x["h_V"] = x["h_V"].at[l+1,t].set(h_V_t)

      logits_t = self.W_out(h_V_t)
      x["logits"] = x["logits"].at[t].set(logits_t)

      ##############
      # sample
      ##############
      
      # add bias
      if "bias" in I: logits_t += I["bias"][t]          

      # sample character
      logits_t = logits_t/temperature + jax.random.gumbel(key, logits_t.shape)

      # tie positions
      logits_t = logits_t.mean(0, keepdims=True)

      S_t = jax.nn.one_hot(logits_t[...,:20].argmax(-1), 21)

      # update
      x["h_S"] = x["h_S"].at[t].set(self.W_s(S_t))
      x["S"]   = x["S"].at[t].set(S_t)
      return x, None
    
    # initial values
    X = {"h_S":    jnp.zeros_like(h_V),
         "h_V":    jnp.array([h_V] + [jnp.zeros_like(h_V)] * len(self.decoder_layers)),
         "S":      jnp.zeros((L,21)),
         "logits": jnp.zeros((L,21))}

    # scan over decoding order
    t = I["decoding_order"]
    if t.ndim == 1: t = t[:,None]
    XS = {"t":t, "key":jax.random.split(key,t.shape[0])}
    X = hk.scan(lambda x, xs: fwd(x, xs["t"], xs["key"]), X, XS)[0]
    
    return {"S":X["S"], "logits":X["logits"], "decoding_order":t}
  
  def tied_sample(self, I):    
    """
    I = {
         [[required]]
         'X' = (L,4,3)  # Input features for L residues, each with 4 features of size 3
         'mask' = (L,)  # Mask indicating valid residues (1) and padding (0)
         'residue_index' = (L,)  # Index of each residue
         'chain_idx' = (L,)  # Index of the chain for each residue
         'decoding_order' = (L,)  # Order in which residues will be decoded
         'tied_positions' = [(indices of tied group 1), (indices of tied group 2), ...]  # Groups of tied positions
         
         [[optional]]
         'ar_mask' = (L,L)  # Autoregressive mask to control visibility between residues
         'bias' = (L,21)  # Bias to add to logits for each residue and possible amino acid
         'temperature' = 1.0  # Temperature for softmax sampling
        }
    """

    key = hk.next_rng_key()  # Get a random key for sampling
    L = I["X"].shape[0]  # Number of residues
    temperature = I.get("temperature", 1.0)  # Temperature for sampling
    tied_positions = I["tied_positions"]  # Groups of tied positions

    # Prepare node and edge embeddings
    E, E_idx = self.features(I)  # E: edge features, E_idx: indices of connected nodes
    h_V = jnp.zeros((E.shape[0], E.shape[-1]))  # Initialize node embeddings
    h_E = self.W_e(E)  # Transform edge features

    # Encoder: process embeddings with self-attention
    mask_attend = jnp.take_along_axis(I["mask"][:, None] * I["mask"][None, :], E_idx, 1)  # Compute attention mask
    for layer in self.encoder_layers:
      h_V, h_E = layer(h_V, h_E, E_idx, I["mask"], mask_attend)  # Update embeddings

    # Get autoregressive mask
    ar_mask = I.get("ar_mask", get_ar_mask(I["decoding_order"]))  # Compute or get provided autoregressive mask
    
    mask_attend = jnp.take_along_axis(ar_mask, E_idx, 1)  # Apply autoregressive mask
    mask_1D = I["mask"][:, None]  # Reshape mask for broadcasting
    mask_bw = mask_1D * mask_attend  # Mask for backward connections
    mask_fw = mask_1D * (1 - mask_attend)  # Mask for forward connections
    
    h_EX_encoder = cat_neighbors_nodes(jnp.zeros_like(h_V), h_E, E_idx)  # Aggregate neighbor embeddings
    h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)  # Combine node and edge embeddings
    h_EXV_encoder = mask_fw[..., None] * h_EXV_encoder  # Apply forward mask

    def fwd(x, t, key):
      # For tied positions, average logits across all tied positions before sampling
      tied_positions_mask = jnp.zeros((L,), dtype=jnp.bool_)  # Initialize mask for tied positions
      for group in tied_positions:
        if t in group:
          tied_positions_mask = jax.ops.index_update(tied_positions_mask, jnp.array(group), True)  # Mark tied positions
          break

      tied_positions_indices = jnp.where(tied_positions_mask)[0]  # Get indices of tied positions
      logits_t = jnp.zeros((21,))  # Initialize logits

      for tied_t in tied_positions_indices:
        # Process each tied position
        h_EXV_encoder_t = h_EXV_encoder[tied_t]  # Get encoder output for tied position
        E_idx_t = E_idx[tied_t]  # Get indices of connected nodes for tied position
        mask_t = I["mask"][tied_t]  # Get mask for tied position
        mask_bw_t = mask_bw[tied_t]  # Get backward mask for tied position
        h_ES_t = cat_neighbors_nodes(x["h_S"], h_E[tied_t], E_idx_t)  # Aggregate neighbor embeddings

        for l, layer in enumerate(self.decoder_layers):
          # Decoder: process embeddings with self-attention
          h_V = x["h_V"][l]  # Get current node embeddings
          h_ESV_decoder_t = cat_neighbors_nodes(h_V, h_ES_t, E_idx_t)  # Combine node and edge embeddings
          h_ESV_t = mask_bw_t[..., None] * h_ESV_decoder_t + h_EXV_encoder_t  # Apply backward mask and combine with encoder output
          h_V_t = layer(h_V[tied_t], h_ESV_t, mask_V=mask_t)  # Update node embeddings for tied position
          x["h_V"] = x["h_V"].at[l+1, tied_t].set(h_V_t)  # Store updated embeddings

        tied_logits_t = self.W_out(h_V_t)  # Compute logits for tied position
        logits_t += tied_logits_t  # Accumulate logits across tied positions

      logits_t /= len(tied_positions_indices)  # Average logits across tied positions

      if "bias" in I:
        logits_t += I["bias"][tied_t]  # Add bias to logits

      logits_t = logits_t / temperature + jax.random.gumbel(key, logits_t.shape)  # Apply temperature and add Gumbel noise for sampling
      S_t = jax.nn.one_hot(logits_t[..., :20].argmax(-1), 21)  # Sample amino acid

      for tied_t in tied_positions_indices:
        # Update embeddings and sampled sequence for all tied positions
        x["h_S"] = x["h_S"].at[tied_t].set(self.W_s(S_t))  # Update state embeddings
        x["S"] = x["S"].at[tied_t].set(S_t)  # Update sampled sequence

      return x, None

    # Initialize state for scanning
    X = {"h_S": jnp.zeros_like(h_V),  # State embeddings
         "h_V": jnp.array([h_V] + [jnp.zeros_like(h_V)] * len(self.decoder_layers)),  # Node embeddings for each decoder layer
         "S": jnp.zeros((L, 21)),  # Sampled sequence
         "logits": jnp.zeros((L, 21))}  # Logits for each position

    # Scan over decoding order
    t = I["decoding_order"]
    if t.ndim == 1: t = t[:, None]
    XS = {"t": t, "key": jax.random.split(key, t.shape[0])}  # Prepare inputs for scanning
    X = hk.scan(lambda x, xs: fwd(x, xs["t"], xs["key"]), X, XS)[0]  # Scan over positions

    return {"S": X["S"], "logits": X["logits"], "decoding_order": t}  # Return sampled sequence and logits