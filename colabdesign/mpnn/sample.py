import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np

from .utils import cat_neighbors_nodes, get_ar_mask

import haiku as hk
import jax
import jax.numpy as jnp
from typing import Any, Dict, Optional

class mpnn_sample:
  """
  A class representing a message passing neural network (MPNN) for sampling.
  This class contains methods to process input features and generate samples
  using an encoder-decoder architecture with autoregressive sampling.
  """
  def sample(self, I: Dict[str, Any]) -> Dict[str, jnp.ndarray]:
    """
    Generate samples from the MPNN model given the input features.
    Parameters:
    I (Dict[str, Any]): A dictionary containing the input features with the following keys:
        - 'X' (jnp.ndarray): Node features with shape (L, 4, 3), where L is the sequence length.
        - 'mask' (jnp.ndarray): A mask array with shape (L,).
        - 'residue_index' (jnp.ndarray): Residue indices with shape (L,).
        - 'chain_idx' (jnp.ndarray): Chain indices with shape (L,).
        - 'decoding_order' (jnp.ndarray): Decoding order indices with shape (L,).
        - 'ar_mask' (Optional[jnp.ndarray]): Autoregressive mask with shape (L, L). (optional)
        - 'bias' (Optional[jnp.ndarray]): Bias terms with shape (L, 21). (optional)
        - 'temperature' (Optional[float]): Temperature parameter for sampling. (optional)
    Returns:
    Dict[str, jnp.ndarray]: A dictionary with the following keys:
        - 'S' (jnp.ndarray): Sampled sequences with shape (L, 21).
        - 'logits' (jnp.ndarray): Logits corresponding to the sampled sequences with shape (L, 21).
        - 'decoding_order' (jnp.ndarray): Decoding order indices used during sampling with shape (L,).
        
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

    # Get a new random key for random number generation
    key = hk.next_rng_key() 
    # Get the sequence length (L) from the shape of the node features 'X' 
    L = I["X"].shape[0]
    # Get the temperature for sampling, default to 1.0 if not provided
    temperature = I.get("temperature", 1.0)  


    # Prepare node and edge embeddings
    E, E_idx = self.features(I)  # Compute edge features and their indices from input features
    h_V = jnp.zeros((E.shape[0], E.shape[-1]))  # Initialize node embeddings to zeros
    h_E = self.W_e(E)  # Apply a learned transformation to edge features to get edge embeddings

    # Encoder: process node and edge embeddings
    mask_attend = jnp.take_along_axis(I["mask"][:, None] * I["mask"][None, :], E_idx, 1)
    for layer in self.encoder_layers:  # Iterate over encoder layers
        h_V, h_E = layer(h_V, h_E, E_idx, I["mask"], mask_attend)  # Update node and edge embeddings

    # Get autoregressive mask, either from input or by computing it based on decoding order
    ar_mask = I.get("ar_mask", get_ar_mask(I["decoding_order"]))
    
    # Compute masked attention for forward and backward directions
    mask_attend = jnp.take_along_axis(ar_mask, E_idx, 1)
    mask_1D = I["mask"][:, None]
    mask_bw = mask_1D * mask_attend  # Backward mask
    mask_fw = mask_1D * (1 - mask_attend)  # Forward mask
    
    # Concatenate node embeddings with edge embeddings for encoder
    h_EX_encoder = cat_neighbors_nodes(jnp.zeros_like(h_V), h_E, E_idx)
    h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
    h_EXV_encoder = mask_fw[..., None] * h_EXV_encoder  # Apply forward mask

    # Define forward function for autoregressive sampling
    def fwd(x, t, key):
        # Extract the encoder's node and edge embeddings at time step t
        h_EXV_encoder_t = h_EXV_encoder[t]
        # Extract the edge indices at time step t
        E_idx_t = E_idx[t]
        # Extract the mask at time step t
        mask_t = I["mask"][t]
        # Extract the backward mask at time step t
        mask_bw_t = mask_bw[t]
        # Concatenate sampled node embeddings with edge embeddings at time step t
        h_ES_t = cat_neighbors_nodes(x["h_S"], h_E[t], E_idx_t)

        # Decoder: process node embeddings and generate new node states
        for l, layer in enumerate(self.decoder_layers):
            # Get the node embeddings from the previous layer (or initial embeddings for the first layer)
            h_V = x["h_V"][l]
            # Concatenate node embeddings with sampled edge embeddings at time step t
            h_ESV_decoder_t = cat_neighbors_nodes(h_V, h_ES_t, E_idx_t)
            # Apply backward mask and add encoder's embeddings to get the final input for the decoder layer
            h_ESV_t = mask_bw_t[..., None] * h_ESV_decoder_t + h_EXV_encoder_t
            # Pass the input through the decoder layer to get new node embeddings at time step t
            h_V_t = layer(h_V[t], h_ESV_t, mask_V=mask_t)
            # Update the node embeddings for the next layer (or final embeddings if this is the last layer)
            x["h_V"] = x["h_V"].at[l + 1, t].set(h_V_t)

        # Apply a learned transformation to the final node embeddings to get logits at time step t
        logits_t = self.W_out(h_V_t)
        # Update the logits in the state dictionary
        x["logits"] = x["logits"].at[t].set(logits_t)

        # Add bias to logits if provided in the input features
        if "bias" in I:
            logits_t += I["bias"][t]

        # Sample a character by applying temperature scaling and adding Gumbel noise for reparameterization trick
        logits_t = logits_t / temperature + jax.random.gumbel(key, logits_t.shape)

        # Tie positions by taking the mean across the batch dimension (if present)
        logits_t = logits_t.mean(0, keepdims=True)

        # Convert logits to one-hot encoded samples, ignoring the last class (assumed to be a padding class)
        S_t = jax.nn.one_hot(logits_t[..., :20].argmax(-1), 21)

        # Update the sampled node embeddings with the transformation of the sampled one-hot vectors
        x["h_S"] = x["h_S"].at[t].set(self.W_s(S_t))
        # Update the sampled sequences in the state dictionary
        x["S"] = x["S"].at[t].set(S_t)

        # Return the updated state and None (the second value is required by the scan function but not used here)
        return x, None
    
    # Initial values for node states, logits, and sampled sequences
    X = {"h_S":    jnp.zeros_like(h_V),
         "h_V":    jnp.array([h_V] + [jnp.zeros_like(h_V)] * len(self.decoder_layers)),
         "S":      jnp.zeros((L,21)),
         "logits": jnp.zeros((L,21))}

    # Scan over decoding order to perform autoregressive sampling
    t = I["decoding_order"]
    if t.ndim == 1: t = t[:,None]  # Ensure decoding order is a 2D array
    XS = {"t":t, "key":jax.random.split(key,t.shape[0])} # Prepare inputs for scan
    X = hk.scan(lambda x, xs: fwd(x, xs["t"], xs["key"]), X, XS)[0]  # Perform the scan
    
    # Return the sampled sequences, logits, and decoding order
    return {"S":X["S"], "logits":X["logits"], "decoding_order":t}
  
################ Dual attempt ################
class mpnn_sample_dual_backbone(mpnn_sample):
    def combine_embeddings(self, h_V1, h_E1, h_V2, h_E2, method='concatenate'):
        """
        Combine the node and edge embeddings from two backbones using the specified method.
        
        Args:
        h_V1, h_E1: Node and edge embeddings from the first backbone.
        h_V2, h_E2: Node and edge embeddings from the second backbone.
        method (str): The method to use for combining embeddings. Options are 'concatenate', 'average',
                      'learned', 'weighted', or 'dimensionality_reduction'.
        
        Returns:
        Combined node and edge embeddings.
        """
        if method == 'concatenate':
            return self.concatenate_embeddings(h_V1, h_E1, h_V2, h_E2)
        elif method == 'average':
            return self.average_embeddings(h_V1, h_E1, h_V2, h_E2)
        elif method == 'learned':
            return self.learned_combination(h_V1, h_E1, h_V2, h_E2)
        elif method == 'weighted':
            return self.weighted_combination(h_V1, h_E1, h_V2, h_E2)
        elif method == 'dimensionality_reduction':
            return self.dimensionality_reduction(h_V1, h_E1, h_V2, h_E2)
        else:
            raise ValueError(f"Unknown combination method: {method}")

    def concatenate_embeddings(self, h_V1, h_E1, h_V2, h_E2):
        # Concatenate embeddings along the feature dimension
        h_V_combined = jnp.concatenate([h_V1, h_V2], axis=-1)
        h_E_combined = jnp.concatenate([h_E1, h_E2], axis=-1)
        return h_V_combined, h_E_combined

    def average_embeddings(self, h_V1, h_E1, h_V2, h_E2):
        # Compute the element-wise average of the embeddings
        h_V_combined = (h_V1 + h_V2) / 2
        h_E_combined = (h_E1 + h_E2) / 2
        return h_V_combined, h_E_combined

    def learned_combination(self, h_V1, h_E1, h_V2, h_E2):
        # Use a learned neural network layer to combine the embeddings
        # This requires defining and training a suitable combination layer
        raise NotImplementedError("Learned combination method is not implemented.")

    def weighted_combination(self, h_V1, h_E1, h_V2, h_E2):
        # Use weighted sum to combine the embeddings
        # Weights can be predefined or learned as part of the model
        raise NotImplementedError("Weighted combination method is not implemented.")

    def dimensionality_reduction(self, h_V1, h_E1, h_V2, h_E2):
        # Apply dimensionality reduction to combined embeddings
        # This can involve a linear layer or other techniques like PCA
        raise NotImplementedError("Dimensionality reduction method is not implemented.")


    
    def sample_dual(self, I1: Dict[str, Any], I2: Dict[str, Any], combination_method='concatenate') -> Dict[str, jnp.ndarray]:

        """
        Generate a single sequence that satisfies two different protein backbones simultaneously.
        
        Parameters:
        I1 (Dict[str, Any]): A dictionary containing the input features for the first backbone.
        I2 (Dict[str, Any]): A dictionary containing the input features for the second backbone.
        
        Returns:
        Dict[str, jnp.ndarray]: A dictionary with the following keys:
            - 'S' (jnp.ndarray): Sampled sequences with shape (L, 21).
            - 'logits' (jnp.ndarray): Logits corresponding to the sampled sequences with shape (L, 21).
            - 'decoding_order' (jnp.ndarray): Decoding order indices used during sampling with shape (L,).
        """
        # Get a new random key for random number generation
        key = hk.next_rng_key()
        # Get the sequence length (L) from the shape of the node features 'X'
        L = I1["X"].shape[0]
        # Get the temperature for sampling, default to 1.0 if not provided
        temperature = I1.get("temperature", 1.0)

        # Encoder processing for the first backbone
        E1, E_idx1 = self.features(I1)
        h_V1 = jnp.zeros((E1.shape[0], E1.shape[-1]))
        h_E1 = self.W_e(E1)
        mask_attend1 = jnp.take_along_axis(I1["mask"][:, None] * I1["mask"][None, :], E_idx1, 1)
        for layer in self.encoder_layers:
            h_V1, h_E1 = layer(h_V1, h_E1, E_idx1, I1["mask"], mask_attend1)
        
        # Encoder processing for the second backbone
        E2, E_idx2 = self.features(I2)
        h_V2 = jnp.zeros((E2.shape[0], E2.shape[-1]))
        h_E2 = self.W_e(E2)
        mask_attend2 = jnp.take_along_axis(I2["mask"][:, None] * I2["mask"][None, :], E_idx2, 1)
        for layer in self.encoder_layers:
            h_V2, h_E2 = layer(h_V2, h_E2, E_idx2, I2["mask"], mask_attend2)
        
        
        # Combine the embeddings from both backbones using the specified method
        h_V_combined, h_E_combined = self.combine_embeddings(h_V1, h_E1, h_V2, h_E2, method=combination_method)

        # Since we're combining two backbones, we need to ensure that the masks and decoding orders are compatible.
        # For simplicity, we'll assume that both backbones have the same length and mask.
        # In practice, you may need to align or interpolate between the two backbones.
        assert I1['mask'].shape == I2['mask'].shape, "Backbone masks have different shapes."
        assert I1['decoding_order'].shape == I2['decoding_order'].shape, "Backbone decoding orders have different shapes."
        
        # Get autoregressive mask, either from input or by computing it based on decoding order
        ar_mask = I1.get("ar_mask", get_ar_mask(I1["decoding_order"]))

        # Compute masked attention for forward and backward directions based on the combined autoregressive mask
        mask_attend = jnp.take_along_axis(ar_mask, E_idx1, 1)
        mask_1D = I1["mask"][:, None]
        mask_bw = mask_1D * mask_attend  # Backward mask
        mask_fw = mask_1D * (1 - mask_attend)  # Forward mask
        

        # Concatenate node embeddings with edge embeddings for encoder
        h_EX_encoder = cat_neighbors_nodes(jnp.zeros_like(h_V_combined), h_E_combined, E_idx1)
        h_EXV_encoder_fw = cat_neighbors_nodes(h_V_combined, h_EX_encoder, E_idx1)
        h_EXV_encoder_fw = mask_fw[..., None] * h_EXV_encoder_fw  # Apply forward mask
        
        # Define forward_dual function for autoregressive sampling
        def fwd_dual(self, x, t, key, h_EXV_encoder, E_idx, mask, mask_bw, temperature):
          # Extract the encoder's node and edge embeddings at time step t
          h_EXV_encoder_t = h_EXV_encoder[t]
          # Extract the edge indices at time step t
          E_idx_t = E_idx[t]
          # Extract the mask at time step t
          mask_t = mask[t]
          # Extract the backward mask at time step t
          mask_bw_t = mask_bw[t]
          # Concatenate sampled node embeddings with edge embeddings at time step t
          h_ES_t = cat_neighbors_nodes(x["h_S"], x["h_E"][t], E_idx_t)

          # Decoder: process node embeddings and generate new node states
          for l, layer in enumerate(self.decoder_layers):
              # Get the node embeddings from the previous layer (or initial embeddings for the first layer)
              h_V = x["h_V"][l]
              # Concatenate node embeddings with sampled edge embeddings at time step t
              h_ESV_decoder_t = cat_neighbors_nodes(h_V, h_ES_t, E_idx_t)
              # Apply backward mask and add encoder's embeddings to get the final input for the decoder layer
              h_ESV_t = mask_bw_t[..., None] * h_ESV_decoder_t + h_EXV_encoder_t
              # Pass the input through the decoder layer to get new node embeddings at time step t
              h_V_t = layer(h_V[t], h_ESV_t, mask_V=mask_t)
              # Update the node embeddings for the next layer (or final embeddings if this is the last layer)
              x["h_V"] = x["h_V"].at[l + 1, t].set(h_V_t)

          # Apply a learned transformation to the final node embeddings to get logits at time step t
          logits_t = self.W_out(h_V_t)
          # Update the logits in the state dictionary
          x["logits"] = x["logits"].at[t].set(logits_t)

          # Add bias to logits if provided in the input features
          if "bias" in x:
              logits_t += x["bias"][t]

          # Sample a character by applying temperature scaling and adding Gumbel noise for reparameterization trick
          logits_t = logits_t / temperature + jax.random.gumbel(key, logits_t.shape)

          # Tie positions by taking the mean across the batch dimension (if present)
          logits_t = logits_t.mean(0, keepdims=True)

          # Convert logits to one-hot encoded samples, ignoring the last class (assumed to be a padding class)
          S_t = jax.nn.one_hot(logits_t[..., :20].argmax(-1), 21)

          # Update the sampled node embeddings with the transformation of the sampled one-hot vectors
          x["h_S"] = x["h_S"].at[t].set(self.W_s(S_t))
          # Update the sampled sequences in the state dictionary
          x["S"] = x["S"].at[t].set(S_t)

          # Return the updated state and None (the second value is required by the scan function but not used here)
          return x, None
        
        # Initial values for node states, logits, and sampled sequences
        X = {
            "h_S": jnp.zeros_like(h_V_combined),
            "h_V": jnp.array([h_V_combined] + [jnp.zeros_like(h_V_combined)] * len(self.decoder_layers)),
            "S": jnp.zeros((L, 21)),
            "logits": jnp.zeros((L, 21)),
            "h_E": h_E_combined,
            "bias": I1.get("bias", jnp.zeros((L, 21)))  # Use bias from I1 or zeros if not provided
        }

        # Scan over decoding order to perform autoregressive sampling
        t = I1["decoding_order"]
        if t.ndim == 1:
            t = t[:, None]  # Ensure decoding order is a 2D array
        XS = {"t": t, "key": jax.random.split(key, t.shape[0])}  # Prepare inputs for scan
        X = hk.scan(lambda x, xs: fwd_dual(x, xs["t"], xs["key"], h_EXV_encoder_fw, E_idx1, I1["mask"], mask_bw, temperature), X, XS)[0]  # Perform the scan

        # Return the sampled sequences, logits, and decoding order
        return {"S": X["S"], "logits": X["logits"], "decoding_order": t}

