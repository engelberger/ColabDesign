import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np

from .utils import cat_neighbors_nodes, get_ar_mask

class mpnn_score:
  def score(self, I):
    """
    I = {
         [[required]]
         'X' = (L,4,3) 
         'mask' = (L,)
         'residue_index' = (L,)
         'chain_idx' = (L,)
         
         [[optional]]
         'S' = (L,21)
         'decoding_order' = (L,)
         'ar_mask' = (L,L)
        }
    """
    
    key = hk.next_rng_key()
    # Prepare node and edge embeddings
    E, E_idx = self.features(I)
    h_V = jnp.zeros((E.shape[0], E.shape[-1]))
    h_E = self.W_e(E)
    
    # Encoder is unmasked self-attention
    mask_attend = jnp.take_along_axis(I["mask"][:,None] * I["mask"][None,:], E_idx, 1)
    
    for layer in self.encoder_layers:
      h_V, h_E = layer(h_V, h_E, E_idx, I["mask"], mask_attend)

    # Build encoder embeddings
    h_EX_encoder = cat_neighbors_nodes(jnp.zeros_like(h_V), h_E, E_idx)
    h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

    if "S" not in I:
      ##########################################
      # unconditional_probs
      ##########################################      
      h_EXV_encoder_fw = h_EXV_encoder
      for layer in self.decoder_layers:
        h_V = layer(h_V, h_EXV_encoder_fw, I["mask"])
      decoding_order = None
    else:
      ##########################################
      # conditional_probs
      ##########################################

      # Concatenate sequence embeddings for autoregressive decoder
      h_S = self.W_s(I["S"])
      h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

      # get autoregressive mask
      if "ar_mask" in I:
        decoding_order = None
        ar_mask = I["ar_mask"]
      else:
        decoding_order = I["decoding_order"]
        ar_mask = get_ar_mask(decoding_order)
              
      mask_attend = jnp.take_along_axis(ar_mask, E_idx, 1)
      mask_1D = I["mask"][:,None]
      mask_bw = mask_1D * mask_attend
      mask_fw = mask_1D * (1 - mask_attend)

      h_EXV_encoder_fw = mask_fw[...,None] * h_EXV_encoder
      for layer in self.decoder_layers:
        # Masked positions attend to encoder information, unmasked see. 
        h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
        h_ESV = mask_bw[...,None] * h_ESV + h_EXV_encoder_fw
        h_V = layer(h_V, h_ESV, I["mask"])
    
    logits = self.W_out(h_V)
    S = I.get("S",None)
    return {"logits": logits, "decoding_order":decoding_order, "S":S}
  
class mpnn_score_dual_backbone(mpnn_score):
  def score_dual(self, I1, I2, combination_method='concatenate'):
        """
        Score a sequence against two different protein backbones simultaneously.

        Parameters:
        I1 (Dict[str, Any]): A dictionary containing the input features for the first backbone.
        I2 (Dict[str, Any]): A dictionary containing the input features for the second backbone.
        
        Returns:
        Dict[str, jnp.ndarray]: A dictionary with the following keys:
            - 'logits' (jnp.ndarray): Logits corresponding to the scored sequences with shape (L, 21).
            - 'decoding_order' (jnp.ndarray): Decoding order indices used during scoring with shape (L,).
            - 'S' (jnp.ndarray): The sequence that was scored, if provided, with shape (L, 21).
        """
        key = hk.next_rng_key()
        # Prepare node and edge embeddings
        E1, E_idx1 = self.features(I1)
        h_V1 = jnp.zeros((E1.shape[0], E1.shape[-1]))
        h_E1 = self.W_e(E1)
        mask_attend1 = jnp.take_along_axis(I1["mask"][:, None] * I1["mask"][None, :], E_idx1, 1)
        for layer in self.encoder_layers:
            h_V1, h_E1 = layer(h_V1, h_E1, E_idx1, I1["mask"], mask_attend1)
            
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
        
        # Combine the embeddings from both backbones using the specified method
        h_V_combined, h_E_combined = self.combine_embeddings(h_V1, h_E1, h_V2, h_E2, method=combination_method)
        
        # Ensure that the masks and decoding orders are compatible
        assert I1['mask'].shape == I2['mask'].shape, "Backbone masks have different shapes."
        assert I1['decoding_order'].shape == I2['decoding_order'].shape, "Backbone decoding orders have different shapes."
        
        # Get autoregressive mask, either from input or by computing it based on decoding order
        ar_mask = I1.get("ar_mask", get_ar_mask(I1["decoding_order"]))
        
        # Build encoder embeddings
        h_EX_encoder = cat_neighbors_nodes(jnp.zeros_like(h_V_combined), h_E_combined, E_idx1)
        h_EXV_encoder = cat_neighbors_nodes(h_V_combined, h_EX_encoder, E_idx1)

        # Check if a sequence is provided for scoring
        if "S" in I1:
            # Concatenate sequence embeddings for autoregressive decoder
            h_S = self.W_s(I1["S"])
            h_ES = cat_neighbors_nodes(h_S, h_E_combined, E_idx1)

            # Apply autoregressive mask
            mask_attend = jnp.take_along_axis(ar_mask, E_idx1, 1)
            mask_1D = I1["mask"][:, None]
            mask_bw = mask_1D * mask_attend
            h_EXV_encoder = mask_bw[..., None] * h_EXV_encoder

            # Process sequence embeddings through the decoder layers
            for layer in self.decoder_layers:
                h_ESV = cat_neighbors_nodes(h_V_combined, h_ES, E_idx1)
                h_ESV += h_EXV_encoder
                h_V_combined = layer(h_V_combined, h_ESV, I1["mask"])
        else:
            # Process embeddings through the decoder layers without sequence information
            for layer in self.decoder_layers:
                h_V_combined = layer(h_V_combined, h_EXV_encoder, I1["mask"])

        # Compute logits
        logits = self.W_out(h_V_combined)

        # Return the logits, decoding order, and optionally the scored sequence
        return {
            "logits": logits,
            "decoding_order": I1["decoding_order"],
            "S": I1.get("S", None)
        }