import functools
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import joblib
from typing import Optional, Tuple


from colabdesign.shared.prng import SafeKey
from .utils import cat_neighbors_nodes, get_ar_mask
from .sample import mpnn_sample
from .score import mpnn_score

Gelu = functools.partial(jax.nn.gelu, approximate=False)

class dropout_cust(hk.Module):
  """Custom dropout module using Haiku.
    
  Attributes:
        rate: The dropout rate.
  """
  def __init__(self, rate) -> None:
    """Initializes the dropout module with a given rate.
        
        Args:
            rate: The dropout rate.
    """
    super().__init__()
    self.rate = rate
    self.safe_key = SafeKey(hk.next_rng_key())
  
  def __call__(self, x):
    """Applies dropout to the input tensor.
        
        Args:
            x: The input tensor to which dropout will be applied.
            
        Returns:
            The tensor after applying dropout.
    """
    self.safe_key, use_key = self.safe_key.split()
    return hk.dropout(use_key.get(), self.rate, x)


class EncLayer(hk.Module):
    """Encoder layer for the MPNN model."""
    
    def __init__(self, num_hidden: int, num_in: int, dropout: float = 0.1,
                 num_heads: Optional[int] = None, scale: float = 30,
                 name: Optional[str] = None) -> None:
        """Initializes the encoder layer with the given parameters.
        
        Args:
            num_hidden: The number of hidden units in the layer.
            num_in: The number of input features.
            dropout: The dropout rate.
            num_heads: The number of attention heads (unused in current implementation).
            scale: Scaling factor for normalization.
            name: Optional name for the layer.
        """
        super(EncLayer, self).__init__()
        # Store the number of hidden units, input features, and scaling factor.
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale

        # Initialize custom dropout modules with the specified dropout rate.
        self.dropout1 = dropout_cust(dropout)
        self.dropout2 = dropout_cust(dropout)
        self.dropout3 = dropout_cust(dropout)
        
        # Initialize layer normalization modules.
        self.norm1 = hk.LayerNorm(-1, create_scale=True, create_offset=True, name=name + '_norm1')
        self.norm2 = hk.LayerNorm(-1, create_scale=True, create_offset=True, name=name + '_norm2')
        self.norm3 = hk.LayerNorm(-1, create_scale=True, create_offset=True, name=name + '_norm3')

        # Initialize linear transformations for message passing.
        self.W1 = hk.Linear(num_hidden, with_bias=True, name=name + '_W1')
        self.W2 = hk.Linear(num_hidden, with_bias=True, name=name + '_W2')
        self.W3 = hk.Linear(num_hidden, with_bias=True, name=name + '_W3')
        self.W11 = hk.Linear(num_hidden, with_bias=True, name=name + '_W11')
        self.W12 = hk.Linear(num_hidden, with_bias=True, name=name + '_W12')
        self.W13 = hk.Linear(num_hidden, with_bias=True, name=name + '_W13')
        
        # Define the activation function to be used in the network.
        self.act = Gelu
        
        # Initialize a position-wise feedforward network.
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4, name=name + '_dense')

    def __call__(self, h_V: jnp.ndarray, h_E: jnp.ndarray, E_idx: jnp.ndarray,
                 mask_V: Optional[jnp.ndarray] = None, mask_attend: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass of the encoder layer.
        
        Args:
            h_V: Node features tensor.
            h_E: Edge features tensor.
            E_idx: Edge indices tensor.
            mask_V: Optional mask for valid nodes.
            mask_attend: Optional mask for attention mechanism.
            
        Returns:
            A tuple containing updated node and edge features tensors.
        """

        # Concatenate node features with edge features for message passing.
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        
        # Expand node features to match the shape of edge features for concatenation.
        h_V_expand = jnp.tile(jnp.expand_dims(h_V, -2), [1, h_EV.shape[-2], 1])
        
        # Concatenate expanded node features with edge features.
        h_EV = jnp.concatenate([h_V_expand, h_EV], -1)

        # Apply a series of linear transformations with activation functions to the concatenated features.
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        
        # If an attention mask is provided, apply it to the messages.
        if mask_attend is not None:
            h_message = jnp.expand_dims(mask_attend, -1) * h_message
        
        # Sum the messages and scale down.
        dh = jnp.sum(h_message, -2) / self.scale
        
        # Apply layer normalization and dropout to the node features, then add the scaled messages.
        h_V = self.norm1(h_V + self.dropout1(dh))

        # Apply the position-wise feedforward network to the node features.
        dh = self.dense(h_V)
        
        # Apply another layer normalization and dropout to the node features.
        h_V = self.norm2(h_V + self.dropout2(dh))
        
        # If a validity mask for nodes is provided, apply it to the node features.
        if mask_V is not None:
            mask_V = jnp.expand_dims(mask_V, -1)
            h_V = mask_V * h_V

        # Repeat the concatenation and message passing for the edge features.
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = jnp.tile(jnp.expand_dims(h_V, -2), [1, h_EV.shape[-2], 1])
        h_EV = jnp.concatenate([h_V_expand, h_EV], -1)

        # Apply a second series of linear transformations with activation functions to the concatenated features.
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        
        # Apply layer normalization and dropout to the edge features, then add the messages.
        h_E = self.norm3(h_E + self.dropout3(h_message))
        
        # Return the updated node and edge features tensors.
        return h_V, h_E

class DecLayer(hk.Module):
  """Decoder layer for the MPNN model."""
    
  def __init__(self, num_hidden: int, num_in: int, dropout: float = 0.1,
                 num_heads: Optional[int] = None, scale: float = 30,
                 name: Optional[str] = None) -> None:
    """Initializes the decoder layer with the given parameters.
        
        Args:
            num_hidden: The number of hidden units in the layer.
            num_in: The number of input features.
            dropout: The dropout rate.
            num_heads: The number of attention heads (unused in current implementation).
            scale: Scaling factor for normalization.
            name: Optional name for the layer.
    """
    super(DecLayer, self).__init__()
    self.num_hidden = num_hidden
    self.num_in = num_in
    self.scale = scale
    self.dropout1 = dropout_cust(dropout)
    self.dropout2 = dropout_cust(dropout)
    self.norm1 = hk.LayerNorm(-1, create_scale=True, create_offset=True,
                  name=name + '_norm1')
    self.norm2 = hk.LayerNorm(-1, create_scale=True, create_offset=True,
                  name=name + '_norm2')

    self.W1 = hk.Linear(num_hidden, with_bias=True, name=name + '_W1')
    self.W2 = hk.Linear(num_hidden, with_bias=True, name=name + '_W2')
    self.W3 = hk.Linear(num_hidden, with_bias=True, name=name + '_W3')
    self.act = Gelu
    self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4,
                       name=name + '_dense')

  
  def __call__(self, h_V: jnp.ndarray, h_E: jnp.ndarray,
             mask_V: Optional[jnp.ndarray] = None, mask_attend: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """Forward pass of the decoder layer.
    
    Args:
        h_V: Node features tensor.
        h_E: Edge features tensor.
        mask_V: Optional mask for valid nodes.
        mask_attend: Optional mask for attention mechanism.
        
    Returns:
        Updated node features tensor after processing by the decoder layer.
    """

    # Expand the node features tensor to match the shape of the edge features tensor for concatenation.
    h_V_expand = jnp.tile(jnp.expand_dims(h_V, -2), [1, h_E.shape[-2], 1])
    
    # Concatenate the expanded node features with the edge features along the last dimension.
    h_EV = jnp.concatenate([h_V_expand, h_E], -1)

    # Apply a series of linear transformations with GELU activation functions to the concatenated features.
    h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
    
    # If an attention mask is provided, apply it to the messages.
    if mask_attend is not None:
        h_message = jnp.expand_dims(mask_attend, -1) * h_message
    
    # Sum the messages along the second-to-last dimension and scale down.
    dh = jnp.sum(h_message, -2) / self.scale

    # Apply layer normalization and dropout to the node features, then add the scaled messages.
    h_V = self.norm1(h_V + self.dropout1(dh))

    # Apply a position-wise feedforward network to the node features.
    dh = self.dense(h_V)
    
    # Apply another layer normalization and dropout to the node features.
    h_V = self.norm2(h_V + self.dropout2(dh))

    # If a validity mask for nodes is provided, apply it to the node features.
    if mask_V is not None:
        mask_V = jnp.expand_dims(mask_V, -1)
        h_V = mask_V * h_V
    
    # Return the updated node features tensor.
    return h_V 

class PositionWiseFeedForward(hk.Module):
    """Position-wise feedforward neural network for the MPNN model."""
    
    def __init__(self, num_hidden: int, num_ff: int, name: Optional[str] = None) -> None:
        """Initializes the position-wise feedforward network with the given parameters.
        
        Args:
            num_hidden: The number of hidden units in the layer.
            num_ff: The size of the inner layer in the feedforward network.
            name: Optional name for the layer.
        """
        super(PositionWiseFeedForward, self).__init__()
        # Initialize a linear transformation for the input with the specified number of features.
        self.W_in = hk.Linear(num_ff, with_bias=True, name=name + '_W_in')
        # Initialize a linear transformation for the output with the specified number of hidden units.
        self.W_out = hk.Linear(num_hidden, with_bias=True, name=name + '_W_out')
        # Define the activation function to be used between the linear transformations.
        self.act = Gelu

    def __call__(self, h_V: jnp.ndarray) -> jnp.ndarray:
        """Forward pass of the position-wise feedforward network.
        
        Args:
            h_V: Node features tensor.
            
        Returns:
            Updated node features tensor after processing by the feedforward network.
        """
        # Apply the GELU activation function to the output of the first linear transformation.
        h = self.act(self.W_in(h_V), approximate=False)
        # Apply the second linear transformation to the activated tensor.
        h = self.W_out(h)
        # Return the transformed tensor, which represents the updated node features.
        return h

class PositionalEncodings(hk.Module):
    """Generates positional encodings for the MPNN model."""
    
    def __init__(self, num_embeddings: int, max_relative_feature: int = 32) -> None:
        """Initializes the positional encodings module with the given parameters.
        
        Args:
            num_embeddings: The dimensionality of the positional embeddings.
            max_relative_feature: The maximum relative position for which embeddings will be generated.
        """
        super(PositionalEncodings, self).__init__()
        # Store the number of embeddings and the maximum relative feature.
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        # Initialize a linear layer to transform one-hot encoded positions into embeddings.
        self.linear = hk.Linear(num_embeddings, name='embedding_linear')

    def __call__(self, offset: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        """Computes positional encodings based on the provided offset and mask.
        
        Args:
            offset: A tensor containing the relative positions.
            mask: A binary mask tensor indicating valid positions.
            
        Returns:
            A tensor containing the positional encodings.
        """
        # Adjust the offset by the maximum relative feature and clip it to the valid range.
        # Apply the mask to the offset, setting invalid positions to a value beyond the valid range.
        d = jnp.clip(offset + self.max_relative_feature, 0, 2 * self.max_relative_feature) * mask + \
            (1 - mask) * (2 * self.max_relative_feature + 1)
        
        # Convert the adjusted and masked offset to one-hot encodings.
        d_onehot = jax.nn.one_hot(d, 2 * self.max_relative_feature + 1 + 1)
        
        # Apply the linear layer to the one-hot encodings to obtain the positional encodings.
        E = self.linear(d_onehot)
        
        # Return the positional encodings.
        return E

class RunModel:
    """Wrapper class to run the MPNN model for both sampling and scoring."""
    
    def __init__(self, config: dict) -> None:
        """Initializes the RunModel with the given configuration.
        
        Args:
            config: A dictionary containing the configuration parameters for the MPNN model.
        """
        # Store the configuration dictionary for the MPNN model.
        self.config = config

        # Define a private method to forward the scoring process.
        def _forward_score(inputs):
            # Instantiate the ProteinMPNN model using the stored configuration.
            model = ProteinMPNN(**self.config)
            # Call the score method of the model with the provided inputs.
            return model.score(inputs)
        # Transform the private scoring method using Haiku and store the transformed apply function.
        self.score = hk.transform(_forward_score).apply

        # Define a private method to forward the sampling process.
        def _forward_sample(inputs):
            # Instantiate the ProteinMPNN model using the stored configuration.
            model = ProteinMPNN(**self.config)
            # Call the sample method of the model with the provided inputs.
            return model.sample(inputs)
        # Transform the private sampling method using Haiku and store the transformed apply function.
        self.sample = hk.transform(_forward_sample).apply

class RunModelDual:
  """ Wrapper class to run the Dual Backbone MPNN model for both sampling and scoring."""
  def __init__(self, config: dict) -> None:
      """ Initializes the RunModelDual with the given configuration.
      
      Args:
          config: A dictionary containing the configuration parameters for the MPNN model.
      """
      # Store the configuration dictionary for the MPNN model.
      self.config = config
      
      # Define a private method to forward the scoring process.
      def _forward_score(inputs):
        # Instantiate the DualProteinMPNN model using the stored configuration.
        model = DualProteinMPNN(**self.config)
        # Call the score method of the model with the provided inputs.
        return model.score(inputs)
      # Transform the private scoring method using Haiku and store the transformed apply function.
      self.score = hk.transform(_forward_score).apply
      
      # Define a private method to forward the sampling process.
      def _forward_sample(inputs):
        # Instantiate the DualProteinMPNN model using the stored configuration.
        model = DualProteinMPNN(**self.config)
        # Call the sample method of the model with the provided inputs.
        return model.sample(inputs)
      # Transform the private sampling method using Haiku and store the transformed apply function.
      self.sample = hk.transform(_forward_sample).apply
      
class ProteinFeatures(hk.Module):
    """Extracts protein features for the MPNN model."""
    
    def __init__(self, edge_features: int, node_features: int,
                 num_positional_embeddings: int = 16,
                 num_rbf: int = 16, top_k: int = 30,
                 augment_eps: float = 0., num_chain_embeddings: int = 16):
        """Initializes the ProteinFeatures module with the given parameters.
        
        Args:
            edge_features: The number of features for each edge.
            node_features: The number of features for each node.
            num_positional_embeddings: The number of positional embeddings.
            num_rbf: The number of radial basis functions.
            top_k: The number of top nearest neighbors to consider in the graph.
            augment_eps: Epsilon value for data augmentation.
            num_chain_embeddings: The number of chain embeddings (unused in current implementation).
        """
        super(ProteinFeatures, self).__init__()
        # Store the provided parameters as attributes of the class.
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps 
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        # Initialize the PositionalEncodings module for generating positional embeddings.
        self.embeddings = PositionalEncodings(num_positional_embeddings)
        # Initialize a linear layer for embedding edge features.
        self.edge_embedding = hk.Linear(edge_features, with_bias=False, name='edge_embedding')
        # Initialize a layer normalization module for normalizing edge features.
        self.norm_edges = hk.LayerNorm(-1, create_scale=True, create_offset=True, name='norm_edges')

        # Initialize a key for random number generation used in data augmentation.
        self.safe_key = SafeKey(hk.next_rng_key())

    # Define a private method to compute edge indices based on distances between atoms.
    def _get_edge_idx(self, X, mask, eps=1E-6):
        # Calculate pairwise distances between atoms, taking into account the mask.
        mask_2D = mask[..., None, :] * mask[..., :, None]
        dX = X[..., None, :, :] - X[..., :, None, :]
        D = jnp.sqrt(jnp.square(dX).sum(-1) + eps)
        # Mask the distances to ignore pairs involving masked atoms.
        D_masked = jnp.where(mask_2D, D, D.max(-1, keepdims=True))
        # Determine the top k nearest neighbors.
        k = min(self.top_k, X.shape[-2])
        return jax.lax.approx_min_k(D_masked, k, reduction_dimension=-1)[1]

    # Define a private method to compute radial basis function (RBF) features.
    def _rbf(self, D):
        # Define the range and resolution for the RBF.
        D_min, D_max, D_count = 2., 22., self.num_rbf
        D_mu = jnp.linspace(D_min, D_max, D_count)
        D_sigma = (D_max - D_min) / D_count
        # Compute the RBF features.
        return jnp.exp(-((D[..., None] - D_mu) / D_sigma)**2)

    # Define a private method to compute RBF features for neighboring atoms.
    def _get_rbf(self, A, B, E_idx):
        # Compute distances between neighboring atoms.
        D = jnp.sqrt(jnp.square(A[..., :, None, :] - B[..., None, :, :]).sum(-1) + 1e-6)
        # Select the distances for the top k neighbors.
        D_neighbors = jnp.take_along_axis(D, E_idx, 1)
        # Compute the RBF features for the selected distances.
        return self._rbf(D_neighbors)

    def __call__(self, I: dict) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Computes the edge features and edge indices for the input protein structure.
        
        Args:
            I: A dictionary containing the input features such as atom positions, masks, and residue indices.
            
        Returns:
            A tuple containing the edge features and edge indices tensors.
        """
        # Apply data augmentation if specified.
        if self.augment_eps > 0:
            self.safe_key, use_key = self.safe_key.split()
            X = I["X"] + self.augment_eps * jax.random.normal(use_key.get(), I["X"].shape)
        else:
            X = I["X"]
        
        # Swap axes to organize the atom positions for feature extraction.
        Y = X.swapaxes(0, 1)
        # If only backbone atoms are provided, compute the position of the C-beta atom.
        if Y.shape[0] == 4:
            b, c = (Y[1] - Y[0]), (Y[2] - Y[1])
            Cb = -0.58273431 * jnp.cross(b, c) + 0.56802827 * b - 0.54067466 * c + Y[1]
            Y = jnp.concatenate([Y, Cb[None]], 0)

        # Compute edge indices based on distances between C-alpha atoms.
        E_idx = self._get_edge_idx(Y[1], I["mask"])

        # Compute RBF features for all pairs of atoms defined in the 'edges' array.
        edges = jnp.array([[1, 1], [0, 0], [2, 2], [3, 3], [4, 4],
                           # ... (omitted for brevity)
                           [3, 0], [4, 0], [2, 4], [3, 4], [2, 3]])
        RBF_all = jax.vmap(lambda x: self._get_rbf(Y[x[0]], Y[x[1]], E_idx))(edges)
        RBF_all = RBF_all.transpose((1, 2, 0, 3))
        RBF_all = RBF_all.reshape(RBF_all.shape[:-2] + (-1,))

        # Compute residue index offsets for positional embeddings.
        if "offset" not in I:
            I["offset"] = I["residue_idx"][:, None] - I["residue_idx"][None, :]
        offset = jnp.take_along_axis(I["offset"], E_idx, 1)

        # Compute chain index offsets for positional embeddings.
        E_chains = (I["chain_idx"][:, None] == I["chain_idx"][None, :]).astype(int)
        E_chains = jnp.take_along_axis(E_chains, E_idx, 1)
        E_positional = self.embeddings(offset, E_chains)

        # Concatenate positional embeddings with RBF features to define edge features.
        E = jnp.concatenate((E_positional, RBF_all), -1)
        # Apply a linear transformation to the edge features.
        E = self.edge_embedding(E)
        # Normalize the edge features.
        E = self.norm_edges(E)
        
        # Return the edge features and edge indices.
        return E, E_idx
      
class EmbedToken(hk.Module):
    """Embedding layer for token representation in the MPNN model."""
    
    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        """Initializes the EmbedToken layer with the given vocabulary size and embedding dimension.
        
        Args:
            vocab_size: The size of the vocabulary (number of unique tokens).
            embed_dim: The dimensionality of the embeddings.
        """
        super().__init__()
        # Store the vocabulary size and embedding dimension.
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        # Initialize the weights for the embedding matrix with a truncated normal distribution.
        self.w_init = hk.initializers.TruncatedNormal()

    @property
    def embeddings(self) -> jnp.ndarray:
        """Retrieves the embedding matrix.
        
        Returns:
            The embedding matrix with shape (vocab_size, embed_dim).
        """
        # Get the embedding matrix parameter with the specified shape and initializer.
        return hk.get_parameter("W_s", [self.vocab_size, self.embed_dim], init=self.w_init)

    def __call__(self, arr: jnp.ndarray) -> jnp.ndarray:
        """Looks up embeddings for the input array of tokens.
        
        Args:
            arr: An array of token indices or one-hot encoded tokens.
            
        Returns:
            The corresponding embeddings for the input tokens.
        """
        # Check if the input array is composed of integer token indices.
        if jnp.issubdtype(arr.dtype, jnp.integer):
            # Convert the array of token indices to one-hot encoded vectors.
            one_hot = jax.nn.one_hot(arr, self.vocab_size)
        else:
            # If the input is already one-hot encoded, use it as is.
            one_hot = arr
        # Perform a tensor dot product between the one-hot encoded tokens and the embedding matrix.
        # This operation effectively looks up the embedding for each token in the input array.
        return jnp.tensordot(one_hot, self.embeddings, 1)

class ProteinMPNN(hk.Module, mpnn_sample, mpnn_score):
    """Protein Message Passing Neural Network (MPNN) model."""
    
    def __init__(self, num_letters: int, node_features: int, edge_features: int,
                 hidden_dim: int, num_encoder_layers: int = 3,
                 num_decoder_layers: int = 3, vocab: int = 21, k_neighbors: int = 64,
                 augment_eps: float = 0.05, dropout: float = 0.1) -> None:
        """Initializes the ProteinMPNN with the given parameters.
        
        Args:
            num_letters: The number of amino acid types (including padding).
            node_features: The number of features for each node.
            edge_features: The number of features for each edge.
            hidden_dim: The size of the hidden dimension in the network.
            num_encoder_layers: The number of encoder layers.
            num_decoder_layers: The number of decoder layers.
            vocab: The size of the vocabulary (number of unique tokens).
            k_neighbors: The number of nearest neighbors to consider in the graph.
            augment_eps: Epsilon value for data augmentation.
            dropout: The dropout rate.
        """
        super(ProteinMPNN, self).__init__()
        # Store the provided hyperparameters as attributes of the class.
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim

        # Initialize the feature extraction module for processing protein structures.
        self.features = ProteinFeatures(edge_features, node_features,
                                        top_k=k_neighbors, augment_eps=augment_eps)

        # Initialize a linear layer for embedding edge features.
        self.W_e = hk.Linear(hidden_dim, with_bias=True, name='W_e')
        # Initialize the token embedding module for embedding sequence information.
        self.W_s = EmbedToken(vocab_size=vocab, embed_dim=hidden_dim)

        # Create a list of encoder layers, each responsible for processing features in the encoder part of the model.
        self.encoder_layers = [
            EncLayer(hidden_dim, hidden_dim * 2, dropout=dropout, name='enc' + str(i))
            for i in range(num_encoder_layers)
        ]

        # Create a list of decoder layers, each responsible for processing features in the decoder part of the model.
        self.decoder_layers = [
            DecLayer(hidden_dim, hidden_dim * 3, dropout=dropout, name='dec' + str(i))
            for i in range(num_decoder_layers)
        ]
        # Initialize a linear layer for producing output logits from the final hidden state.
        self.W_out = hk.Linear(num_letters, with_bias=True, name='W_out')
        
from .sample import mpnn_sample_dual_backbone
from .score import mpnn_score_dual_backbone

class DualProteinMPNN(hk.Module, mpnn_sample_dual_backbone, mpnn_score_dual_backbone):
    """Dual Backbone Protein Message Passing Neural Network (MPNN) model."""
    
    def __init__(self, num_letters: int, node_features: int, edge_features: int,
                 hidden_dim: int, num_encoder_layers: int = 3,
                 num_decoder_layers: int = 3, vocab: int = 21, k_neighbors: int = 64,
                 augment_eps: float = 0.05, dropout: float = 0.1) -> None:
        """Initializes the DualProteinMPNN with the given parameters.
        
        Args:
            num_letters: The number of amino acid types (including padding).
            node_features: The number of features for each node.
            edge_features: The number of features for each edge.
            hidden_dim: The size of the hidden dimension in the network.
            num_encoder_layers: The number of encoder layers.
            num_decoder_layers: The number of decoder layers.
            vocab: The size of the vocabulary (number of unique tokens).
            k_neighbors: The number of nearest neighbors to consider in the graph.
            augment_eps: Epsilon value for data augmentation.
            dropout: The dropout rate.
        """
        super(DualProteinMPNN, self).__init__()
        # Store the provided hyperparameters as attributes of the class.
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim

        # Initialize the feature extraction module for processing protein structures.
        self.features = ProteinFeatures(edge_features, node_features,
                                        top_k=k_neighbors, augment_eps=augment_eps)

        # Initialize a linear layer for embedding edge features.
        self.W_e = hk.Linear(hidden_dim, with_bias=True, name='W_e')
        # Initialize the token embedding module for embedding sequence information.
        self.W_s = EmbedToken(vocab_size=vocab, embed_dim=hidden_dim)

        # Create a list of encoder layers, each responsible for processing features in the encoder part of the model.
        self.encoder_layers = [
            EncLayer(hidden_dim, hidden_dim * 2, dropout=dropout, name='enc' + str(i))
            for i in range(num_encoder_layers)
        ]

        # Create a list of decoder layers, each responsible for processing features in the decoder part of the model.
        self.decoder_layers = [
            DecLayer(hidden_dim, hidden_dim * 3, dropout=dropout, name='dec' + str(i))
            for i in range(num_decoder_layers)
        ]
        # Initialize a linear layer for producing output logits from the final hidden state.
        self.W_out = hk.Linear(num_letters, with_bias=True, name='W_out')