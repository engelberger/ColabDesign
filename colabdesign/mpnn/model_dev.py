import jax
import jax.numpy as jnp
import numpy as np
import re
import copy
import random
import os
import joblib
from typing import Optional, Union, List, Dict, Any


from .weights import __file__ as mpnn_path
from .modules import RunModel
from .modules_dev import RunModelDual

from colabdesign.shared.prep import prep_pos
from colabdesign.shared.utils import Key, copy_dict

# borrow some stuff from AfDesign
from colabdesign.af.prep import prep_pdb
from colabdesign.af.alphafold.common import protein, residue_constants
aa_order = residue_constants.restype_order
order_aa = {b:a for a,b in aa_order.items()}

from scipy.special import softmax, log_softmax

class mk_mpnn_model():
  """
    Class for creating and managing an MPNN model for protein design.
    
    Attributes:
        _model (RunModel): The underlying MPNN model.
        _model_params (dict): Parameters of the MPNN model.
        _num (int): Number of sequences to generate or score.
        _inputs (dict): Input features for the model.
        _tied_lengths (bool): Whether the lengths of sequences are tied (for homooligomers).
        _lengths (list): List of sequence lengths for each chain.
        _len (int): Total length of the concatenated sequences.
        pdb (dict): Parsed PDB data.
  """
  def __init__(self, model_name: str = "v_48_020",
               backbone_noise: float = 0.0, dropout: float = 0.0,
               seed: Optional[int] = None, verbose: bool = False):
    """
        Initializes the MPNN model with the specified configuration.
        
        Args:
            model_name (str): Name of the model to load.
            backbone_noise (float): Standard deviation of Gaussian noise to add to backbone positions.
            dropout (float): Dropout rate for the model.
            seed (int, optional): Random seed for reproducibility.
            verbose (bool): Whether to print verbose output.
        """
    
    # configure model
    path = os.path.join(os.path.dirname(mpnn_path), f'{model_name}.pkl')    
    checkpoint = joblib.load(path)
    config = {'num_letters': 21,
              'node_features': 128,
              'edge_features': 128,
              'hidden_dim': 128,
              'num_encoder_layers': 3,
              'num_decoder_layers': 3,
              'augment_eps': backbone_noise,
              'k_neighbors': checkpoint['num_edges'],
              'dropout': dropout}
    
    self._model = RunModel(config)
    
    # load model params
    self._model_params = jax.tree_map(jnp.array, checkpoint['model_state_dict'])
    
    self._setup()
    self.set_seed(seed)

    self._num = 1
    self._inputs = {}
    self._tied_lengths = False

  def prep_inputs(self, pdb_filename: Optional[str] = None, chain: Optional[str] = None,
                    homooligomer: bool = False, ignore_missing: bool = True,
                    fix_pos: Optional[Union[List[int], str]] = None, inverse: bool = False,
                    rm_aa: Optional[str] = None, verbose: bool = False, **kwargs) -> None:
    """
        Prepares input features from a PDB file for the MPNN model.
        
        Args:
            pdb_filename (str, optional): Path to the PDB file.
            chain (str, optional): Chain identifier to use from the PDB file.
            homooligomer (bool): Whether the input is a homooligomer.
            ignore_missing (bool): Whether to ignore missing residues.
            fix_pos (str, optional): Positions to fix during design.
            inverse (bool): Whether to fix all positions except those specified.
            rm_aa (str, optional): Amino acids to remove from consideration.
            verbose (bool): Whether to print verbose output.
            **kwargs: Additional keyword arguments.
    """
    pdb = prep_pdb(pdb_filename, chain, ignore_missing=ignore_missing)
    atom_idx = tuple(residue_constants.atom_order[k] for k in ["N","CA","C","O"])
    chain_idx = np.concatenate([[n]*l for n,l in enumerate(pdb["lengths"])])
    self._lengths = pdb["lengths"]
    L = sum(self._lengths)

    self._inputs = {"X":           pdb["batch"]["all_atom_positions"][:,atom_idx],
                    "mask":        pdb["batch"]["all_atom_mask"][:,1],
                    "S":           pdb["batch"]["aatype"],
                    "residue_idx": pdb["residue_index"],
                    "chain_idx":   chain_idx,
                    "lengths":     np.array(self._lengths),
                    "bias":        np.zeros((L,20))}
    

    if rm_aa is not None:
      for aa in rm_aa.split(","):
        self._inputs["bias"][...,aa_order[aa]] -= 1e6
    
    if fix_pos is not None:
      p = prep_pos(fix_pos, **pdb["idx"])["pos"]
      if inverse:
        p = np.delete(np.arange(L),p)
      self._inputs["fix_pos"] = np.full(L,False)
      self._inputs["fix_pos"][p] = True
      self._inputs["bias"][p] = 1e7 * np.eye(21)[self._inputs["S"]][p,:20]
    
    if homooligomer:
      assert min(self._lengths) == max(self._lengths)
      self._tied_lengths = True
      self._len = self._lengths[0]
    else:
      self._tied_lengths = False    
      self._len = sum(self._lengths)  

    self.pdb = pdb
    
    if verbose:
      print("lengths", self._lengths)
      if "fix_pos" in self._inputs:
        print("the following positions will be fixed:")
        print(np.where(self._inputs["fix_pos"])[0])

  def get_af_inputs(self, af: protein.Protein) -> None:
    """
        Retrieves input features from an AlphaFold model.
        
        Args:
            af (AlphaFoldModel): An AlphaFold model object.
    """

    self._lengths = af._lengths
    self._len = af._len

    self._inputs["residue_idx"] = af._inputs["residue_index"]
    self._inputs["chain_idx"]   = af._inputs["asym_id"]
    self._inputs["lengths"]     = np.array(self._lengths)

    # set bias
    L = sum(self._lengths)
    self._inputs["bias"] = np.zeros((L,20))
    self._inputs["bias"][-af._len:] = af._inputs["bias"]
    
    if "offset" in af._inputs:
      self._inputs["offset"] = af._inputs["offset"]

    if "batch" in af._inputs:
      atom_idx = tuple(residue_constants.atom_order[k] for k in ["N","CA","C","O"])
      batch = af._inputs["batch"]
      self._inputs["X"]    = batch["all_atom_positions"][:,atom_idx]
      self._inputs["mask"] = batch["all_atom_mask"][:,1]
      self._inputs["S"]    = batch["aatype"]

    # fix positions
    if af.protocol == "binder":
      fix_pos = np.array([True]*self._target_len + [False] * self._binder_len)
    else:
      fix_pos = af._inputs["fix_pos"]
    
    self._inputs["fix_pos"] = fix_pos
    self._inputs["bias"][fix_pos] = 1e7 * np.eye(21)[self._inputs["S"]][fix_pos,:20]

    # tie positions
    if af._args["copies"] > 1:
      assert min(self._lengths) == max(self._lengths)
      self._tied_lengths = True
    else:
      self._tied_lengths = False

  def sample(self, num: int = 1, batch: int = 1, temperature: float = 0.1,
             rescore: bool = False, **kwargs) -> Dict[str, Any]:
    """
        Samples sequences from the MPNN model.
        
        Args:
            num (int): Number of sequences to sample.
            batch (int): Batch size for parallel sampling.
            temperature (float): Sampling temperature.
            rescore (bool): Whether to rescore the sequences after sampling.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Dict[str, Any]: A dictionary containing sampled sequences and associated data.
        """
    O = []
    for _ in range(num):
      O.append(self.sample_parallel(batch, temperature, rescore, **kwargs))
    return jax.tree_map(lambda *x:np.concatenate(x,0),*O)    

  def sample_parallel(self, batch: int = 10, temperature: float = 0.1,
                        rescore: bool = False, **kwargs) -> Dict[str, Any]:
    """
    Sample new sequences in parallel.

    Args:
      batch (int): Batch size for parallel sampling.
      temperature (float): Temperature parameter for sampling.
      rescore (bool): Whether to rescore the sampled sequences.
      **kwargs: Additional keyword arguments.

    Returns:
      Dict[str, Any]: A dictionary containing sampled sequences and associated data.
    """
    I = copy_dict(self._inputs)
    I.update(kwargs)
    key = I.pop("key",self.key())
    keys = jax.random.split(key,batch)
    O = self._sample_parallel(keys, I, temperature, self._tied_lengths)
    if rescore:
      O = self._rescore_parallel(keys, I, O["S"], O["decoding_order"])
    O = jax.tree_map(np.array, O)

    # process outputs to human-readable form
    O.update(self._get_seq(O))
    O.update(self._get_score(I,O))
    return O

  def _get_seq(self, O: Dict[str, jnp.ndarray]) -> Dict[str, np.ndarray]:
    """
    Convert one-hot encoded sequence to amino acid sequence.

    Args:
      O (Dict[str, jnp.ndarray]): Output dictionary containing one-hot encoded sequences.

    Returns:
      Dict[str, np.ndarray]: A dictionary containing the amino acid sequences.
    """
    def split_seq(seq):
      if len(self._lengths) > 1:
        seq = "".join(np.insert(list(seq),np.cumsum(self._lengths[:-1]),"/"))
        if self._tied_lengths:
          seq = seq.split("/")[0]
      return seq
    seqs, S = [], O["S"].argmax(-1)
    if S.ndim == 1: S = [S]
    for s in S:
      seq = "".join([order_aa[a] for a in s])
      seq = split_seq(seq)
      seqs.append(seq)
    
    return {"seq": np.array(seqs)}

  def _get_score(self, I: Dict[str, jnp.ndarray], O: Dict[str, jnp.ndarray]) -> Dict[str, np.ndarray]:
    """
    Compute the logits to score and sequence recovery.

    Args:
      I (Dict[str, jnp.ndarray]): Input dictionary.
      O (Dict[str, jnp.ndarray]): Output dictionary containing logits.

    Returns:
      Dict[str, np.ndarray]: A dictionary containing scores and sequence recovery information.
    """
    mask = I["mask"].copy()
    if "fix_pos" in I:
      mask[I["fix_pos"]] = 0

    log_q = log_softmax(O["logits"],-1)[...,:20]
    q = softmax(O["logits"][...,:20],-1)
    if "S" in O:
      S = O["S"][...,:20]
      score = -(S * log_q).sum(-1)
      seqid = S.argmax(-1) == self._inputs["S"]
    else:
      score = -(q * log_q).sum(-1)
      seqid = np.zeros_like(score)
      
    score = (score * mask).sum(-1) / (mask.sum() + 1e-8)
    seqid = (seqid * mask).sum(-1) / (mask.sum() + 1e-8)

    return {"score":score, "seqid":seqid}

  def score(self, seq: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Score a given sequence.

        Args:
        seq (Optional[str]): Amino acid sequence to score.

        Returns:
        Dict[str, Any]: A dictionary containing the score and other related data.
        """
        # Copy the current input features of the model.
        I = copy_dict(self._inputs)
        
        # If a sequence is provided, update the input features with the sequence information.
        if seq is not None:
            # If the model is configured for homooligomers and the sequence length matches,
            # replicate the sequence for each chain.
            if self._tied_lengths and len(seq) == self._lengths[0]:
                seq = seq * len(self._lengths)

            # Determine the positions to update based on the provided sequence and fixed positions.
            p = np.arange(I["S"].shape[0])
            if "fix_pos" in I and len(seq) == (I["S"].shape[0] - sum(I["fix_pos"])):
                p = p[I["fix_pos"]]
            
            # Update the sequence features with the provided sequence.
            I["S"][p] = np.array([aa_order.get(aa, -1) for aa in seq])
        
        # Update the input features with any additional keyword arguments.
        I.update(kwargs)
        # Remove the key from the input features and obtain a new random key.
        key = I.pop("key", self.key())
        # Score the sequence using the MPNN model.
        O = jax.tree_map(np.array, self._score(**I, key=key))
        # Update the output with the calculated scores.
        O.update(self._get_score(I, O))
        
        # Return the scores and sequence identities.
        return O

  def get_logits(self, **kwargs) -> jnp.ndarray:
        """
        Retrieves the logits for the current sequence or a provided sequence using the MPNN model.
        The `get_logits` method is a convenience function that calls the `score` method and 
        extracts the logits from the output, which represent the unnormalized log probabilities 
        of the amino acids at each position in the sequence.
        
        Args:
            **kwargs: Additional keyword arguments to be passed to the scoring function.
            
        Returns:
            jnp.ndarray: An array of logits corresponding to the sequence probabilities.
        """
        # Call the score method and return the logits from the output dictionary.
        return self.score(**kwargs)["logits"]

  def get_unconditional_logits(self, **kwargs) -> jnp.ndarray:
        """
        Retrieves the logits without any autoregressive masking, effectively treating the
        sequence generation as unconditional.
        
        The `get_unconditional_logits` method retrieves logits in an unconditional manner 
        by setting the autoregressive mask to all zeros, indicating that the generation of 
        each amino acid does not depend on the previously generated amino acids. This is 
        useful for tasks where the sequence is generated independently at each position.
        
        Args:
            **kwargs: Additional keyword arguments to be passed to the scoring function.
            
        Returns:
            jnp.ndarray: An array of unconditional logits corresponding to the sequence probabilities.
        """
        # Determine the length of the sequences from the input features.
        L = self._inputs["X"].shape[0]
        # Create an autoregressive mask filled with zeros, indicating no positions are conditioned on previous ones.
        kwargs["ar_mask"] = np.zeros((L, L))
        # Call the score method with the updated kwargs and return the logits.
        return self.score(**kwargs)["logits"]

  def set_seed(self, seed: Optional[int] = None) -> None:
    """
    Set the random seed for reproducibility.

    Args:
      seed (Optional[int]): Random seed.

    Returns:
      None
    """
    np.random.seed(seed=seed)
    self.key = Key(seed=seed).get

  def _setup(self):
    """
      Sets up the model for sampling and scoring by compiling the necessary functions.
      This method prepares the internal functions for efficient execution using JAX's JIT compilation.
      The `_setup` method prepares the model for efficient execution by compiling the scoring 
      and sampling functions using JAX's just-in-time (JIT) compilation. 
      The nested functions `_score`, `_sample`, `_sample_parallel`, and `_rescore_parallel` 
      are defined within `_setup` and are responsible for scoring sequences, sampling new sequences, 
      and handling parallel execution. These functions are then vectorized and compiled to optimize 
      performance. The docstrings explain the purpose and functionality of each nested function and 
      the arguments they accept.
    """
    def _score(X, mask, residue_idx, chain_idx, key, **kwargs):
      """
            Scores the given sequences using the MPNN model.
            
            Args:
                X: Node features tensor.
                mask: Mask tensor for valid positions.
                residue_idx: Residue index tensor.
                chain_idx: Chain index tensor.
                key: A key for random number generation.
                **kwargs: Additional keyword arguments.
                
            Returns:
                A dictionary containing the logits and optionally the sequences.
      """
      I = {'X': X,
           'mask': mask,
           'residue_idx': residue_idx,
           'chain_idx': chain_idx}
      I.update(kwargs)

      # define decoding order
      if "decoding_order" not in I:
        key, sub_key = jax.random.split(key)
        randn = jax.random.uniform(sub_key, (I["X"].shape[0],))    
        randn = jnp.where(I["mask"], randn, randn+1)
        if "fix_pos" in I: 
          randn = jnp.where(I["fix_pos"],randn-1,randn)
        I["decoding_order"] = randn.argsort()

      for k in ["S","bias"]:
        if k in I: I[k] = _aa_convert(I[k])

      O = self._model.score(self._model_params, key, I)
      O["S"] = _aa_convert(O["S"], rev=True)
      O["logits"] = _aa_convert(O["logits"], rev=True)
      return O
    
    def _sample(X, mask, residue_idx, chain_idx, key,
                    temperature=0.1, tied_lengths=False, **kwargs):
      """
            Samples new sequences using the MPNN model.
            
            Args:
                X: Node features tensor.
                mask: Mask tensor for valid positions.
                residue_idx: Residue index tensor.
                chain_idx: Chain index tensor.
                key: A key for random number generation.
                temperature: Sampling temperature.
                tied_lengths: Whether the lengths of sequences are tied (for homooligomers).
                **kwargs: Additional keyword arguments.
                
            Returns:
                A dictionary containing the sampled sequences and logits.
      """
      I = {'X': X,
           'mask': mask,
           'residue_idx': residue_idx,
           'chain_idx': chain_idx,
           'temperature': temperature}
      I.update(kwargs)

      # define decoding order
      if "decoding_order" in I:
        if I["decoding_order"].ndim == 1:
          I["decoding_order"] = I["decoding_order"][:,None]
      else:
        key, sub_key = jax.random.split(key)
        randn = jax.random.uniform(sub_key, (I["X"].shape[0],))    
        randn = jnp.where(I["mask"], randn, randn+1)
        if "fix_pos" in I:
          randn = jnp.where(I["fix_pos"],randn-1,randn)
        if tied_lengths:
          copies = I["lengths"].shape[0]
          decoding_order_tied = randn.reshape(copies,-1).mean(0).argsort()
          I["decoding_order"] = jnp.arange(I["X"].shape[0]).reshape(copies,-1).T[decoding_order_tied]
        else:
          I["decoding_order"] = randn.argsort()[:,None]

      for k in ["S","bias"]:
        if k in I: I[k] = _aa_convert(I[k])
      
      O = self._model.sample(self._model_params, key, I)
      O["S"] = _aa_convert(O["S"], rev=True)
      O["logits"] = _aa_convert(O["logits"], rev=True)
      return O

    # Compile the _score function using JAX's JIT for faster execution.
    self._score = jax.jit(_score)
    # Compile the _sample function using JAX's JIT for faster execution.
    self._sample = jax.jit(_sample, static_argnames=["tied_lengths"])

    def _sample_parallel(key, inputs, temperature, tied_lengths=False):
      """
            Helper function to sample new sequences in parallel using the MPNN model.
            
            Args:
                key: A key for random number generation.
                inputs: A dictionary containing the input features.
                temperature: Sampling temperature.
                tied_lengths: Whether the lengths of sequences are tied (for homooligomers).
                
            Returns:
                A dictionary containing the sampled sequences and logits.
      """
      inputs.pop("temperature",None)
      inputs.pop("key",None)
      return _sample(**inputs, key=key, temperature=temperature, tied_lengths=tied_lengths)
    # Vectorize the _sample_parallel function to enable parallel execution.
    fn = jax.vmap(_sample_parallel, in_axes=[0, None, None, None])
    # Compile the vectorized _sample_parallel function using JAX's JIT for faster execution.
    self._sample_parallel = jax.jit(fn, static_argnames=["tied_lengths"])

    def _rescore_parallel(key, inputs, S, decoding_order):
      """
            Helper function to rescore sequences in parallel using the MPNN model.
            
            Args:
                key: A key for random number generation.
                inputs: A dictionary containing the input features.
                S: One-hot encoded sequences tensor.
                decoding_order: Tensor representing the order in which positions were decoded.
                
            Returns:
                A dictionary containing the rescored logits and sequences.
      """
      inputs.pop("S",None)
      inputs.pop("decoding_order",None)
      inputs.pop("key",None)
      return _score(**inputs, key=key, S=S, decoding_order=decoding_order)
    # Vectorize the _rescore_parallel function to enable parallel execution.
    fn = jax.vmap(_rescore_parallel, in_axes=[0, None, 0, 0])
    # Compile the vectorized _rescore_parallel function using JAX's JIT for faster execution.
    self._rescore_parallel = jax.jit(fn)
#######################################################################################
class mk_mpnn_model_dual():
  def __init__(self, model_name: str = "v_48_020",
               backbone_noise: float = 0.0, dropout: float = 0.0,
               seed: Optional[int] = None, verbose: bool = False):
    # Configure model for dual backbone
    path = os.path.join(os.path.dirname(mpnn_path), f'{model_name}.pkl')    
    checkpoint = joblib.load(path)
    config = {'num_letters': 21,
              'node_features': 128,
              'edge_features': 128,
              'hidden_dim': 128,
              'num_encoder_layers': 3,
              'num_decoder_layers': 3,
              'augment_eps': backbone_noise,
              'k_neighbors': checkpoint['num_edges'],
              'dropout': dropout}
    
    self._model = RunModelDual(config)
    
    # Load model params
    self._model_params = jax.tree_map(jnp.array, checkpoint['model_state_dict'])
    
    # Initialize other attributes
    self._setup_dual_backbone()
    self.set_seed(seed)

    self._num = 1
    self._inputs = {}
    self._tied_lengths = False

    # TODO : Finish the init method

  def prep_inputs(self, pdb_filename: Optional[str] = None, chain: Optional[str] = None,
                    homooligomer: bool = False, ignore_missing: bool = True,
                    fix_pos: Optional[Union[List[int], str]] = None, inverse: bool = False,
                    rm_aa: Optional[str] = None, verbose: bool = False, **kwargs) -> None:
    """
        Prepares input features from a PDB file for the MPNN model.
        
    """
    pdb = prep_pdb(pdb_filename, chain, ignore_missing=ignore_missing)
    atom_idx = tuple(residue_constants.atom_order[k] for k in ["N","CA","C","O"])
    chain_idx = np.concatenate([[n]*l for n,l in enumerate(pdb["lengths"])])
    self._lengths = pdb["lengths"]
    L = sum(self._lengths)
    
    self._inputs = {"X1":           pdb["batch"]["all_atom_positions"][:,atom_idx],
                    "X2":           pdb["batch"]["all_atom_positions"][:,atom_idx],
                    "mask1":        pdb["batch"]["all_atom_mask"][:,1],
                    "mask2":        pdb["batch"]["all_atom_mask"][:,1],
                    "residue_idx1": pdb["residue_index"],
                    "residue_idx2": pdb["residue_index"],
                    "chain_idx1":   chain_idx,
                    "chain_idx2":   chain_idx,
                    "lengths":      np.array(self._lengths),
                    "bias":         np.zeros((L,20))}
    
    if rm_aa is not None:
      for aa in rm_aa.split(","):
        self._inputs["bias"][...,aa_order[aa]] -= 1e6
        
    if fix_pos is not None:
      p = prep_pos(fix_pos, **pdb["idx"])["pos"]
      if inverse:
        p = np.delete(np.arange(L),p)
      self._inputs["fix_pos"] = np.full(L,False)
      self._inputs["fix_pos"][p] = True
      self._inputs["bias"][p] = 1e7 * np.eye(21)[self._inputs["S"]][p,:20]
  
    if homooligomer:
      assert min(self._lengths) == max(self._lengths)
      self._tied_lengths = True
      self._len = self._lengths[0]
    else:
      self._tied_lengths = False    
      self._len = sum(self._lengths)
      
    self.pdb = pdb
    
    if verbose:
      print("lengths", self._lengths)
      if "fix_pos" in self._inputs:
        print("the following positions will be fixed:")
        print(np.where(self._inputs["fix_pos"])[0])
    
  def get_af_inputs(self, af: protein.Protein) -> None:
    """
        Retrieves input features from an AlphaFold model.
        
        Args:
            af (AlphaFoldModel): An AlphaFold model object.
    """
    self._lengths = af._lengths
    self._len = af._len

    self._inputs["residue_idx1"] = af._inputs["residue_index"]
    self._inputs["residue_idx2"] = af._inputs["residue_index"]
    self._inputs["chain_idx1"]   = af._inputs["asym_id"]
    self._inputs["chain_idx2"]   = af._inputs["asym_id"]
    self._inputs["lengths"]      = np.array(self._lengths)

    # set bias
    L = sum(self._lengths)
    self._inputs["bias"] = np.zeros((L,20))
    self._inputs["bias"][-af._len:] = af._inputs["bias"]
    
    if "offset" in af._inputs:
      self._inputs["offset"] = af._inputs["offset"]

    if "batch" in af._inputs:
      atom_idx = tuple(residue_constants.atom_order[k] for k in ["N","CA","C","O"])
      batch = af._inputs["batch"]
      self._inputs["X1"]    = batch["all_atom_positions"][:,atom_idx]
      self._inputs["X2"]    = batch["all_atom_positions"][:,atom_idx]
      self._inputs["mask1"] = batch["all_atom_mask"][:,1]
      self._inputs["mask2"] = batch["all_atom_mask"][:,1]

    # fix positions
    if af.protocol == "binder":
      fix_pos = np.array([True]*self._target_len + [False] * self._binder_len)
    else:
      fix_pos = af._inputs["fix_pos"]
    
    self._inputs["fix_pos"] = fix_pos
    self._inputs["bias"][fix_pos] = 1e7 * np.eye(21)[self._inputs["S"]][fix_pos,:20]

    # tie positions
    if af._args["copies"] > 1:
      assert min(self._lengths) == max(self._lengths)
      self._tied_lengths = True
    else:
      self._tied_lengths = False
      
  def sample(self, num: int = 1, batch: int = 1, temperature: float = 0.1,
              rescore: bool = False, **kwargs) -> Dict[str, Any]:
    """
        Samples sequences from the MPNN model.
    """
    O = []
    for _ in range(num):
      O.append(self.sample_parallel(batch, temperature, rescore, **kwargs))
    return jax.tree_map(lambda *x:np.concatenate(x,0),*O)

  def sample_parallel(self, batch: int = 10, temperature: float = 0.1,
                        rescore: bool = False, **kwargs) -> Dict[str, Any]:
    """
    Sample new sequences in parallel.
    """
    I = copy_dict(self._inputs)
    I.update(kwargs)
    key = I.pop("key",self.key())
    keys = jax.random.split(key,batch)
    O = self._sample_parallel(keys, I, temperature, self._tied_lengths)
    if rescore:
      O = self._rescore_parallel(keys, I, O["S"], O["decoding_order"])
    O = jax.tree_map(np.array, O)

    # process outputs to human-readable form
    O.update(self._get_seq(O))
    O.update(self._get_score(I,O))
    return O
  
  def _get_seq(self, O: Dict[str, jnp.ndarray]) -> Dict[str, np.ndarray]:
    """
    Convert one-hot encoded sequence to amino acid sequence.
    """
    def split_seq(seq):
      if len(self._lengths) > 1:
        seq = "".join(np.insert(list(seq),np.cumsum(self._lengths[:-1]),"/"))
        if self._tied_lengths:
          seq = seq.split("/")[0]
      return seq
    seqs, S = [], O["S"].argmax(-1)
    if S.ndim == 1: S = [S]
    for s in S:
      seq = "".join([order_aa[a] for a in s])
      seq = split_seq(seq)
      seqs.append(seq)
    
    return {"seq": np.array(seqs)}
  
  def _get_score(self, I: Dict[str, jnp.ndarray], O: Dict[str, jnp.ndarray]) -> Dict[str, np.ndarray]:
    """
    Compute the logits to score and sequence recovery.
    """
    mask = I["mask"].copy()
    if "fix_pos" in I:
      mask[I["fix_pos"]] = 0

    log_q = log_softmax(O["logits"],-1)[...,:20]
    q = softmax(O["logits"][...,:20],-1)
    if "S" in O:
      S = O["S"][...,:20]
      score = -(S * log_q).sum(-1)
      seqid = S.argmax(-1) == self._inputs["S"]
    else:
      score = -(q * log_q).sum(-1)
      seqid = np.zeros_like(score)
      
    score = (score * mask).sum(-1) / (mask.sum() + 1e-8)
    seqid = (seqid * mask).sum(-1) / (mask.sum() + 1e-8)

    return {"score":score, "seqid":seqid}
  
  def score(self, seq: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    """Score a sequence

    Args:
        seq (Optional[str], optional): _description_. Defaults to None.

    Returns:
        Dict[str, Any]: _description_
    """
    I = copy_dict(self._inputs)
    if seq is not None:
      if self._tied_lengths and len(seq) == self._lengths[0]:
        seq = seq * len(self._lengths)
      p = np.arange(I["S"].shape[0])
      if "fix_pos" in I and len(seq) == (I["S"].shape[0] - sum(I["fix_pos"])):
        p = p[I["fix_pos"]]
      I["S"][p] = np.array([aa_order.get(aa, -1) for aa in seq])
      
    I.update(kwargs)
    key = I.pop("key", self.key())
    O = jax.tree_map(np.array, self._score(**I, key=key))
    O.update(self._get_score(I, O))
    return O
  
  def get_logits(self, **kwargs) -> jnp.ndarray:
    """Get logits

    Args:
        **kwargs: _description_

    Returns:
        jnp.ndarray: _description_
    """
    return self.score(**kwargs)["logits"]
  
  def get_unconditional_logits(self, **kwargs) -> jnp.ndarray:
    """Get unconditional logits

    Args:
        **kwargs: _description_

    Returns:
        jnp.ndarray: _description_
    """
    L = self._inputs["X1"].shape[0]
    kwargs["ar_mask"] = np.zeros((L, L))
    return self.score(**kwargs)["logits"]
  
    

  def set_seed(self, seed: Optional[int] = None):
    # Set the random seed for reproducibility
    np.random.seed(seed=seed)
    self.key = Key(seed=seed).get

  def _setup_dual_backbone(self):
    # Define private methods for dual backbone sampling and scoring
    # These methods will be similar to those in mk_mpnn_model but will
    # use the dual backbone functionalities from RunModelDual and DualProteinMPNN

    def _score_dual_backbone(X1, 
                             X2, 
                             mask1, 
                             mask2, 
                             residue_idx1, 
                             residue_idx2, 
                             chain_idx1, 
                             chain_idx2, 
                             key, 
                             **kwargs):
      """
            Scores a pair of sequences against two different protein backbones simultaneously using the Dual Backbone MPNN model.
      
      This method takes two sets of input features corresponding to two different protein backbones and computes the score
      for a sequence against both backbones. The method relies on the DualProteinMPNN model's ability to handle dual backbone
      inputs and produce a combined score.
      """
      I_combined = {
        'X1': X1,
        'X2': X2,
        'mask1': mask1,
        'mask2': mask2,
        'residue_idx1': residue_idx1,
        'residue_idx2': residue_idx2,
        'chain_idx1': chain_idx1,
        'chain_idx2': chain_idx2
      }
      I_combined.update(kwargs)
      # Define decoding order
      if "decoding_order" not in I_combined:
        key, sub_key = jax.random.split(key)
        randn = jax.random.uniform(sub_key, (I_combined["X1"].shape[0],))    
        randn = jnp.where(I_combined["mask1"], randn, randn+1)
        if "fix_pos" in I_combined: 
          randn = jnp.where(I_combined["fix_pos"],randn-1,randn)
        I_combined["decoding_order"] = randn.argsort()
      
      for k in ["S","bias"]:
        if k in I_combined: I_combined[k] = _aa_convert(I_combined[k])
      
      O = self._model.score(self._model_params, key, I_combined)
      O["S"] = _aa_convert(O["S"], rev=True)
      O["logits"] = _aa_convert(O["logits"], rev=True)
            
      return O


    def _sample_dual_backbone(X1, 
                              X2, 
                              mask1, 
                              mask2, 
                              residue_idx1, 
                              residue_idx2, 
                              chain_idx1, 
                              chain_idx2, 
                              key,
                              temperature=0.1, 
                              tied_lengths=False, 
                              **kwargs):
      """
            Samples a pair of sequences against two different protein backbones simultaneously using the Dual Backbone MPNN model.
      
      This method takes two sets of input features corresponding to two different protein backbones and samples a sequence against
      both backbones. The method relies on the DualProteinMPNN model's ability to handle dual backbone inputs and produce a combined
      sample.
      """
      I_combined = {
        'X1': X1,
        'X2': X2,
        'mask1': mask1,
        'mask2': mask2,
        'residue_idx1': residue_idx1,
        'residue_idx2': residue_idx2,
        'chain_idx1': chain_idx1,
        'chain_idx2': chain_idx2,
        'temperature': temperature
      }
      I_combined.update(kwargs)
      # Define decoding order
      if "decoding_order" in I_combined:
        if I_combined["decoding_order"].ndim == 1:
          I_combined["decoding_order"] = I_combined["decoding_order"][:,None]
      else:
        key, sub_key = jax.random.split(key)
        randn = jax.random.uniform(sub_key, (I_combined["X1"].shape[0],))    
        randn = jnp.where(I_combined["mask1"], randn, randn+1)
        if "fix_pos" in I_combined:
          randn = jnp.where(I_combined["fix_pos"],randn-1,randn)
        if tied_lengths:
          copies = I_combined["lengths"].shape[0]
          decoding_order_tied = randn.reshape(copies,-1).mean(0).argsort()
          I_combined["decoding_order"] = jnp.arange(I_combined["X1"].shape[0]).reshape(copies,-1).T[decoding_order_tied]
        else:
          I_combined["decoding_order"] = randn.argsort()[:,None]

      for k in ["S","bias"]:
        if k in I_combined: I_combined[k] = _aa_convert(I_combined[k])
      
      O = self._model.sample(self._model_params, key, I_combined)
      O["S"] = _aa_convert(O["S"], rev=True)
      O["logits"] = _aa_convert(O["logits"], rev=True)
      return O
    
    self._score_dual_backbone = jax.jit(_score_dual_backbone)
    self._sample_dual_backbone = jax.jit(_sample_dual_backbone, static_argnames=["tied_lengths"])
    
    def _sample_parallel_dual_backbone(key, inputs, temperature, tied_lengths=False):
      """
            Helper function to sample new sequences in parallel using the Dual Backbone MPNN model.
            
            Args:
                key: A key for random number generation.
                inputs: A dictionary containing the input features.
                temperature: Sampling temperature.
                tied_lengths: Whether the lengths of sequences are tied (for homooligomers).
                
            Returns:
                A dictionary containing the sampled sequences and logits.
      """
      inputs.pop("temperature",None)
      inputs.pop("key",None)
      return _sample_dual_backbone(**inputs, key=key, temperature=temperature, tied_lengths=tied_lengths)
    # Vectorize the _sample_parallel function to enable parallel execution.
    fn = jax.vmap(_sample_parallel_dual_backbone, in_axes=[0, None, None, None])
    # Compile the vectorized _sample_parallel function using JAX's JIT for faster execution.
    self._sample_parallel_dual_backbone = jax.jit(fn, static_argnames=["tied_lengths"])
    
      
    def _rescore_parallel_dual_backbone(key, inputs, S, decoding_order):
      """
            Helper function to rescore sequences in parallel using the Dual Backbone MPNN model.
            
            Args:
                key: A key for random number generation.
                inputs: A dictionary containing the input features.
                S: One-hot encoded sequences tensor.
                decoding_order: Tensor representing the order in which positions were decoded.
                
            Returns:
                A dictionary containing the rescored logits and sequences.
      """
      inputs.pop("S",None)
      inputs.pop("decoding_order",None)
      inputs.pop("key",None)
      return _score_dual_backbone(**inputs, key=key, S=S, decoding_order=decoding_order)
    # Vectorize the _rescore_parallel function to enable parallel execution.
    fn = jax.vmap(_rescore_parallel_dual_backbone, in_axes=[0, None, 0, 0])
    # Compile the vectorized _rescore_parallel function using JAX's JIT for faster execution.
    self._rescore_parallel_dual_backbone = jax.jit(fn)
      
                              
                              
                              


#######################################################################################

def _aa_convert(x, rev=False):
  mpnn_alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
  af_alphabet =   'ARNDCQEGHILKMFPSTWYVX'
  if x is None:
    return x
  else:
    if rev:
      return x[...,tuple(mpnn_alphabet.index(k) for k in af_alphabet)]
    else:
      x = jax.nn.one_hot(x,21) if jnp.issubdtype(x.dtype, jnp.integer) else x
      if x.shape[-1] == 20:
        x = jnp.pad(x,[[0,0],[0,1]])
      return x[...,tuple(af_alphabet.index(k) for k in mpnn_alphabet)]