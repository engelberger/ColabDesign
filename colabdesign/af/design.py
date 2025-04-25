import os
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

# Assuming these imports are from a specific library structure (e.g., ColabDesign)
# If the structure is different, adjust the import paths accordingly.
try:
    from colabdesign.af.alphafold.common import residue_constants
    from colabdesign.af.utils import dgram_from_positions
    from colabdesign.shared.utils import (Key, categorical, copy_dict,
                                          copy_missing, dict_to_str, softmax,
                                          to_float, to_list, update_dict)
except ImportError:
    print("Warning: ColabDesign specific imports failed. "
          "Functionality might be limited.")
    # Define dummy types or functions if needed for type hinting
    ResidueConstants = Any
    Key = Any
    # Add dummy functions if necessary, e.g.:
    def dgram_from_positions(*args, **kwargs): return np.zeros(1)
    def copy_dict(d): return d.copy()
    def update_dict(d1, d2): d1.update(d2); return d1
    def copy_missing(target, source):
        for k, v in source.items(): target.setdefault(k, v)
    def dict_to_str(d, **kwargs): return str(d)
    def to_float(x): return x # Simplified placeholder
    def softmax(x): return x # Simplified placeholder
    def categorical(x): return np.zeros(1) # Simplified placeholder
    def to_list(x): return list(x) if isinstance(x, (list, tuple)) else [x]

# Type Aliases for clarity
JaxArray = Union[jnp.ndarray, np.ndarray]
Params = Dict[str, JaxArray]
AuxOutput = Dict[str, Any]
ModelState = Any # Type for optimizer state
ModelParams = Dict[str, Any] # Type for model parameters (p in original code)
LogData = Dict[str, Union[float, int, List[int]]]
TrajectoryData = Dict[str, List[np.ndarray]]


class _AFDesign:
    """
    Manages the AlphaFold protein design process, including optimization and prediction.

    This class orchestrates the interaction with the underlying AlphaFold model,
    handles sequence optimization through various strategies (gradient descent,
    MCMC, semi-greedy), manages model parameters and hyperparameters, and tracks
    the design trajectory.

    Attributes:
        _args (Dict[str, Any]): Configuration arguments for the design process.
        _opt (Dict[str, Any]): Default optimization options.
        _inputs (Dict[str, Any]): Current inputs to the AlphaFold model.
        _params (Params): Trainable parameters (e.g., sequence logits).
        _model (Dict[str, Callable]): Compiled JAX functions for model execution.
        _model_params (List[ModelParams]): List of loaded model parameters.
        _model_names (List[str]): Names corresponding to _model_params.
        _optimizer (Callable): The optimizer function (e.g., from optax).
        _state (ModelState): The current state of the optimizer.
        _k (int): Current optimization iteration step.
        _tmp (Dict[str, Any]): Temporary storage for logs, best results, etc.
        _callbacks (Dict[str, Dict[str, List[Callable]]]): Callbacks for different stages.
        key (Key): JAX random key generator state.
        opt (Dict[str, Any]): Current optimization options (mutable).
        aux (AuxOutput): Auxiliary output from the last model run.
        protocol (str): Design protocol type (e.g., "binder").
        _len (int): Length of the sequence being designed/analyzed.
        _target_len (int): Length of the target part (for binder protocol).
        _lengths (List[int]): Lengths of chains in a multimer setting.
    """

    def __init__(self, args: Dict[str, Any], model_runner: Any):
        """
        Initializes the _AFDesign instance.

        Args:
            args: Configuration arguments.
            model_runner: An object responsible for providing the model functions
                          and parameters (replace with actual dependency).
        """
        self._args = args
        self._opt = model_runner.get_default_optimizer_options() # Placeholder
        self._inputs: Dict[str, Any] = {}
        self._params: Params = {}
        self._model = model_runner.get_model_functions() # Placeholder
        self._model_params = model_runner.get_model_params() # Placeholder
        self._model_names = model_runner.get_model_names() # Placeholder
        self._optimizer = model_runner.get_optimizer() # Placeholder
        self._state: ModelState = None
        self._k: int = 0
        self._tmp: Dict[str, Any] = {"traj": {}, "log": [], "best": {}}
        self._callbacks = {"design": {"pre": [], "post": []}} # Example structure
        self.key = Key(0) # Initialize JAX key sequence
        self.opt: Dict[str, Any] = {}
        self.aux: AuxOutput = {}
        self.protocol: str = args.get("protocol", "monomer")
        self._len: int = 0 # Should be set during setup
        self._target_len: int = 0 # Should be set for binder protocol
        self._lengths: List[int] = [] # Should be set for multimer

        # Placeholder attributes - these should be properly initialized
        # based on the actual `model_runner` or setup process.
        if "use_ptm" not in self._args: self._args["use_ptm"] = False
        if "optimize_seq" not in self._args: self._args["optimize_seq"] = True
        if "traj_iter" not in self._args: self._args["traj_iter"] = 1
        if "traj_max" not in self._args: self._args["traj_max"] = 10
        if "best_metric" not in self._args: self._args["best_metric"] = "loss"
        if "use_bfloat16" not in self._args: self._args["use_bfloat16"] = False
        if "clear_prev" not in self._args: self._args["clear_prev"] = False
        if "use_batch_as_template" not in self._args: self._args["use_batch_as_template"] = False
        if "use_initial_guess" not in self._args: self._args["use_initial_guess"] = False
        if "use_dgram" not in self._args: self._args["use_dgram"] = False
        if "use_dgram_pred" not in self._args: self._args["use_dgram_pred"] = False
        if "use_initial_atom_pos" not in self._args: self._args["use_initial_atom_pos"] = False
        if "alphabet_size" not in self._args: self._args["alphabet_size"] = 20 # Default AA size


    # --------------------------------------------------------------------------
    # Initialization and Setup Methods
    # --------------------------------------------------------------------------
    def set_seed(self, seed: Optional[int] = None):
        """Sets the random seed for JAX and NumPy."""
        if seed is not None:
            self.key = Key(seed=seed)
            np.random.seed(seed)
            random.seed(seed)

    def set_optimizer(self, optimizer: Optional[Callable] = None, **kwargs):
        """Sets or resets the optimizer and its state."""
        if optimizer is not None:
            self._optimizer = optimizer
        # Initialize optimizer state (example, replace with actual init)
        # self._state = self._optimizer.init(self._params)
        pass # Placeholder for actual optimizer state initialization

    def set_seq(self, seq: Optional[Union[str, np.ndarray]] = None,
                mode: Optional[str] = None, **kwargs):
        """Sets the initial sequence for optimization."""
        # Placeholder: Implement sequence initialization logic
        # This would involve setting self._params['seq'] based on input seq
        # and mode (e.g., 'logits', 'gumbel').
        # It should also handle fixed positions based on self._inputs['fix_pos'].
        print(f"Setting sequence (mode={mode}): {seq} (kwargs: {kwargs})")
        # Example: self._len = len(seq)
        pass

    def set_msa(self, msa: np.ndarray):
        """Sets the MSA (Multiple Sequence Alignment) when sequence is not optimized."""
        # Placeholder: Implement MSA setting logic
        print("Setting MSA.")
        # Example: self._inputs['msa'] = msa
        # Example: self._len = msa.shape[1]
        pass

    def set_opt(self, options: Optional[Dict[str, Any]] = None, **kwargs):
        """Sets optimization options."""
        if options is None: options = {}
        options.update(kwargs)
        update_dict(self.opt, options)
        # Also update the persistent _inputs['opt']
        if "opt" in self._inputs:
            update_dict(self._inputs["opt"], options)
        else:
            self._inputs["opt"] = copy_dict(options)

    def set_weights(self, weights: Optional[Dict[str, float]] = None, **kwargs):
        """Sets weights for different loss components."""
        if weights is None: weights = {}
        weights.update(kwargs)
        # Assuming weights are stored within self.opt or a dedicated dict
        if "weights" not in self.opt: self.opt["weights"] = {}
        update_dict(self.opt["weights"], weights)
        # Also update the persistent _inputs['opt']['weights']
        if "opt" in self._inputs and "weights" in self._inputs["opt"]:
             update_dict(self._inputs["opt"]["weights"], weights)
        else:
             self._inputs["opt"]["weights"] = copy_dict(weights)


    def restart(self, seed: Optional[int] = None,
                seq: Optional[Union[str, np.ndarray]] = None,
                mode: Optional[str] = None,
                keep_history: bool = False,
                reset_opt: bool = True, **kwargs):
        """
        Restarts the optimization process.

        Args:
            seed: Optional random seed for reproducibility.
            seq: Optional initial sequence.
            mode: Optional mode for sequence initialization.
            keep_history: If True, retains optimization trajectory and logs.
            reset_opt: If True (default), resets options and weights to defaults.
                       Set to False to keep current settings.
            **kwargs: Additional arguments passed to `set_seq`.
        """
        # Reset options and weights to defaults if requested
        if reset_opt and not keep_history:
            # Restore default options from _opt
            if hasattr(self, "_opt"):
                self.opt = copy_dict(self._opt)
                if "opt" in self._inputs:
                    self._inputs["opt"] = copy_dict(self._opt)
                # Ensure all default keys are present if self.opt already existed
                if hasattr(self, "opt"):
                     copy_missing(self.opt, self._opt)
                if "opt" in self._inputs:
                     copy_missing(self._inputs["opt"], self._opt)

            # Clear previous auxiliary outputs if they exist
            if hasattr(self, "aux"):
                del self.aux

        # Initialize trajectory and logs unless keeping history
        if not keep_history:
            self._tmp = {
                "traj": {"seq": [], "xyz": [], "plddt": []},
                "log": [],
                "best": {}
            }
            if self._args.get("use_ptm", False):
                self._tmp["traj"]["pae"] = []

        # Initialize sequence
        self.set_seed(seed)
        if self._args.get("optimize_seq", True):
            # Pass sequence initialization arguments
            self.set_seq(seq=seq, mode=mode, **kwargs)
        else:
            # Use the wild-type sequence if not optimizing
            if "wt_aatype" in self._inputs:
                 self.set_msa(msa=self._inputs["wt_aatype"])
            else:
                 print("Warning: 'wt_aatype' not found in inputs for non-optimizing restart.")


        # Reset optimizer state and iteration count
        self._k = 0
        self.set_optimizer() # Re-initialize optimizer state

    # --------------------------------------------------------------------------
    # Core Model Execution Logic
    # --------------------------------------------------------------------------
    def _get_model_nums(self, num_models: Optional[int] = None,
                        sample_models: Optional[bool] = None,
                        models: Optional[Union[List[Union[int, str]], int, str]] = None
                        ) -> List[int]:
        """
        Determines which model parameters (by index) to use for a run.

        Args:
            num_models: Max number of models to use. Defaults to opt['num_models'].
            sample_models: Whether to sample models randomly. Defaults to opt['sample_models'].
            models: Specific model indices or names to use.

        Returns:
            A list of integer indices corresponding to the models in self._model_params.
        """
        opt = self._inputs.get("opt", {})
        if num_models is None:
            num_models = opt.get("num_models", 1)
        if sample_models is None:
            sample_models = opt.get("sample_models", False)

        available_model_names = self._model_names
        available_model_indices = list(range(len(available_model_names)))

        selected_indices: List[int]
        if models is not None:
            model_list = to_list(models)
            selected_indices = []
            for m in model_list:
                if isinstance(m, int):
                    if 0 <= m < len(available_model_indices):
                        selected_indices.append(available_model_indices[m])
                    else:
                        raise ValueError(f"Model index {m} out of range.")
                elif isinstance(m, str):
                    try:
                        idx = available_model_names.index(m)
                        selected_indices.append(idx)
                    except ValueError:
                        raise ValueError(f"Model name '{m}' not found.")
                else:
                    raise TypeError(f"Invalid model identifier type: {type(m)}")
        else:
            selected_indices = available_model_indices

        # Limit the number of models
        num_to_use = min(num_models, len(selected_indices))

        # Sample or take the first `num_to_use`
        final_model_indices: List[int]
        if sample_models and num_to_use < len(selected_indices):
            final_model_indices = list(np.random.choice(selected_indices, size=num_to_use, replace=False))
        else:
            final_model_indices = selected_indices[:num_to_use]

        return final_model_indices

    def _single(self, model_params: ModelParams, backprop: bool = True) -> AuxOutput:
        """
        Performs a single forward pass through the AlphaFold model.

        Args:
            model_params: The specific parameters for the model being run.
            backprop: If True, computes gradients using the 'grad_fn'. Otherwise,
                      uses the forward 'fn' and returns zero gradients.

        Returns:
            An AuxOutput dictionary containing model outputs, loss, and gradients.
        """
        # Prepare inputs for the JAX function
        # Ensure self.key() advances the key state
        inputs = [self._params, model_params, self._inputs, self.key()]

        if backprop:
            # Compute loss, auxiliary outputs, and gradients
            (loss, aux), grad = self._model["grad_fn"](*inputs)
        else:
            # Compute only loss and auxiliary outputs
            loss, aux = self._model["fn"](*inputs)
            # Create zero gradients matching the structure of trainable parameters
            grad = jax.tree_map(np.zeros_like, self._params)

        # Combine results into a single dictionary
        aux.update({"loss": loss, "grad": grad})
        return aux

    def _initialize_recycle_inputs(self) -> Dict[str, np.ndarray]:
        """Initializes the 'prev' dictionary for recycling."""
        length = self._inputs["residue_index"].shape[0]
        dtype = jnp.bfloat16 if self._args.get("use_bfloat16", False) else jnp.float32

        prev = {
            'prev_msa_first_row': np.zeros([length, 256], dtype=dtype),
            'prev_pair': np.zeros([length, length, 128], dtype=dtype)
        }

        # Determine initial coordinates/distogram source
        if self._args.get("use_batch_as_template", False):
            batch = self._inputs.get("batch", {})
        else:
            batch = {}
            for key in ["aatype", "all_atom_positions", "all_atom_mask", "dgram"]:
                template_key = f"template_{key}"
                if template_key in self._inputs:
                    # Assuming template shape is [num_templates, length, ...]
                    batch[key] = self._inputs[template_key][0] # Use first template

        initial_positions = batch.get("all_atom_positions", np.zeros([length, residue_constants.atom_type_num, 3]))
        initial_aatype = batch.get("aatype", np.zeros(length, dtype=int)) # Assuming aatype needed for dgram

        use_dgram_feature = self._args.get("use_dgram", False) or self._args.get("use_dgram_pred", False)

        if self._args.get("use_initial_guess", False):
            if use_dgram_feature:
                 if "dgram" in batch:
                     # Process dgram if provided (example conversion)
                     dgram_bins = batch["dgram"]
                     # Assuming dgram_bins format needs adjustment for prev_dgram
                     # Example: Concatenate first 14 bins and sum of remaining bins
                     prev["prev_dgram"] = np.concatenate(
                         [dgram_bins[..., :14], dgram_bins[..., 14:].sum(-1, keepdims=True)],
                         axis=-1
                     )
                 else:
                     # Compute distogram from positions if dgram not in batch
                     prev["prev_dgram"] = dgram_from_positions(
                         positions=initial_positions,
                         sequence=initial_aatype, # Pass sequence if needed by the function
                         num_bins=15, min_bin=3.25, max_bin=20.75 # Example parameters
                     )
            else: # Use positions directly
                 prev["prev_pos"] = initial_positions
        else: # No initial guess
            if use_dgram_feature:
                 prev["prev_dgram"] = np.zeros([length, length, 15], dtype=dtype) # 15 bins
            else:
                 prev["prev_pos"] = np.zeros([length, residue_constants.atom_type_num, 3], dtype=dtype)


        # Handle initial atom positions input for structure module
        if self._args.get("use_initial_atom_pos", False):
             self._inputs["initial_atom_pos"] = initial_positions

        return prev


    def _recycle(self, model_params: ModelParams,
                 num_recycles: Optional[int] = None,
                 backprop: bool = True) -> AuxOutput:
        """
        Performs multiple passes (recycles) through the model.

        Handles different recycling strategies: 'backprop', 'add_prev', 'sample',
        'average', 'last', 'first'.

        Args:
            model_params: The parameters for the model being run.
            num_recycles: Number of recycling iterations. Defaults to opt['num_recycles'].
            backprop: If True, enables gradient computation during the relevant passes.

        Returns:
            An AuxOutput dictionary containing averaged/selected outputs, loss,
            gradients, and the number of recycles performed.
        """
        recycle_mode = self._args.get("recycle_mode", "add_prev") # Default mode
        if num_recycles is None:
            num_recycles = self._inputs.get("opt", {}).get("num_recycles", 0)

        # Modes where recycling is handled internally by the compiled model function
        if recycle_mode in ["backprop", "add_prev"]:
            # Only a single call to _single is needed
            aux = self._single(model_params, backprop=backprop)

        # Modes requiring explicit iteration in Python
        else:
            # Initialize 'prev' inputs if needed
            if "prev" not in self._inputs or self._args.get("clear_prev", False):
                self._inputs["prev"] = self._initialize_recycle_inputs()

            total_cycles = num_recycles + 1
            # Determine which cycles contribute to the gradient based on the mode
            gradient_mask = [0.0] * total_cycles
            if recycle_mode == "sample":
                # Randomly select one cycle for backprop
                gradient_mask[np.random.randint(0, total_cycles)] = 1.0
            elif recycle_mode == "average":
                # Average gradients over all cycles
                gradient_mask = [1.0 / total_cycles] * total_cycles
            elif recycle_mode == "last":
                # Use gradient from the last cycle only
                gradient_mask[-1] = 1.0
            elif recycle_mode == "first":
                 # Use gradient from the first cycle only
                 gradient_mask[0] = 1.0
            else:
                 print(f"Warning: Unknown recycle mode '{recycle_mode}'. Defaulting to 'last'.")
                 gradient_mask[-1] = 1.0


            collected_grads = []
            final_aux = {} # To store the output from the last cycle

            for cycle_index, grad_weight in enumerate(gradient_mask):
                # Determine if backprop is needed for this specific cycle
                do_backprop_this_cycle = backprop and (grad_weight > 0)

                # Run a single pass
                current_aux = self._single(model_params, backprop=do_backprop_this_cycle)

                if do_backprop_this_cycle:
                    # Scale gradients by the weight for this cycle and store
                    weighted_grad = jax.tree_map(lambda g: g * grad_weight, current_aux["grad"])
                    collected_grads.append(weighted_grad)

                # Update 'prev' inputs for the next cycle
                self._inputs["prev"] = current_aux["prev"]
                # Optionally update initial atom positions for structure module
                if self._args.get("use_initial_atom_pos", False):
                    self._inputs["initial_atom_pos"] = current_aux["atom_positions"]

                # Store the aux output of the last cycle
                if cycle_index == total_cycles - 1:
                    final_aux = current_aux

            # Combine gradients from selected cycles
            if collected_grads:
                 # Sum the weighted gradients across cycles
                 final_aux["grad"] = jax.tree_map(lambda *x: jnp.stack(x).sum(0), *collected_grads)
            else:
                 # If no backprop occurred (e.g., backprop=False), ensure zero grads exist
                 final_aux["grad"] = jax.tree_map(np.zeros_like, self._params)

            aux = final_aux # Use the outputs from the final cycle

        # Store the number of recycles performed in the output
        aux["num_recycles"] = num_recycles
        return aux

    def _update_aux_and_log(self, auxs_list: List[AuxOutput], model_nums: List[int]):
        """Averages outputs across models and updates the log."""

        # Stack outputs from different models for averaging
        # Use jax.tree_map for robust handling of nested structures
        stacked_auxs = jax.tree_map(lambda *x: np.stack(x), *auxs_list)

        # Define averaging function: mean for floats, first element for ints/others
        def avg_or_first(arr):
            if np.issubdtype(arr.dtype, np.number) and not np.issubdtype(arr.dtype, np.integer):
                return arr.mean(axis=0)
            else:
                return arr[0] # Keep first model's output for non-float arrays

        # Average the stacked outputs
        self.aux = jax.tree_map(avg_or_first, stacked_auxs)

        # Special handling: Keep specific outputs from the first model if needed
        # Example: Use atom positions from the first model directly
        self.aux["atom_positions"] = stacked_auxs["atom_positions"][0]

        # Store all individual model outputs
        self.aux["all"] = stacked_auxs

        # --- Update Log ---
        log_data: LogData = {**self.aux["losses"]} # Start with loss components

        # Remap pLDDT loss to pLDDT score (assuming loss = 1 - plddt)
        if "plddt" in log_data:
             log_data["plddt"] = 1.0 - log_data["plddt"]

        # Add PTM scores if available
        if self._args.get("use_ptm", False):
            for key in ["loss", "i_ptm", "ptm"]: # Add interface PTM (i_ptm) if calculated
                if key in self.aux:
                    log_data[key] = self.aux[key]
        else:
            log_data["loss"] = self.aux["loss"] # Overall loss

        # Add sequence-related metrics if optimizing sequence
        if self._args.get("optimize_seq", True):
            # Log sequence sampling parameters
            for key in ["hard", "soft", "temp"]:
                log_data[key] = self._inputs.get("opt", {}).get(key, np.nan)

            # Compute sequence recovery (if applicable)
            seq_len = self._len # Use the stored sequence length
            true_aatype = self._inputs.get("wt_aatype", [])[:seq_len]
            fix_pos_mask = self._inputs.get("fix_pos", [])[:seq_len]

            if len(true_aatype) > 0: # Ensure wild type is available
                # Create mask for positions that are not fixed and have a valid wild type
                valid_mask = np.logical_and(~fix_pos_mask, true_aatype != -1) # Assuming -1 indicates unknown/gap

                if valid_mask.sum() > 0:
                    # Get predicted sequence (argmax over alphabet dim)
                    # Assuming aux["seq"] shape is [num_models, length, alphabet_size]
                    # Use the averaged seq probability or the first model's prediction
                    if "seq" in self.aux and self.aux["seq"].ndim == 2: # Averaged prob [L, A]
                         pred_aatype = self.aux["seq"].argmax(-1)
                    elif "seq" in stacked_auxs and stacked_auxs["seq"].ndim == 3: # Use first model [N, L, A]
                         pred_aatype = stacked_auxs["seq"][0,:seq_len].argmax(-1)
                    else:
                         pred_aatype = [] # Cannot calculate seqid

                    if len(pred_aatype) == seq_len:
                         correct_preds = (true_aatype == pred_aatype)
                         seqid = (correct_preds * valid_mask).sum() / valid_mask.sum()
                         log_data["seqid"] = float(seqid)

        # Convert log values to standard Python floats for easier handling/serialization
        self.aux["log"] = to_float(log_data)

        # Add metadata about the run
        self.aux["log"].update({
            "recycles": int(self.aux.get("num_recycles", np.nan)),
            "models": model_nums
        })


    def run(self, num_recycles: Optional[int] = None,
            num_models: Optional[int] = None,
            sample_models: Optional[bool] = None,
            models: Optional[Union[List[Union[int, str]], int, str]] = None,
            backprop: bool = True,
            callback: Optional[Callable[['_AFDesign'], None]] = None,
            model_nums: Optional[List[int]] = None,
            return_aux: bool = False) -> Optional[AuxOutput]:
        """
        Runs the AlphaFold model for selected model parameters and recycles.

        Args:
            num_recycles: Number of recycles. Defaults to opt['num_recycles'].
            num_models: Max number of models to use. Defaults to opt['num_models'].
            sample_models: Sample models randomly. Defaults to opt['sample_models'].
            models: Specific model indices or names to use.
            backprop: Enable gradient computation.
            callback: Optional function to call after the run.
            model_nums: Pre-selected list of model indices to run. Overrides other model selection args.
            return_aux: If True, returns the auxiliary output dictionary.

        Returns:
            The auxiliary output dictionary if return_aux is True, otherwise None.
        """
        # Pre-design callbacks
        for fn in self._callbacks["design"]["pre"]:
            fn(self)

        # Determine model indices to use
        if model_nums is None:
            model_nums = self._get_model_nums(num_models, sample_models, models)
        if not model_nums:
            raise ValueError("No model parameters selected or available.")

        # Run model for each selected parameter set
        auxs_list = []
        for model_index in model_nums:
            model_params = self._model_params[model_index]
            aux_output = self._recycle(model_params,
                                       num_recycles=num_recycles,
                                       backprop=backprop)
            auxs_list.append(aux_output)

        # Average results across models and update logs
        self._update_aux_and_log(auxs_list, model_nums)

        # Post-design callbacks
        for fn in (self._callbacks["design"]["post"] + to_list(callback)):
             if fn: fn(self) # Ensure callback is not None

        if return_aux:
            return self.aux
        return None


    # --------------------------------------------------------------------------
    # Optimization Step Logic
    # --------------------------------------------------------------------------
    def _norm_seq_grad(self):
         """Normalizes the sequence gradient."""
         # Check if sequence optimization is active and gradient exists
         if self._args.get("optimize_seq", True) and "seq" in self.aux.get("grad", {}):
             grad_seq = self.aux["grad"]["seq"]
             # Normalize gradient: Subtract mean across the alphabet dimension
             norm_grad = grad_seq - grad_seq.mean(-1, keepdims=True)
             self.aux["grad"]["seq"] = norm_grad
         else:
             print("Warning: Sequence gradient normalization skipped (not optimizing seq or grad missing).")


    def step(self, lr_scale: float = 1.0,
             num_recycles: Optional[int] = None,
             num_models: Optional[int] = None,
             sample_models: Optional[bool] = None,
             models: Optional[Union[List[Union[int, str]], int, str]] = None,
             backprop: bool = True,
             callback: Optional[Callable[['_AFDesign'], None]] = None,
             save_best: bool = False,
             save_results: bool = True,
             verbose: int = 1):
        """
        Performs one optimization step (forward pass, backward pass, gradient update).

        Args:
            lr_scale: Scaling factor for the learning rate.
            num_recycles, num_models, sample_models, models: Passthrough to `run`.
            backprop: Enable gradient computation.
            callback: Optional function to call after the run.
            save_best: If True, saves the state if it's the best seen so far.
            save_results: If True, logs results and updates trajectory.
            verbose: Print log frequency (0 for silent, 1 for every step, N for every N steps).
        """
        # Run forward and backward pass to get gradients
        self.run(num_recycles=num_recycles, num_models=num_models,
                 sample_models=sample_models, models=models,
                 backprop=backprop, callback=callback)

        # Optional gradient modifications (e.g., normalization)
        if self._args.get("optimize_seq", True) and self._inputs.get("opt", {}).get("norm_seq_grad", False):
            self._norm_seq_grad()

        # Compute parameter updates using the optimizer
        # Assumes optimizer returns (new_state, updates)
        # Replace with actual optimizer call structure
        # Example: updates, self._state = self._optimizer.update(self.aux["grad"], self._state, self._params)
        updates = self.aux["grad"] # Placeholder: Using raw gradients as updates
        # self._state = new_optimizer_state # Update optimizer state

        # Apply parameter updates
        learning_rate = self._inputs.get("opt", {}).get("learning_rate", 1e-3) * lr_scale
        # Use jax.tree_map for applying updates to the parameter structure
        self._params = jax.tree_map(lambda p, u: p - learning_rate * u,
                                    self._params, updates)

        # Save results and potentially update the best state found
        if save_results:
            self._save_results(save_best=save_best, verbose=verbose)

        # Increment iteration counter
        self._k += 1

    # --------------------------------------------------------------------------
    # Logging and Saving Results
    # --------------------------------------------------------------------------
    def _print_log(self, print_prefix: str = "", aux_data: Optional[AuxOutput] = None):
        """Prints a formatted log line of the current state."""
        if aux_data is None:
            aux_data = self.aux
        if "log" not in aux_data:
            print(f"{print_prefix} No log data found.")
            return

        log_to_print = aux_data["log"]
        weights = self._inputs.get("opt", {}).get("weights", {})

        # Define standard keys to print
        base_keys = ["models", "recycles", "hard", "soft", "temp", "seqid", "loss",
                     "seq_ent", "mlm", "helix", "exp_res", "con", "i_con",
                     "sc_fape", "sc_chi", "sc_chi_norm", "dgram_cce", "fape", "plddt"]
        ptm_keys = ["pae", "i_pae", "ptm", "i_ptm"]
        extra_keys = ["rmsd", "composite"] # Metrics that might be added externally

        keys_to_print = list(base_keys)
        if self._args.get("use_ptm", False):
            keys_to_print.extend(ptm_keys)
            # Special handling for i_ptm based on multimer status
            if "i_ptm" in log_to_print and len(getattr(self, '_lengths', [])) <= 1:
                 # Remove i_ptm for monomers even if calculated
                 if "i_ptm" in keys_to_print: keys_to_print.remove("i_ptm")
                 log_to_print.pop("i_ptm", None) # Remove from dict being printed

        # Add extra keys if they exist in the log
        for k in extra_keys:
            if k in log_to_print:
                keys_to_print.append(k)

        # Keys considered "good" (higher is better or specific interest)
        ok_keys = ["plddt", "ptm", "i_ptm", "seqid", "rmsd", "composite"]

        print(dict_to_str(log_to_print,
                          filt=weights, # Filter/highlight based on weights
                          print_str=print_prefix,
                          keys=keys_to_print,
                          ok=ok_keys))

    def _update_trajectory(self, aux_data: AuxOutput):
         """Adds current state to the trajectory buffer."""
         if (self._k % self._args.get("traj_iter", 1)) == 0:
             # Subselect sequence if needed (e.g., binder protocol)
             full_seq = aux_data.get("seq", None) # Shape [N, L, A] or [L, A]
             if full_seq is not None:
                 if self.protocol == "binder" and hasattr(self, "_target_len"):
                     # Assuming seq shape is [..., Length, Alphabet]
                     seq_to_save = full_seq[..., self._target_len:, :]
                 else:
                     seq_to_save = full_seq
             else:
                 seq_to_save = None # Or handle error/default

             # Extract data for trajectory
             traj_item = {
                 "xyz": aux_data.get("atom_positions", [])[:, 1, :], # CA atoms
                 "plddt": aux_data.get("plddt", []),
                 "seq": seq_to_save
             }
             if self._args.get("use_ptm", False):
                 traj_item["pae"] = aux_data.get("pae", [])

             # Append to trajectory, maintaining max length
             traj_buffer = self._tmp["traj"]
             max_len = self._args.get("traj_max", 10)
             for key, value in traj_item.items():
                 if key not in traj_buffer: traj_buffer[key] = []
                 if len(traj_buffer[key]) >= max_len:
                     traj_buffer[key].pop(0) # Remove oldest entry
                 traj_buffer[key].append(value)


    def _save_results(self, aux_data: Optional[AuxOutput] = None,
                      save_best: bool = False,
                      best_metric: Optional[str] = None,
                      metric_higher_is_better: Optional[bool] = None,
                      verbose: bool = True):
        """Logs results, updates trajectory, and saves the best state if applicable."""
        if aux_data is None:
            aux_data = self.aux
        if "log" not in aux_data: return # Nothing to save

        # Append current log to history
        self._tmp["log"].append(aux_data["log"])

        # Update trajectory buffer
        self._update_trajectory(aux_data)

        # Save best state logic
        if save_best:
            if best_metric is None:
                best_metric = self._args.get("best_metric", "loss")

            if best_metric not in aux_data["log"]:
                print(f"Warning: Best metric '{best_metric}' not found in log. Skipping save_best.")
            else:
                current_metric_value = float(aux_data["log"][best_metric])

                # Determine if higher value is better for this metric
                if metric_higher_is_better is None:
                    # Default behavior based on common metrics
                    metric_higher_is_better = best_metric in ["plddt", "ptm", "i_ptm", "seqid", "composite"]

                # Convert to a minimization problem (lower is better)
                comparison_value = -current_metric_value if metric_higher_is_better else current_metric_value

                # Check if this is the new best
                current_best_value = self._tmp["best"].get("metric", np.inf)
                if comparison_value < current_best_value:
                    print(f"Updating best state (metric: {best_metric} = {current_metric_value:.4f})")
                    self._tmp["best"]["metric"] = comparison_value
                    # Deep copy the relevant parts of aux data for the best state
                    self._tmp["best"]["aux"] = copy_dict(aux_data)
                    # Optionally save parameters, optimizer state, etc.
                    # self._tmp["best"]["params"] = copy_dict(self._params)
                    # self._tmp["best"]["state"] = copy_dict(self._state)


        # Print log line if verbose condition met
        if verbose and ((self._k + 1) % verbose == 0):
            self._print_log(f"Step {self._k+1}", aux_data=aux_data)

    # --------------------------------------------------------------------------
    # Prediction Functionality
    # --------------------------------------------------------------------------
    def predict(self, seq: Optional[Union[str, np.ndarray]] = None,
                bias: Optional[np.ndarray] = None,
                num_models: Optional[int] = None,
                num_recycles: Optional[int] = None,
                models: Optional[Union[List[Union[int, str]], int, str]] = None,
                sample_models: bool = False,
                dropout: bool = False, # Note: Typically False for prediction
                hard: bool = True,     # Use argmax for sequence
                soft: bool = False,    # Use softmax probability
                temp: float = 1.0,     # Temperature for softmax
                return_aux: bool = False,
                verbose: bool = True,
                seed: Optional[int] = None,
                **kwargs) -> Optional[AuxOutput]:
        """
        Predicts structure and metrics for a given sequence without optimization.

        Temporarily overrides optimization settings for a pure forward pass.

        Args:
            seq: Input sequence (string or array). If None, uses current state.
            bias: Optional sequence bias.
            num_models, num_recycles, models, sample_models: Model selection for prediction.
            dropout: Enable dropout during prediction (usually False).
            hard, soft, temp: Sequence generation mode for the prediction pass.
            return_aux: If True, returns the full auxiliary output.
            verbose: Print prediction log.
            seed: Optional random seed for this prediction run.
            **kwargs: Additional arguments passed to `run`.

        Returns:
            Auxiliary output dictionary if return_aux is True.
        """
        # Save current state (args, params, inputs) to restore later
        saved_state = {
            "args": copy_dict(self._args),
            "params": copy_dict(self._params),
            "inputs": copy_dict(self._inputs),
            "opt": copy_dict(self.opt) # Also save current options
        }

        try:
            # Set seed if provided for this specific prediction
            if seed is not None:
                self.set_seed(seed)

            # Set sequence if provided
            if seq is not None:
                # Need to handle bias application appropriately during set_seq or model input prep
                self.set_seq(seq=seq, bias=bias) # Assuming set_seq can handle bias

            # Override optimization options for prediction
            # Ensure pssm_hard=True if using PSSM input during prediction
            self.set_opt(hard=hard, soft=soft, temp=temp, dropout=dropout, pssm_hard=True)

            # Run the model in inference mode (no backpropagation)
            self.run(num_recycles=num_recycles, num_models=num_models,
                     sample_models=sample_models, models=models,
                     backprop=False, **kwargs) # Ensure backprop is False

            if verbose:
                self._print_log("Predict")

        finally:
            # Restore original state
            self._args = saved_state["args"]
            self._params = saved_state["params"]
            self._inputs = saved_state["inputs"]
            self.opt = saved_state["opt"]
            # Restore JAX key state? Depends if predict should be deterministic
            # relative to the main optimization process. Usually, we want predict
            # to use its own key sequence if seeded.

        # Return results if requested
        if return_aux:
            return self.aux
        return None

    # --------------------------------------------------------------------------
    # Design Strategies / Optimization Loops
    # --------------------------------------------------------------------------
    def _ramp_schedule(self, start_val: float, end_val: float,
                       current_iter: int, total_iters: int,
                       mode: str = "linear") -> float:
        """Calculates the value for a parameter based on a ramp schedule."""
        if total_iters <= 1: return end_val # Avoid division by zero

        progress = (current_iter + 1) / total_iters

        if mode == "linear":
            value = start_val + (end_val - start_val) * progress
        elif mode == "cosine_decay": # Example: Temperature decay
            value = end_val + (start_val - end_val) * (1 - progress)**2
        # Add other modes like exponential decay if needed
        else:
            raise ValueError(f"Unknown ramping mode: {mode}")

        return value

    def design(self, iters: int = 100,
               soft_start: float = 0.0, soft_end: Optional[float] = None,
               temp_start: float = 1.0, temp_end: Optional[float] = None,
               hard_start: float = 0.0, hard_end: Optional[float] = None,
               step_start: float = 1.0, step_end: Optional[float] = None,
               dropout: bool = True,
               opt: Optional[Dict[str, Any]] = None,
               weights: Optional[Dict[str, float]] = None,
               num_recycles: Optional[int] = None,
               ramp_recycles: bool = False,
               num_models: Optional[int] = None,
               sample_models: Optional[bool] = None,
               models: Optional[Union[List[Union[int, str]], int, str]] = None,
               backprop: bool = True,
               callback: Optional[Callable[['_AFDesign'], None]] = None,
               save_best: bool = False,
               verbose: int = 1):
        """
        Generic design loop with optional ramping of parameters.

        Args:
            iters: Number of optimization iterations.
            soft_start, soft_end: Start/end values for 'soft' sampling.
            temp_start, temp_end: Start/end values for sampling temperature.
            hard_start, hard_end: Start/end values for 'hard' sampling.
            step_start, step_end: Start/end values for learning rate scale ('step').
            dropout: Enable dropout during optimization.
            opt: Dictionary of optimization options to set.
            weights: Dictionary of loss weights to set.
            num_recycles: Fixed number of recycles, or max value if ramp_recycles=True.
            ramp_recycles: If True, linearly increase recycles from 0 to `num_recycles`.
            num_models, sample_models, models: Model selection parameters for each step.
            backprop: Enable gradient computation.
            callback: Function to call after each step.
            save_best: Save the best state encountered during the loop.
            verbose: Log printing frequency.
        """
        # Update options and weights if provided
        self.set_opt(opt, dropout=dropout) # Set dropout globally for the loop
        self.set_weights(weights)

        current_opt = self._inputs["opt"] # Get current options reference

        # Determine end values if not provided (use start values)
        schedule = {
            "soft": (soft_start, soft_end if soft_end is not None else soft_start),
            "temp": (temp_start, temp_end if temp_end is not None else temp_start),
            "hard": (hard_start, hard_end if hard_end is not None else hard_start),
            "step": (step_start, step_end if step_end is not None else step_start),
        }

        # Handle recycle ramping
        recycle_schedule = None
        if ramp_recycles:
            max_recycles = num_recycles if num_recycles is not None else current_opt.get("num_recycles", 0)
            recycle_schedule = (0, max_recycles) # Ramp from 0 to max
            current_num_recycles = 0 # Start with 0 recycles
        else:
            current_num_recycles = num_recycles # Use fixed value or None (will use opt default)


        for i in range(iters):
            # Update ramped parameters
            current_step_scale = 1.0
            for key, (start_val, end_val) in schedule.items():
                 # Use cosine decay for temperature, linear for others
                 mode = "cosine_decay" if key == "temp" else "linear"
                 current_val = self._ramp_schedule(start_val, end_val, i, iters, mode)

                 if key == "step":
                     current_step_scale = current_val # This scales the LR
                 else:
                     self.set_opt({key: current_val}) # Update the option

            # Update recycles if ramping
            if recycle_schedule:
                 current_num_recycles = round(
                     self._ramp_schedule(recycle_schedule[0], recycle_schedule[1], i, iters, "linear")
                 )

            # Calculate final learning rate scale for this step
            # Scale LR based on step schedule and potentially temperature/softness
            # Original logic: lr_scale = step * ((1 - opt["soft"]) + (opt["soft"] * opt["temp"]))
            effective_lr_scale = current_step_scale * (
                (1.0 - current_opt.get("soft", 0.0)) +
                (current_opt.get("soft", 0.0) * current_opt.get("temp", 1.0))
            )

            # Perform one optimization step
            self.step(lr_scale=effective_lr_scale,
                      num_recycles=current_num_recycles,
                      num_models=num_models,
                      sample_models=sample_models,
                      models=models,
                      backprop=backprop,
                      callback=callback,
                      save_best=save_best,
                      verbose=verbose)

    # --- Convenience wrappers for common design patterns ---

    def design_logits(self, iters: int = 100, **kwargs):
        """Optimize sequence logits directly (soft=0, hard=0)."""
        print("Running design_logits...")
        self.design(iters, soft_start=0.0, hard_start=0.0, **kwargs)

    def design_soft(self, iters: int = 100, temp: float = 1.0, **kwargs):
        """Optimize softmax(logits/temp) (soft=1, hard=0)."""
        print(f"Running design_soft (temp={temp})...")
        self.design(iters, soft_start=1.0, temp_start=temp, hard_start=0.0, **kwargs)

    def design_hard(self, iters: int = 100, **kwargs):
        """Optimize argmax(logits) via straight-through (soft=1, hard=1)."""
        print("Running design_hard...")
        # Temperature usually kept low for hard optimization
        temp = kwargs.pop("temp", 1e-2)
        self.design(iters, soft_start=1.0, temp_start=temp, hard_start=1.0, **kwargs)

    # --------------------------------------------------------------------------
    # Experimental / Advanced Design Strategies
    # --------------------------------------------------------------------------
    def design_3stage(self, soft_iters: int = 300, temp_iters: int = 100, hard_iters: int = 10,
                      ramp_recycles: bool = True, **kwargs):
        """
        Three-stage design: Logits -> Soft (decreasing temp) -> Hard.

        Args:
            soft_iters: Iterations for initial logits->soft phase (ramping soft 0->1).
            temp_iters: Iterations for soft phase (ramping temp 1->low).
            hard_iters: Iterations for final hard phase (hard=1, low temp).
            ramp_recycles: Apply recycle ramping during applicable stages.
            **kwargs: Passed to underlying design methods.
        """
        verbose = kwargs.get("verbose", 1)

        # Stage 1: Logits -> Softmax(logits/1.0)
        if soft_iters > 0:
            if verbose: print("\n--- Stage 1: Optimizing Logits → Soft (soft 0 → 1) ---")
            self.design(soft_iters, soft_start=0.0, soft_end=1.0, temp_start=1.0,
                        ramp_recycles=ramp_recycles, **kwargs)
            # Store the final logits (scaled by alpha if applicable)
            alpha = self._inputs.get("opt", {}).get("alpha", 1.0)
            if "seq" in self._params:
                 self._tmp["seq_logits"] = np.array(self._params["seq"]) * alpha
            else:
                 print("Warning: 'seq' not in params after Stage 1.")


        # Stage 2: Softmax(logits/1.0) -> Softmax(logits/low_temp)
        if temp_iters > 0:
            if verbose: print("\n--- Stage 2: Annealing Soft → Hard (temp 1.0 → 0.01) ---")
            # Start with soft=1, ramp temperature down
            self.design(temp_iters, soft_start=1.0, temp_start=1.0, temp_end=1e-2,
                        ramp_recycles=ramp_recycles, **kwargs) # Keep hard=0

        # Stage 3: Hard optimization
        if hard_iters > 0:
            if verbose: print("\n--- Stage 3: Hard Optimization (hard=1) ---")
            final_kwargs = kwargs.copy()
            final_kwargs["dropout"] = False # Turn off dropout for final refinement
            final_kwargs["save_best"] = True # Save the best result from this stage
            # Use all models for final stage unless specified otherwise
            final_kwargs.setdefault("num_models", len(self._model_names))
            final_kwargs.setdefault("sample_models", False)
            # Start with hard=1 and low temperature
            self.design_hard(hard_iters, temp=1e-2, **final_kwargs)


    def _mutate(self, sequence: np.ndarray,
                plddt: Optional[np.ndarray] = None,
                logits: Optional[np.ndarray] = None,
                mutation_rate: int = 1) -> np.ndarray:
        """
        Mutates sequence at random positions, optionally biased by pLDDT and logits.

        Args:
            sequence: The sequence(s) to mutate (shape [N, L] or [L]).
            plddt: pLDDT scores (shape [L]) to bias mutation position (lower pLDDT = higher prob).
            logits: Logits (shape [L, A] or [A]) to bias amino acid choice.
            mutation_rate: Number of mutations to introduce per sequence.

        Returns:
            The mutated sequence(s).
        """
        seq_array = np.array(sequence)
        if seq_array.ndim == 1: # Handle single sequence case
            seq_array = seq_array[None, :] # Add batch dimension N=1
        num_sequences, seq_len = seq_array.shape

        mutated_seq = seq_array.copy()

        # Prepare position sampling probabilities
        position_probs = np.ones(seq_len)
        if plddt is not None:
            # Higher probability for lower pLDDT: prob ~ (1 - plddt)
            # Clamp pLDDT between 0 and 1 if necessary
            clamped_plddt = np.clip(plddt, 0.0, 1.0)
            position_probs = np.maximum(1.0 - clamped_plddt, 0.0) # Use 1-plddt as weight
            position_probs[np.isnan(position_probs)] = 0.0 # Handle potential NaNs

        # Apply fixed position mask
        fix_pos_mask = self._inputs.get("fix_pos", np.zeros(seq_len, dtype=bool))[:seq_len]
        if len(fix_pos_mask) == seq_len:
             position_probs[fix_pos_mask] = 0.0 # Prevent sampling fixed positions
             # Ensure wild type is enforced at fixed positions
             wt_aatype = self._inputs.get("wt_aatype", [])[:seq_len]
             if len(wt_aatype) == seq_len:
                 mutated_seq[:, fix_pos_mask] = wt_aatype[fix_pos_mask]
        else:
             print("Warning: fix_pos length mismatch, not applied in _mutate.")


        if position_probs.sum() == 0.0:
            print("Warning: No mutable positions available in _mutate.")
            return mutated_seq.squeeze() if sequence.ndim == 1 else mutated_seq # Return original if no positions


        # Normalize position probabilities
        position_probs /= position_probs.sum()

        # Prepare logits for amino acid sampling
        aa_logits = np.zeros((seq_len, self._args["alphabet_size"]))
        if logits is not None:
            logits_arr = np.array(logits)
            if logits_arr.ndim == 3 and logits_arr.shape[0] == num_sequences: # Shape [N, L, A]
                # Average logits across batch dim if provided per sequence
                aa_logits = logits_arr.mean(axis=0)
            elif logits_arr.ndim == 2 and logits_arr.shape[0] == seq_len: # Shape [L, A]
                aa_logits = logits_arr
            elif logits_arr.ndim == 1 and logits_arr.shape[0] == self._args["alphabet_size"]: # Shape [A]
                 # Broadcast same AA bias to all positions
                 aa_logits = np.tile(logits_arr, (seq_len, 1))
            else:
                 print(f"Warning: Unexpected logits shape {logits_arr.shape} in _mutate. Using zero logits.")


        for _ in range(mutation_rate): # Introduce specified number of mutations
            for n in range(num_sequences): # Mutate each sequence independently
                # 1. Sample position to mutate
                mut_pos = np.random.choice(seq_len, p=position_probs)

                # 2. Sample new amino acid for that position
                current_aa = mutated_seq[n, mut_pos]
                pos_logits = aa_logits[mut_pos].copy()

                # Prevent sampling the same amino acid (set its logit to large negative)
                pos_logits[current_aa] -= 1e8

                # Sample new amino acid using categorical distribution from softmax probabilities
                new_aa = categorical(softmax(pos_logits)) # Assuming categorical takes probabilities

                # 3. Apply mutation
                mutated_seq[n, mut_pos] = new_aa

        return mutated_seq.squeeze() if sequence.ndim == 1 else mutated_seq


    def design_semigreedy(self, iters: int = 100, tries: int = 10,
                          dropout: bool = False, # Usually False for greedy search
                          save_best: bool = True,
                          seq_logits: Optional[np.ndarray] = None,
                          e_tries: Optional[int] = None, **kwargs):
        """
        Semi-greedy optimization: Mutate current best sequence and keep the best mutation.

        Args:
            iters: Number of greedy steps.
            tries: Number of mutations to try at each step (start value).
            dropout: Enable dropout (usually False).
            save_best: Save the overall best state found.
            seq_logits: Optional logits to bias mutations.
            e_tries: Number of mutations to try at the end (for ramping).
            **kwargs: Passed to `predict` method.
        """
        if verbose := kwargs.pop("verbose", 1):
            print("\n--- Running Semi-Greedy Optimization ---")

        if e_tries is None: e_tries = tries # Default end_tries = start_tries

        # Get starting sequence (from aux, params, or wildtype)
        # Assuming bias shape is [..., L, A]
        bias = self._inputs.get("bias", np.zeros((self._len, self._args["alphabet_size"])))
        if bias.ndim > 2: bias = bias[0] # Use first bias if batch dim exists
        if bias.shape[0] != self._len or bias.shape[1] != self._args["alphabet_size"]:
             print(f"Warning: Bias shape mismatch ({bias.shape}). Using zero bias.")
             bias = np.zeros((self._len, self._args["alphabet_size"]))


        if hasattr(self, "aux") and "seq" in self.aux:
            # Use argmax of the sequence probabilities from the last run
            last_seq_prob = self.aux["seq"] # Shape [L, A] or [N, L, A]
            if last_seq_prob.ndim == 3: last_seq_prob = last_seq_prob[0] # Use first model's seq
            current_seq = last_seq_prob[:self._len].argmax(-1)[None, :] # Shape [1, L]
        else:
            # Use argmax of current parameters + bias
            param_seq = self._params.get("seq", np.zeros((self._len, self._args["alphabet_size"])))
            current_seq = (param_seq + bias).argmax(-1)[None, :] # Shape [1, L]

        # Combine external logits with internal bias for mutation sampling
        mutation_logits = bias.copy()
        if seq_logits is not None:
            # Ensure seq_logits has shape [L, A]
            if seq_logits.shape == mutation_logits.shape:
                 mutation_logits += seq_logits
            else:
                 print(f"Warning: seq_logits shape mismatch ({seq_logits.shape}). Not added to bias.")


        # Get model selection flags from kwargs
        model_flags = {k: kwargs.pop(k, None) for k in ["num_models", "sample_models", "models"]}

        # Initial prediction to get starting pLDDT and loss
        initial_aux = self.predict(seq=current_seq.squeeze(), return_aux=True, verbose=False,
                                   dropout=dropout, **model_flags, **kwargs)
        current_plddt = initial_aux.get("plddt", np.ones(self._len) * 0.5)[:self._len] # Default plddt if missing
        current_loss = initial_aux.get("loss", np.inf)
        if verbose: print(f"Initial loss: {current_loss:.4f}")


        # Optimization loop
        for i in range(iters):
            mutation_candidates = []
            # Determine number of tries for this iteration (ramping)
            num_tries = int(self._ramp_schedule(tries, e_tries, i, iters, "linear"))

            # Get model numbers once for all tries in this iteration for consistency
            model_nums = self._get_model_nums(**model_flags)

            for _ in range(num_tries):
                # Generate a mutated sequence
                mutated_seq = self._mutate(seq=current_seq, plddt=current_plddt,
                                           logits=mutation_logits, mutation_rate=1) # Single mutation

                # Predict metrics for the mutated sequence
                aux_output = self.predict(seq=mutated_seq.squeeze(), return_aux=True,
                                          model_nums=model_nums, verbose=False,
                                          dropout=dropout, **kwargs)
                mutation_candidates.append({"aux": aux_output, "seq": mutated_seq})

            # Select the best mutation from the candidates based on loss
            losses = [cand["aux"].get("loss", np.inf) for cand in mutation_candidates]
            best_candidate_idx = np.argmin(losses)
            best_candidate = mutation_candidates[best_candidate_idx]
            best_loss = losses[best_candidate_idx]

            # Update current state if the best mutation is better
            if best_loss < current_loss:
                 current_seq = best_candidate["seq"]
                 current_loss = best_loss
                 current_plddt = best_candidate["aux"].get("plddt", current_plddt)[:self._len]
                 self.aux = best_candidate["aux"] # Update main aux with the best candidate's aux
                 # Update internal sequence representation (e.g., params['seq']) if needed
                 self.set_seq(seq=current_seq.squeeze(), bias=bias) # Update sequence state

                 # Save results (logs, trajectory, best overall)
                 self._save_results(save_best=save_best, verbose=False) # Verbose printing handled below
                 if verbose and ((i + 1) % verbose == 0):
                      self._print_log(f"Greedy Step {i+1}/{iters} (Try {num_tries})", aux_data=self.aux)

            elif verbose and ((i + 1) % verbose == 0):
                 # Print log even if no improvement, using the *previous* best aux
                 self._print_log(f"Greedy Step {i+1}/{iters} (Try {num_tries}, no improvement)", aux_data=self.aux)


            self._k += 1 # Increment main iteration counter


    def design_pssm_semigreedy(self, soft_iters: int = 300, hard_iters: int = 32,
                               tries: int = 10, e_tries: Optional[int] = None,
                               ramp_recycles: bool = True, ramp_models: bool = True,
                               **kwargs):
        """
        Combines initial soft optimization (like 3-stage) with semi-greedy search.

        Args:
            soft_iters: Iterations for the initial soft optimization stage.
            hard_iters: Iterations for the semi-greedy stage.
            tries, e_tries: Number of tries for semi-greedy search.
            ramp_recycles: Use recycle ramping in the soft stage.
            ramp_models: Gradually increase number of models used during semi-greedy stage.
            **kwargs: Passed to underlying design methods.
        """
        verbose = kwargs.get("verbose", 1)

        # Stage 1: Soft optimization (Logits -> Soft)
        if soft_iters > 0:
            if verbose: print("\n--- Running PSSM Semi-Greedy: Stage 1 (Soft Optimization) ---")
            # Run only the first part of 3-stage (logits -> soft=1, temp=1)
            self.design_3stage(soft_iters=soft_iters, temp_iters=0, hard_iters=0,
                               ramp_recycles=ramp_recycles, **kwargs)
            # Pass the learned logits to the greedy stage
            kwargs["seq_logits"] = self._tmp.get("seq_logits", None)

        # Stage 2: Semi-greedy optimization
        if hard_iters > 0:
            if verbose: print("\n--- Running PSSM Semi-Greedy: Stage 2 (Semi-Greedy Search) ---")
            greedy_kwargs = kwargs.copy()
            greedy_kwargs["dropout"] = False # Turn off dropout for greedy search
            greedy_kwargs["verbose"] = verbose # Pass verbose level

            if ramp_models:
                # Gradually increase the number of models used
                total_models_available = len(greedy_kwargs.get("models", self._model_names))
                iters_per_model_count = max(1, hard_iters // total_models_available) # Distribute iters

                for m in range(total_models_available):
                    num_models_this_round = m + 1
                    iters_this_round = iters_per_model_count
                    # Assign remaining iters to the last round
                    if m == total_models_available - 1:
                        iters_this_round = hard_iters - (m * iters_per_model_count)

                    if iters_this_round <= 0: continue

                    if verbose and m > 0:
                        print(f"\nIncreasing number of models to {num_models_this_round} for {iters_this_round} iterations.")

                    greedy_kwargs["num_models"] = num_models_this_round
                    greedy_kwargs["sample_models"] = False # Don't sample when ramping
                    # Only save best on the final round with all models
                    greedy_kwargs["save_best"] = (num_models_this_round == total_models_available)

                    self.design_semigreedy(iters=iters_this_round, tries=tries, e_tries=e_tries,
                                           **greedy_kwargs)
            else:
                # Run semi-greedy with fixed model settings
                greedy_kwargs.setdefault("num_models", len(self._model_names))
                greedy_kwargs.setdefault("sample_models", False)
                greedy_kwargs["save_best"] = True # Save best if not ramping models

                self.design_semigreedy(iters=hard_iters, tries=tries, e_tries=e_tries,
                                       **greedy_kwargs)

    # --------------------------------------------------------------------------
    # Experimental MCMC Optimizer
    # --------------------------------------------------------------------------
    def _design_mcmc(self, steps: int = 1000, half_life: int = 200,
                     T_init: float = 0.01, mutation_rate: int = 1,
                     seq_logits: Optional[np.ndarray] = None,
                     save_best: bool = True, **kwargs):
        """
        Experimental MCMC optimization with simulated annealing.

        Args:
            steps: Number of MCMC steps.
            half_life: Half-life for exponential temperature decay.
            T_init: Initial temperature.
            mutation_rate: Number of mutations per step.
            seq_logits: Optional logits to bias mutations.
            save_best: Save the overall best state found.
            **kwargs: Passed to `predict`.
        """
        if verbose := kwargs.pop("verbose", 1):
            print("\n--- Running MCMC with Simulated Annealing (Experimental) ---")

        # Get model selection flags
        model_flags = {k: kwargs.pop(k, None) for k in ["num_models", "sample_models", "models"]}

        # Initialization
        current_plddt = None
        best_loss = np.inf
        current_loss = np.inf

        # Bias for sequence generation and mutation
        bias = self._inputs.get("bias", np.zeros((self._len, self._args["alphabet_size"])))
        if bias.ndim > 2: bias = bias[0] # Use first bias if batch dim exists
        if bias.shape[0] != self._len or bias.shape[1] != self._args["alphabet_size"]:
             bias = np.zeros((self._len, self._args["alphabet_size"]))

        # Starting sequence
        param_seq = self._params.get("seq", np.zeros((self._len, self._args["alphabet_size"])))
        current_seq = (param_seq + bias).argmax(-1)[None, :] # Shape [1, L]

        # Logits for mutation bias
        mutation_logits = bias.copy()
        if seq_logits is not None:
            if seq_logits.shape == mutation_logits.shape:
                 mutation_logits += seq_logits
            else:
                 print("Warning: seq_logits shape mismatch in MCMC. Not added.")


        # MCMC Loop
        for i in range(steps):
            # Calculate current temperature using exponential decay
            temperature = T_init * (np.exp(np.log(0.5) / half_life) ** i)

            # Propose a new state (mutate sequence)
            if i == 0:
                # Evaluate the initial sequence
                mutated_seq = current_seq
            else:
                mutated_seq = self._mutate(seq=current_seq, plddt=current_plddt,
                                           logits=mutation_logits,
                                           mutation_rate=mutation_rate)

            # Evaluate the proposed state (get loss)
            model_nums = self._get_model_nums(**model_flags)
            aux_output = self.predict(seq=mutated_seq.squeeze(), return_aux=True,
                                      verbose=False, model_nums=model_nums, **kwargs)
            proposed_loss = aux_output.get("loss", np.inf)

            # Acceptance probability (Metropolis criterion)
            delta_loss = proposed_loss - current_loss
            accept = False
            if i == 0: # Always accept the first evaluation
                 accept = True
            elif delta_loss < 0: # Always accept improvements
                 accept = True
            elif temperature > 1e-6: # Avoid division by zero or NaNs
                 acceptance_prob = np.exp(-delta_loss / temperature)
                 if np.random.uniform() < acceptance_prob:
                     accept = True

            # Update current state if accepted
            if accept:
                current_seq = mutated_seq
                current_loss = proposed_loss
                # Use average pLDDT across models if available
                if "all" in aux_output and "plddt" in aux_output["all"]:
                     current_plddt = aux_output["all"]["plddt"].mean(axis=0)[:self._len]
                else:
                     current_plddt = aux_output.get("plddt", np.ones(self._len) * 0.5)[:self._len]

                # Update main aux if accepted
                self.aux = aux_output

                # Check if this is the best state found so far
                if current_loss < best_loss:
                    best_loss = current_loss
                    self._k = i # Store the step number where best was found
                    # Update sequence state
                    self.set_seq(seq=current_seq.squeeze(), bias=bias)
                    # Save results (log, trajectory, best overall)
                    self._save_results(save_best=save_best, verbose=False) # Verbose handled below

                if verbose and ((i + 1) % verbose == 0):
                    status = "Accept" if i > 0 else "Init"
                    self._print_log(f"MCMC Step {i+1}/{steps} ({status}, T={temperature:.4f})", aux_data=self.aux)

            elif verbose and ((i + 1) % verbose == 0):
                 # Print log even if rejected, using the *previous* accepted state's aux
                 self._print_log(f"MCMC Step {i+1}/{steps} (Reject, T={temperature:.4f})", aux_data=self.aux)

        if verbose: print("MCMC Finished.")

