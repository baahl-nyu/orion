import os
import logging
from typing import Literal, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger("orion")


@dataclass
class CKKSParameters:
    """Parameters for the CKKS encryption scheme."""
    
    logn: int
    logq: List[int]
    logp: List[int]
    logscale: Optional[int] = field(default=None)
    h: int = 192
    ringtype: str = "standard"
    boot_logp: Optional[List[int]] = field(default=None)

    def __post_init__(self):
        # Validate logp and logq lengths
        if self.logq and self.logp and len(self.logp) > len(self.logq):
            error_msg = (
                f"Invalid parameters: The length of logp ({len(self.logp)}) "
                f"cannot exceed the length of logq ({len(self.logq)})."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Validate ring type
        valid_ringtypes = {"standard", "conjugateinvariant"}
        ring = self.ringtype.lower()
        if ring not in valid_ringtypes:
            error_msg = (
                f"Invalid ringtype: {self.ringtype}. Only 'Standard' or "
                f"'ConjugateInvariant' ring types are supported."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Set defaults
        self.logscale = self.logscale or self.logq[-1]
        self.boot_logp = self.boot_logp or self.logp
        self.logslots = (
            self.logn - 1 if self.ringtype.lower() == "standard" 
            else self.logn
        )
        
        logger.debug(f"Initialized CKKS parameters: N=2^{self.logn}, slots=2^{self.logslots}, "
                    f"L_eff={len(self.logq)-1}, scale=2^{self.logscale}")

    def __str__(self):
        ring_type_display = (
            "Standard" if self.ringtype.lower() == "standard" 
            else "Conjugate invariant"
        )
        
        output = [
            "CKKS Parameters:",
            f"  Ring degree (N): {1 << self.logn} (LogN = {self.logn})",
            f"  Number of slots (n): {1 << self.logslots}",
            f"  Effective levels (L_eff): {len(self.logq) - 1}",
            f"  Ring type: {ring_type_display}",
            f"  Scale: 2^{self.logscale}",
            f"  Hamming weight: {self.h}"
        ]
        
        # Format LogQ values
        logq_str = ", ".join(str(q) for q in self.logq)
        output.append(f"  LogQ: [{logq_str}] (length: {len(self.logq)})")
        
        # Format LogP values
        logp_str = ", ".join(str(p) for p in self.logp)
        output.append(f"  LogP: [{logp_str}] (length: {len(self.logp)})")
        
        # Format Boot LogP values if different from LogP
        if self.boot_logp != self.logp:
            boot_logp_str = ", ".join(str(p) for p in self.boot_logp)
            output.append(f"  Boot LogP: [{boot_logp_str}] (length: {len(self.boot_logp)})")
        
        return "\n".join(output)


@dataclass
class OrionParameters:
    """Orion-specific configuration parameters."""
    
    batch_size: int = 1
    margin: int = 2
    fuse_modules: bool = True
    debug: bool = True
    embedding_method: Literal["hybrid", "square"] = "hybrid"
    backend: Literal["lattigo", "openfhe", "heaan"] = "lattigo"
    io_mode: Literal["none", "save", "load"] = "none"
    diags_path: str = ""
    keys_path: str = ""

    def __post_init__(self):
        # Validate batch size
        if self.batch_size < 1:
            error_msg = f"Batch size must be >= 1, got {self.batch_size}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Validate margin
        if self.margin < 1:
            error_msg = f"Margin must be >= 1, got {self.margin}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Validate backend
        valid_backends = {"lattigo", "openfhe", "heaan"}
        if self.backend.lower() not in valid_backends:
            error_msg = f"Invalid backend: {self.backend}. Must be one of {valid_backends}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Validate embedding method
        valid_methods = {"hybrid", "square"}
        if self.embedding_method.lower() not in valid_methods:
            error_msg = f"Invalid embedding method: {self.embedding_method}. Must be one of {valid_methods}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Validate IO mode
        valid_modes = {"none", "save", "load"}
        if self.io_mode.lower() not in valid_modes:
            error_msg = f"Invalid IO mode: {self.io_mode}. Must be one of {valid_modes}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.debug(f"Initialized Orion parameters: batch_size={self.batch_size}, "
                    f"margin={self.margin}, backend={self.backend}, "
                    f"embedding={self.embedding_method}, io_mode={self.io_mode}")

    def __str__(self) -> str:
        output = [
            "Orion Parameters:",
            f"  Backend: {self.backend}",
            f"  Batch Size: {self.batch_size}",
            f"  Margin: {self.margin}",
            f"  Embedding Method: {self.embedding_method}",
            f"  Fuse Modules: {self.fuse_modules}",
            f"  Debug Mode: {self.debug}"
        ]
        
        output.append(f"  I/O Mode: {self.io_mode}")
        if self.diags_path:
            output.append(f"  Diagonals Path: {self.diags_path}")
        if self.keys_path:
            output.append(f"  Keys Path: {self.keys_path}")
        
        return "\n".join(output)


@dataclass
class NewParameters:
    """Main parameters container for Orion."""
    
    params_json: dict
    ckks_params: CKKSParameters = field(init=False)
    orion_params: OrionParameters = field(init=False)

    def __post_init__(self):
        logger.info("Initializing Orion parameters from configuration")
        
        params = self.params_json
        
        # Extract and normalize parameter sections
        ckks_params = self._normalize_params(params.get("ckks_params", {}))
        boot_params = self._normalize_params(params.get("boot_params", {}))
        orion_params = self._normalize_params(params.get("orion", {}))
        
        # Log the configuration being loaded
        logger.debug(f"CKKS params: {ckks_params}")
        logger.debug(f"Boot params: {boot_params}")
        logger.debug(f"Orion params: {orion_params}")

        # Initialize sub-parameters
        try:
            self.ckks_params = CKKSParameters(
                **ckks_params, 
                boot_logp=boot_params.get("logp")
            )
        except (TypeError, ValueError) as e:
            logger.error(f"Failed to initialize CKKS parameters: {e}")
            raise
        
        try:
            self.orion_params = OrionParameters(**orion_params)
        except (TypeError, ValueError) as e:
            logger.error(f"Failed to initialize Orion parameters: {e}")
            raise

        # Handle IO mode and file cleanup
        if self.get_io_mode() == "save" and self.io_paths_exist():
            logger.info("IO mode is 'save' - cleaning up existing files")
            self.reset_stored_keys()
            self.reset_stored_diags()
        elif self.get_io_mode() == "load":
            self._validate_load_paths()
        
        logger.info("Parameters initialized successfully")

    @staticmethod
    def _normalize_params(params_dict: dict) -> dict:
        """Convert all keys to lowercase for consistent access."""
        return {k.lower(): v for k, v in params_dict.items()}
    
    def _validate_load_paths(self):
        """Validate that required files exist when loading."""
        if self.get_io_mode() == "load":
            if self.get_diags_path() and not os.path.exists(self.get_diags_path()):
                logger.warning(f"Diagonals file not found at {self.get_diags_path()}")
            if self.get_keys_path() and not os.path.exists(self.get_keys_path()):
                logger.warning(f"Keys file not found at {self.get_keys_path()}")

    def __str__(self) -> str:
        border = "=" * 50
        return f"\n{border}\n{self.ckks_params}\n\n{self.orion_params}\n{border}\n"
    
    # CKKS parameter getters
    def get_logn(self) -> int:
        return self.ckks_params.logn

    def get_logq(self) -> List[int]:
        return self.ckks_params.logq
    
    def get_logp(self) -> List[int]:
        return self.ckks_params.logp
    
    def get_logscale(self) -> int:
        return self.ckks_params.logscale
    
    def get_default_scale(self) -> int:
        return 1 << self.ckks_params.logscale
    
    def get_hamming_weight(self) -> int:
        return self.ckks_params.h
    
    def get_ringtype(self) -> str:
        return self.ckks_params.ringtype.lower()

    def get_max_level(self) -> int:
        return len(self.ckks_params.logq) - 1

    def get_slots(self) -> int:
        return 1 << self.ckks_params.logslots

    def get_ring_degree(self) -> int:
        return 1 << self.ckks_params.logn

    def get_boot_logp(self) -> List[int]:
        return self.ckks_params.boot_logp

    # Orion parameter getters
    def get_batch_size(self) -> int:
        return self.orion_params.batch_size
    
    def get_margin(self) -> int:
        return self.orion_params.margin
    
    def get_fuse_modules(self) -> bool:
        return self.orion_params.fuse_modules
    
    def get_debug_status(self) -> bool:
        return self.orion_params.debug

    def get_backend(self) -> str:
        return self.orion_params.backend.lower()

    def get_embedding_method(self) -> str:
        return self.orion_params.embedding_method.lower()

    def get_io_mode(self) -> str:
        return self.orion_params.io_mode.lower()

    def get_diags_path(self) -> str:
        if not self.orion_params.diags_path:
            return ""
        return os.path.abspath(os.path.join(os.getcwd(), self.orion_params.diags_path))

    def get_keys_path(self) -> str:
        if not self.orion_params.keys_path:
            return ""
        return os.path.abspath(os.path.join(os.getcwd(), self.orion_params.keys_path))

    def io_paths_exist(self) -> bool:
        return bool(self.get_diags_path()) and bool(self.get_keys_path())

    def reset_stored_file(self, path: str, file_type: str):
        """Remove a stored file if in save mode."""
        if self.get_io_mode() == "save" and path:
            abs_path = os.path.abspath(os.path.join(os.getcwd(), path))
            if os.path.exists(abs_path):
                logger.info(f"Deleting existing {file_type} at {abs_path}")
                try:
                    os.remove(abs_path)
                    logger.debug(f"Successfully deleted {file_type}")
                except OSError as e:
                    logger.error(f"Failed to delete {file_type}: {e}")
                    raise
            else:
                logger.debug(f"No existing {file_type} to delete at {abs_path}")

    def reset_stored_diags(self):
        """Remove stored diagonals file."""
        self.reset_stored_file(self.get_diags_path(), "diagonals")

    def reset_stored_keys(self):
        """Remove stored keys file."""
        self.reset_stored_file(self.get_keys_path(), "keys")
    
    def summary(self) -> dict:
        """Get a summary of all parameters as a dictionary."""
        return {
            "ckks": {
                "ring_degree": self.get_ring_degree(),
                "slots": self.get_slots(),
                "max_level": self.get_max_level(),
                "scale": self.get_default_scale(),
                "ring_type": self.get_ringtype()
            },
            "orion": {
                "batch_size": self.get_batch_size(),
                "margin": self.get_margin(),
                "backend": self.get_backend(),
                "embedding": self.get_embedding_method(),
                "fuse_modules": self.get_fuse_modules(),
                "debug": self.get_debug_status(),
                "io_mode": self.get_io_mode()
            }
        }