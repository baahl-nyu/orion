import os
from typing import Literal, List
from dataclasses import dataclass, field


@dataclass
class CKKSParameters:
    logn: int
    logq: List[int]
    logp: List[int]
    logscale: int = field(default=None)
    h: int = 192
    ringtype: str = "standard"
    boot_logp: List[int] = field(default=None)

    def __post_init__(self):
        if self.logq and self.logp and len(self.logp) > len(self.logq):
            raise ValueError(
                f"Invalid parameters: The length of logp ({len(self.logp)}) "
                f"cannot exceed the length of logq ({len(self.logq)})."
            )
        
        valid_ringtypes = {"standard", "conjugateinvariant"}
        ring = self.ringtype.lower()
        if ring not in valid_ringtypes:
            raise ValueError(
                f"Invalid ringtype: {self.ringtype}. Only 'Standard' or "
                f"'ConjugateInvariant' ring types are supported."
            )

        self.logscale = self.logscale or self.logq[-1]
        self.boot_logp = self.boot_logp or self.logp
        self.logslots = (
            self.logn-1 if self.ringtype.lower() == "standard" 
            else self.logn
        )

    def __str__(self):
        if self.ringtype.lower() == "standard":
            ring_type_display = "Standard"
        else:
            ring_type_display = "Conjugate invariant"
        
        output = [
            "CKKS Parameters:",
            f"  Ring degree (N): {1 << self.logn} (LogN = {self.logn})",
            f"  Number of slots (n): {1 << self.logslots}",
            f"  Effective levels (L_eff): {len(self.logq) - 1}"
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
    margin: int = 2
    fuse_modules: bool = True
    debug: bool = True
    embedding_method: Literal["hybrid", "square"] = "hybrid"
    backend: Literal["lattigo", "openfhe", "heaan"] = "lattigo"
    device: Literal["cpu", "gpu"] = "cpu"
    io_mode: Literal["none", "save", "load"] = "none"
    diags_path: str = ""
    keys_path: str = ""

    def __str__(self) -> str:
        output = [
            "Orion Parameters:",
            f"  Backend: {self.backend}",
            f"  Margin: {self.margin}",
            f"  Embedding Method: {self.embedding_method}",
            f"  Fuse Modules: {self.fuse_modules}",
            f"  Debug Mode: {self.debug}",
            f"  Device: {self.device}"
        ]
        
        output.append(f"  I/O Mode: {self.io_mode}")
        if self.diags_path:
            output.append(f"  Diagonals Path: {self.diags_path}")
        if self.keys_path:
            output.append(f"  Keys Path: {self.keys_path}")
        
        return "\n".join(output)


@dataclass
class NewParameters:
    params_json: dict
    ckks_params: CKKSParameters = field(init=False)
    orion_params: OrionParameters = field(init=False)

    def __post_init__(self):
        params = self.params_json
        ckks_params = {
            k.lower(): v for k, v in params.get("ckks_params", {}).items()}
        boot_params = {
            k.lower(): v for k, v in params.get("boot_params", {}).items()}
        orion_params = {
            k.lower(): v for k, v in params.get("orion", {}).items()}

        self.ckks_params = CKKSParameters(
            **ckks_params, boot_logp=boot_params.get("logp")
        )
        self.orion_params = OrionParameters(**orion_params)

        # Finally, we'll delete existing keys/diagonals if the user  
        # specifies to overwrite them.
        if self.get_io_mode() == "save" and self.io_paths_exist():
            self.reset_stored_keys()
            self.reset_stored_diags()

    def __str__(self) -> str:
        border = "=" * 50
        return f"\n{border}\n{self.ckks_params}\n\n{self.orion_params}\n{border}\n"
        
    def get_logn(self):
        return self.ckks_params.logn

    def get_margin(self):
        return self.orion_params.margin
    
    def get_fuse_modules(self):
        return self.orion_params.fuse_modules
    
    def get_debug_status(self):
        return self.orion_params.debug

    def get_backend(self):
        return self.orion_params.backend.lower()

    def get_device(self):
        return self.orion_params.device.lower()
    
    def get_logq(self):
        return self.ckks_params.logq
    
    def get_logp(self):
        return self.ckks_params.logp
    
    def get_logscale(self):
        return self.ckks_params.logscale
    
    def get_default_scale(self):
        return 1 << self.ckks_params.logscale
    
    def get_hamming_weight(self):
        return self.ckks_params.h
    
    def get_ringtype(self):
        return self.ckks_params.ringtype.lower()

    def get_max_level(self):
        return len(self.ckks_params.logq) - 1

    def get_slots(self):
        return int(1 << self.ckks_params.logslots)

    def get_ring_degree(self):
        return int(1 << self.ckks_params.logn)

    def get_embedding_method(self):
        return self.orion_params.embedding_method.lower()

    def get_diags_path(self):
        path = self.orion_params.diags_path
        return os.path.abspath(os.path.join(os.getcwd(), path))

    def get_keys_path(self):
        path = self.orion_params.keys_path
        return os.path.abspath(os.path.join(os.getcwd(), path))

    def get_io_mode(self):
        return self.orion_params.io_mode.lower()

    def get_boot_logp(self):
        return self.ckks_params.boot_logp

    def io_paths_exist(self):
        return bool(self.get_diags_path()) and bool(self.get_keys_path())

    def reset_stored_file(self, path: str, file_type: str):
        if self.get_io_mode() == "save" and path:
            print(f"Deleting existing {file_type} at {path}")
            abs_path = os.path.abspath(os.path.join(os.getcwd(), path))
            if os.path.exists(abs_path):
                os.remove(abs_path)

    def reset_stored_diags(self):
        self.reset_stored_file(self.get_diags_path(), "diagonals")

    def reset_stored_keys(self):
        self.reset_stored_file(self.get_keys_path(), "keys")