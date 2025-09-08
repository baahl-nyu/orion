import h5py
import torch
import numpy as np

from orion.backend.python.tensors import CipherTensor


class NewEvaluator:
    def __init__(self, scheme):
        self.scheme = scheme 
        self.params = scheme.params
        self.backend = scheme.backend
        self.evaluator = scheme.evaluator

        self.embed_method = self.params.get_embedding_method()
        self.io_mode = self.params.get_io_mode()
        self.diags_path = self.params.get_diags_path()
        self.keys_path = self.params.get_keys_path()

        self.saved_rotation_keys = set()
        self.new_evaluator()

    def new_evaluator(self):
        self.backend.NewLinearTransformEvaluator()

    def generate_transforms(self, linear_layer):
        layer_name = linear_layer.name
        diagonals = linear_layer.diagonals 
        level = linear_layer.level
        bsgs_ratio = linear_layer.bsgs_ratio

        # Generate all linear transforms block by block.
        lintransf_ids = {}        
        for (row, col), diags in diagonals.items(): 
            diags_idxs, diags_data = [], []
            for idx, diag in diags.items(): 
                diags_idxs.append(idx)
                diags_data.extend(diag)

            lintransf_id = self.backend.GenerateLinearTransform(
                diags_idxs, diags_data, level, bsgs_ratio, self.io_mode
            )
            lintransf_ids[(row, col)] = lintransf_id

            # Now we can generate any new rotation keys needed for
            # this linear transform.
            self.generate_rotation_keys(lintransf_id)
            if self.io_mode == "save":
                self.save_plaintext_diagonals(
                    layer_name, lintransf_id, row, col, diags_idxs
                )

        return lintransf_ids
    
    def get_required_rotation_keys(self, transform_id):
        return self.backend.GetLinearTransformRotationKeys(transform_id)

    def generate_rotation_keys(self, transform_id):
        curr_keys = self.get_required_rotation_keys(transform_id)

        # Only generate keys that don't exist yet. Depending on the I/O
        # mode, we may also save these keys immediately rather than keep
        # them in RAM.
        keys_to_gen = set(curr_keys).difference(self.saved_rotation_keys)
        self.saved_rotation_keys.update(keys_to_gen)

        if self.io_mode == "none":
            for key in keys_to_gen:
                self.backend.GenerateLinearTransformRotationKey(key)

        elif self.io_mode == "save":
            with h5py.File(self.keys_path, "a") as f:
                for key in keys_to_gen:
                    key_str = str(key)
                    if key_str in f: # don't regenerate the key
                        continue
                    
                    # We'll generate, serialize, and then save the key
                    serial_key, ptr = self.backend.GenerateAndSerializeRotationKey(key)
                    try:
                        f.create_dataset(key_str, data=serial_key)
                    finally:
                        self.backend.FreeCArray(ptr)

    def save_transforms(self, linear_layer):
        layer_name = linear_layer.name
        diagonals = linear_layer.diagonals 
        on_bias = linear_layer.on_bias 
        output_rotations = linear_layer.output_rotations 
        input_shape = linear_layer.input_shape 
        output_shape = linear_layer.output_shape
        input_min = linear_layer.input_min
        input_max = linear_layer.input_max
        output_min = linear_layer.output_min 
        output_max = linear_layer.output_max

        print("└── saving... ", end="", flush=True)
        with h5py.File(self.diags_path, "a") as f:
            layer = f.require_group(layer_name)

            layer.create_dataset("embedding_method", data=self.embed_method)
            layer.create_dataset("output_rotations", data=output_rotations)
            layer.create_dataset("on_bias", data=on_bias.numpy())
            layer.create_dataset("input_shape", data=list(input_shape))
            layer.create_dataset("output_shape", data=list(output_shape))
            layer.create_dataset("input_min", data=input_min)
            layer.create_dataset("input_max", data=input_max)
            layer.create_dataset("output_min", data=output_min)
            layer.create_dataset("output_max", data=output_max)

            diags_group = layer.require_group("diagonals")
            for (row, col), diags in diagonals.items():
                block_idx = f"{row}_{col}"
                block_diags_group = diags_group.create_group(block_idx)
                
                # Iterate over all diagonals in the block and save
                for diag_idx, diag_data in diags.items():
                    block_diags_group.create_dataset(str(diag_idx), data=diag_data)
                    diags[diag_idx] = [] # delete after saving

        print("done!")

    def load_transforms(self, linear_layer):
        self._verify_layer_compatibility(linear_layer)

        layer_name = linear_layer.name
        on_bias = linear_layer.on_bias
        output_rotations = linear_layer.output_rotations

        with h5py.File(self.diags_path, "a") as f:
            layer = f[layer_name]

            # Load the diagonals back into the correct struct
            all_diagonals = {}
            diag_group = layer["diagonals"]
            for block in diag_group:
                row, col = map(int, block.split("_")) # 0_1 -> (0,1)
                diags = {}
                block_group = diag_group[block]
                for diag_idx in block_group:
                    diag_data = block_group[diag_idx][:]
                    diags[int(diag_idx)] = diag_data 
                all_diagonals[(row, col)] = diags

        return all_diagonals, on_bias, output_rotations

    def evaluate_transforms(self, linear_layer, in_ctensor):
        layer_name = linear_layer.name
        out_shape = linear_layer.output_shape
        fhe_out_shape = linear_layer.fhe_output_shape 

        # Order-preserving flatten that can be mapped back to 
        # (row, col) format in backend via len(in_ctensor.ids)
        transform_ids = np.array(list(linear_layer.transform_ids.values()))
        cols = len(in_ctensor)
        rows = len(transform_ids) // cols

        # Now we can perform a blocked linear transform
        transform_ids = transform_ids.reshape(rows, cols)
        cts_out = []
        for i in range(rows):
            ct_out = None
            for j in range(cols):
                t_id = transform_ids[i][j]

                if self.io_mode != "none":
                    self.load_rotation_keys(t_id)
                    self.load_plaintext_diagonals(layer_name, i, j, t_id)

                res = self.backend.EvaluateLinearTransform(t_id, in_ctensor.ids[j]) 
                ct = CipherTensor(self.scheme, res, out_shape, fhe_out_shape)

                # Accumulate results across a row of blocks
                ct_out = ct if j == 0 else ct_out + ct
                    
                if self.io_mode != "none":
                    self.remove_rotation_keys()
                    self.remove_plaintext_diagonals(t_id)
            
            # We know the output of this accumulation will just be one ciphertext
            ct_out_rescaled = self.evaluator.rescale(ct_out.ids[0], in_place=False)
            cts_out.append(ct_out_rescaled)

        return CipherTensor(self.scheme, cts_out, out_shape, fhe_out_shape)
            
    def delete_transforms(self, transform_ids: dict):
        for tid in transform_ids.values():
            self.backend.DeleteLinearTransform(tid)

    def _verify_layer_compatibility(self, linear_layer):
        layer_name = linear_layer.name

        # -------- Current network values -------- #

        curr_embed_method = linear_layer.scheme.params.get_embedding_method()
        curr_output_rotations = linear_layer.output_rotations
        curr_on_bias = linear_layer.on_bias
        curr_input_shape = linear_layer.input_shape 
        curr_output_shape = linear_layer.output_shape
        curr_input_min = linear_layer.input_min 
        curr_input_max = linear_layer.input_max
        curr_output_min = linear_layer.output_min
        curr_output_max = linear_layer.output_max

        # ------- Previous network values ------- #

        with h5py.File(self.diags_path, "r") as f:

            # Check if the layer exists in the h5py file
            if layer_name not in f:
                raise ValueError(
                    f"Layer '{layer_name}' not found in file {self.diags_path}. " + 
                    "First set IO mode in parameters YAML file to `save`."
                )
            
            layer = f[layer_name]
            
            last_embed_method = layer["embedding_method"][()].decode("utf-8")
            last_output_rotations = layer["output_rotations"][()]
            last_on_bias = torch.tensor(layer["on_bias"][:])
            last_input_shape = torch.Size(layer["input_shape"][:])
            last_output_shape = torch.Size(layer["output_shape"][:])
            last_input_min = layer["input_min"][()]
            last_input_max = layer["input_max"][()]
            last_output_min = layer["output_min"][()]
            last_output_max = layer["output_max"][()]

            # Check each parameter and collect mismatches
            mismatches = []
                            
            if curr_on_bias.shape != last_on_bias.shape:
                mismatches.append(f"on_bias: shape mismatch")
            elif not torch.allclose(curr_on_bias, last_on_bias):
                mismatches.append(f"on_bias: values mismatch")
            
            # Simple equality checks
            if curr_output_rotations != last_output_rotations:
                mismatches.append(f"output_rotations mismatch")

            if curr_input_shape != last_input_shape:
                mismatches.append(f"input_shape mismatch")
            
            if curr_output_shape != last_output_shape:
                mismatches.append(f"output_shape mismatch")
            
            if curr_embed_method != last_embed_method:
                mismatches.append(f"embedding_method mismatch")
            
            if curr_input_min != last_input_min:
                mismatches.append(f"input_min mismatch")
            
            if curr_input_max != last_input_max:
                mismatches.append(f"input_max mismatch")
            
            if curr_output_min != last_output_min:
                mismatches.append(f"output_min mismatch")
            
            if curr_output_max != last_output_max:
                mismatches.append(f"output_max mismatch")
            
            # If there are mismatches, raise a detailed error
            if mismatches:
                error_msg = "Saved network does not match currently instantiated network: "
                error_msg += ", ".join(mismatches)
                error_msg += ". First set IO mode in parameters YAML file to `save` to "
                error_msg += "override existing data. Then loading will work."
                
                raise ValueError(error_msg)
            
    def save_plaintext_diagonals(self, layer_name, lintransf_id, row, col, diag_idxs):
        with h5py.File(self.diags_path, "a") as f:
            layer = f[layer_name]
            plaintext_group = layer.require_group("plaintexts")
            block_idx = f"{row}_{col}"
            block_group = plaintext_group.create_group(block_idx)

            for diag_idx in diag_idxs:
                diag_serial, diag_ptr = self.backend.SerializeDiagonal(lintransf_id, diag_idx)
                block_group.create_dataset(str(diag_idx), data=diag_serial)

                # Now that it's saved, we'll free the memory
                self.backend.FreeCArray(diag_ptr)

    def load_plaintext_diagonals(self, layer_name, row, col, transform_id):
        with h5py.File(self.diags_path, "r") as f:
            layer = f[layer_name]
            ptxt_group = layer["plaintexts"]
            block = ptxt_group[f"{row}_{col}"]

            for diag_idx in block:
                serial_diag = block[diag_idx][()]
                self.backend.LoadPlaintextDiagonal(
                    serial_diag, transform_id, int(diag_idx)
                )
    
    def load_rotation_keys(self, transform_id):
        keys = self.get_required_rotation_keys(transform_id)

        with h5py.File(self.keys_path, "r") as f:
            for key in keys:
                serial_key = f[str(key)][()]
                self.backend.LoadRotationKey(serial_key, int(key))

    def remove_rotation_keys(self):
        self.backend.RemoveRotationKeys() 

    def remove_plaintext_diagonals(self, transform_id):
        self.backend.RemovePlaintextDiagonals(transform_id)