import math
import os

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

        linear_layer.slice_masks = self._build_slice_masks(linear_layer)
        return lintransf_ids

    def _build_slice_masks(self, linear_layer):
        """One 0/1 plaintext per distinct (offset, height) window.

        A block read at a reduced height produces a result that repeats
        every `height` slots, so the copy sitting in its correct place is
        accompanied by period/height - 1 copies squatting on other
        groups' slots. Blocks are summed, so those have to be cleared
        before accumulation or the sum is garbage.

        The mask keeps slots whose position within the layer's natural
        output period falls in [offset, offset+height). Because
        _slice_heights aligns offset to a multiple of height, that
        surviving copy is already in the right place -- masking is the
        whole correction, no placement rotation.

        Groups that were never shrunk get an all-ones mask. It is a
        no-op numerically, but it keeps every group at the same level so
        they can be added; mixing masked and unmasked groups would leave
        them one level apart.
        """
        sm = getattr(linear_layer, "slice_meta", None)
        if not sm:
            return None

        slots = linear_layer.slots or self.scheme.params.get_slots()
        period = sm["period"]
        pos = torch.arange(slots) % period

        # The mask multiplies the transform output *after* its rescale,
        # so the ciphertext sits one level below the transform. Encode at
        # that level with scale = that level's modulus (the same idiom as
        # TableMultConsolidate in nn/embedding.py): the product lands at
        # scale*q, and the rescale that follows divides q back out,
        # returning the ciphertext to the scale everything else is at.
        # Encoding with the default scale instead leaves the branch a
        # whole modulus light and the network fails at the next add.
        mask_level = linear_layer.level - 1
        q = self.scheme.encoder.get_moduli_chain()[mask_level]

        masks = {}
        for key in self._slice_groups(linear_layer, sm):
            if key in masks:
                continue
            if key is None:
                values = torch.ones(slots)
            else:
                r0, h = key
                values = ((pos >= r0) & (pos < r0 + h)).to(torch.float32)
            masks[key] = self.scheme.encoder.encode(values, mask_level, q)

        # The shorter periods need fold distances the uniform path never
        # asked for (slots/2 .. slots/height). Register them up front so
        # backends that do not generate keys lazily still work.
        for key in masks:
            h = period if key is None else key[1]
            for t in range(1, int(math.log2(slots // h)) + 1):
                self.backend.AddRotationKey(slots // (2 ** t))
        return masks

    @staticmethod
    def _slice_groups(linear_layer, sm):
        """Every (offset, height) group this layer evaluates, `None` for
        blocks left on the uniform full-height path."""
        blocks = sm["blocks"]
        seen = []
        num_cols = max(blocks) + 1 if blocks else 0
        for bc in range(max(num_cols, 1)):
            m = blocks.get(bc)
            key = (m[0], m[1]) if m else None
            if key not in seen:
                seen.append(key)
        return seen
    
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

            diags_group = layer.require_group("diagonals", track_order=True)
            for (row, col), diags in diagonals.items():
                block_idx = f"{row}_{col}"
                block_diags_group = diags_group.create_group(block_idx, track_order=True)
                
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

        if getattr(linear_layer, "slice_meta", None):
            return self._evaluate_sliced(
                linear_layer, in_ctensor, transform_ids, rows, cols)

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
            
            # We know the output of this accumulation will just be one
            # ciphertext. Backends whose EvaluateLinearTransform already
            # returns a rescaled result (cheddar's double-hoisted path:
            # the native final ModDown folds the rescale in for free)
            # advertise it via lt_output_prerescaled; rescaling again
            # would wrongly burn a second level.
            if getattr(self.backend, "lt_output_prerescaled", False):
                ct_out_rescaled = ct_out.ids[0]
            else:
                ct_out_rescaled = self.evaluator.rescale(
                    ct_out.ids[0], in_place=False)
            cts_out.append(ct_out_rescaled)

        return CipherTensor(self.scheme, cts_out, out_shape, fhe_out_shape)
            
    def _evaluate_sliced(self, linear_layer, in_ctensor, transform_ids,
                         rows, cols):
        """Blocked linear transform with per-block heights.

        Differs from the uniform path in three ways:

        * blocks are bucketed by their (offset, height) window instead of
          all being summed together, because each window needs its own
          fold depth;
        * the fold runs here, per window, rather than once for the whole
          layer in LinearTransform.evaluate_transforms -- a block read at
          height h replicates slots/h times, not slots/block_height;
        * each window is masked before it joins the accumulation, which
          is what makes the shortened heights correct at all.

        Fold and mask both distribute over addition, so bucketing first
        and folding once per bucket is equivalent to folding each block
        -- and far cheaper, since a layer's many column blocks fall into
        only a handful of distinct windows (18 blocks, 5 windows on
        HE-LRM's sparse layer at threshold 500,000).
        """
        sm = linear_layer.slice_meta
        blocks = sm["blocks"]
        period = sm["period"]
        slots = linear_layer.slots or self.scheme.params.get_slots()
        layer_name = linear_layer.name
        out_shape = linear_layer.output_shape
        fhe_out_shape = linear_layer.fhe_output_shape
        masks = linear_layer.slice_masks

        cts_out = []
        for i in range(rows):
            groups = {}
            for j in range(cols):
                t_id = transform_ids[i][j]
                if self.io_mode != "none":
                    self.load_rotation_keys(t_id)
                    self.load_plaintext_diagonals(layer_name, i, j, t_id)

                res = self.backend.EvaluateLinearTransform(
                    t_id, in_ctensor.ids[j])
                ct = CipherTensor(self.scheme, res, out_shape, fhe_out_shape)

                m = blocks.get(j)
                key = (m[0], m[1]) if m else None
                groups[key] = ct if key not in groups else groups[key] + ct

                if self.io_mode != "none":
                    self.remove_rotation_keys()
                    self.remove_plaintext_diagonals(t_id)

            total = None
            dbg = os.environ.get("ORION_SLICE_DEBUG", "") == "1"
            for key, ct in groups.items():
                height = period if key is None else key[1]

                if dbg:
                    print(f"[slice] {layer_name} key={key} h={height} "
                          f"post-LT scale={self.backend.GetCiphertextScale(ct.ids[0]):.6g} "
                          f"level={self.backend.GetCiphertextLevel(ct.ids[0])}",
                          flush=True)

                # Sum this window's slots/height replicas.
                for t in range(1, int(math.log2(slots // height)) + 1):
                    ct = ct + ct.roll(slots // (2 ** t))

                ct_id = self.evaluator.rescale(ct.ids[0], in_place=False)
                ct = CipherTensor(self.scheme, ct_id, out_shape, fhe_out_shape)

                if dbg:
                    print(f"[slice]   post-rescale scale="
                          f"{self.backend.GetCiphertextScale(ct.ids[0]):.6g} "
                          f"level={self.backend.GetCiphertextLevel(ct.ids[0])} "
                          f"| mask scale={self.backend.GetPlaintextScale(masks[key].ids[0]):.6g} "
                          f"level={self.backend.GetPlaintextLevel(masks[key].ids[0])}",
                          flush=True)

                # Clear the copies that landed on other windows' slots.
                # This is the extra level the reduced heights cost.
                # ct * plaintext already rescales -- this codebase keeps
                # ct-pt multiply mult-then-rescale scale-exact, so the
                # mask consumes the level itself. Rescaling again here
                # drops a second level and divides the scale away, which
                # is what the first attempt did (final scale 0.99 against
                # the dense branch's 2^40).
                ct = ct * masks[key]
                if dbg:
                    print(f"[slice]   post-mask scale="
                          f"{self.backend.GetCiphertextScale(ct.ids[0]):.6g} "
                          f"level={self.backend.GetCiphertextLevel(ct.ids[0])}",
                          flush=True)

                total = ct if total is None else total + ct

            cts_out.append(total.ids[0])

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