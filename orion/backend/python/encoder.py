import torch
from .tensors import PlainTensor

class NewEncoder:
    def __init__(self, scheme):
        self.scheme = scheme
        self.params = scheme.params
        self.backend = scheme.backend
        self.setup_encoder()

    def setup_encoder(self):
        self.backend.NewEncoder()

    def encode(self, values, level=None, scale=None, target_slots=None):
        if isinstance(values, list):
            values = torch.tensor(values)
        elif not isinstance(values, torch.Tensor):
            raise TypeError(
                f"Expected 'values' passed to encode() to be a either a list "
                f"or a torch.Tensor, but got {type(values)}.")

        is_complex = torch.is_complex(values)

        # `is None`, not falsiness: level 0 is a legitimate request (the
        # bottom of the chain, e.g. a deliberately low-level upload that
        # the server bootstraps back up) and `not level` silently
        # promoted it to the top of the chain instead.
        if level is None:
            level = self.params.get_max_level()
        if not scale:
            scale = self.params.get_default_scale()

        num_slots = self.params.get_slots()
        num_elements = values.numel()

        values = values.cpu()
        dtype = torch.complex128 if is_complex else torch.float64

        # Narrow single-plaintext encode targeting an explicit physical
        # width. Callers pass this only when they know the exact width
        # the resulting ciphertext must match -- the network's own input
        # (Scheme.encode, via compiled_input_slots) or a layer's own
        # bias/prescale plaintext (self.slots/self.physical_slots) -- see
        # packing.propagate_physical_slots for why an unmatched width
        # causes silently wrong rotation wraparound rather than a crash.
        # Every other call site (masks, embeddings, etc.) is untouched:
        # they never pass target_slots, so behavior there is unchanged.
        if (target_slots is not None and num_elements <= target_slots
                and target_slots <= num_slots):
            pad_length = target_slots - num_elements
            vector = torch.zeros(num_elements + pad_length, dtype=dtype)
            vector[:num_elements] = values.flatten()
            if is_complex:
                interleaved = torch.view_as_real(vector).flatten().tolist()
                plaintext_id = self.backend.EncodeComplex(interleaved, level, scale)
            else:
                plaintext_id = self.backend.Encode(vector.tolist(), level, scale)
            return PlainTensor(
                self.scheme, [plaintext_id], values.shape, is_complex=is_complex)

        # Original behavior: pad up to a multiple of the scheme's global
        # max slots, splitting into multiple plaintexts if needed.
        pad_length = (-num_elements) % num_slots
        vector = torch.zeros(num_elements + pad_length, dtype=dtype)
        vector[:num_elements] = values.flatten()
        num_plaintexts = len(vector) // num_slots

        plaintext_ids = []
        for i in range(num_plaintexts):
            chunk = vector[i*num_slots:(i+1)*num_slots]
            if is_complex:
                # Interleave as [re0, im0, re1, im1, ...] to fill both lanes.
                interleaved = torch.view_as_real(chunk).flatten().tolist()
                plaintext_id = self.backend.EncodeComplex(interleaved, level, scale)
            else:
                plaintext_id = self.backend.Encode(chunk.tolist(), level, scale)
            plaintext_ids.append(plaintext_id)

        return PlainTensor(
            self.scheme, plaintext_ids, values.shape, is_complex=is_complex)

    def decode(self, plaintensor: PlainTensor):
        is_complex = getattr(plaintensor, "is_complex", False)
        values = []
        for plaintext_id in plaintensor.ids:
            if is_complex:
                interleaved = torch.tensor(self.backend.DecodeComplex(plaintext_id))
                values.extend(torch.view_as_complex(
                    interleaved.view(-1, 2).contiguous()).tolist())
            else:
                values.extend(self.backend.Decode(plaintext_id))

        values = torch.tensor(values, dtype=torch.complex128 if is_complex else torch.float64)
        if plaintensor.start is not None:
            values = values[plaintensor.start:plaintensor.stop:plaintensor.stride]
            return values
        else:
            values = values[:plaintensor.on_shape.numel()]
            return values.reshape(plaintensor.on_shape)

    def get_moduli_chain(self):
        return self.backend.GetModuliChain()
