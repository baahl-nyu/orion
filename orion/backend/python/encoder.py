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

        if not level:
            level = self.params.get_max_level()
        if not scale:
            scale = self.params.get_default_scale()

        num_slots = self.params.get_slots()
        num_elements = values.numel()

        values = values.cpu()

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
            vector = torch.zeros(num_elements + pad_length)
            vector[:num_elements] = values.flatten()
            plaintext_id = self.backend.Encode(vector.tolist(), level, scale)
            return PlainTensor(self.scheme, [plaintext_id], values.shape)

        # Original behavior: pad up to a multiple of the scheme's global
        # max slots, splitting into multiple plaintexts if needed.
        pad_length = (-num_elements) % num_slots
        vector = torch.zeros(num_elements + pad_length)
        vector[:num_elements] = values.flatten()
        num_plaintexts = len(vector) // num_slots

        plaintext_ids = []
        for i in range(num_plaintexts):
            to_encode = vector[i*num_slots:(i+1)*num_slots].tolist()
            plaintext_id = self.backend.Encode(to_encode, level, scale)
            plaintext_ids.append(plaintext_id)

        return PlainTensor(self.scheme, plaintext_ids, values.shape)

    def decode(self, plaintensor: PlainTensor):
        values = [] 
        for plaintext_id in plaintensor.ids:
            values.extend(self.backend.Decode(plaintext_id))

        values = torch.tensor(values)
        if plaintensor.start is not None:
            values = values[plaintensor.start:plaintensor.stop:plaintensor.stride]
            return values
        else:
            values = values[:plaintensor.on_shape.numel()]
            return values.reshape(plaintensor.on_shape)

    def get_moduli_chain(self):
        return self.backend.GetModuliChain()
