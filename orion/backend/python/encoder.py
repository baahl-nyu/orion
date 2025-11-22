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

    def encode(self, values, level=None, scale=None):
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

        values = torch.tensor(values)[:plaintensor.on_shape.numel()]
        return values.reshape(plaintensor.on_shape)

    def get_moduli_chain(self):
        return self.backend.GetModuliChain()

    def get_aux_moduli_chain(self):
        return self.backend.GetAuxModuliChain()
