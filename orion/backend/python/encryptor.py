from .tensors import PlainTensor, CipherTensor

class NewEncryptor:
    def __init__(self, scheme):
        self.scheme = scheme
        self.backend = scheme.backend
        self.new_encryptor()
        self.new_decryptor()

    def new_encryptor(self):
        self.backend.NewEncryptor()

    def new_decryptor(self):
        self.backend.NewDecryptor()

    def encrypt(self, plaintensor):
        ciphertext_ids = []
        for ptxt in plaintensor.ids:
            ciphertext_id = self.backend.Encrypt(ptxt)
            ciphertext_ids.append(ciphertext_id)

        return CipherTensor(
            self.scheme, ciphertext_ids, plaintensor.shape, plaintensor.on_shape, plaintensor.start, plaintensor.stride, plaintensor.stop)
    
    def decrypt(self, ciphertensor):
        plaintext_ids = []
        for ctxt in ciphertensor.ids:
            plaintext_id = self.backend.Decrypt(ctxt)
            plaintext_ids.append(plaintext_id)

        return PlainTensor(
           self.scheme,  plaintext_ids, ciphertensor.shape, ciphertensor.on_shape, ciphertensor.start, ciphertensor.stride, ciphertensor.stop
        )
