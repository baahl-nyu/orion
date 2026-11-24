import sys
import math

class PlainTensor:
    def __init__(self, scheme, ptxt_ids, shape, on_shape=None, start=None, stride=None, stop=None, is_complex=False):
        self.scheme = scheme
        self.backend = scheme.backend
        self.encoder = scheme.encoder
        self.evaluator = scheme.evaluator
        
        self.ids = [ptxt_ids] if isinstance(ptxt_ids, int) else ptxt_ids
        self.shape = shape 
        self.on_shape = on_shape or shape
        self.start = start
        self.stride = stride
        self.stop = stop
        self.is_complex = is_complex

    #def __del__(self):
    #    if 'sys' in globals() and sys.modules and self.scheme:
    #        try:
    #            for idx in self.ids:
    #                self.backend.DeletePlaintext(idx)
    #        except Exception: 
    #            pass # avoids errors for GC at program termination

    def __len__(self):
        return len(self.ids)
    
    def __str__(self):
        return str(self.decode())
    
    def mul(self, other, in_place=False):
        if not isinstance(other, CipherTensor):
            raise ValueError(f"Multiplication between PlainTensor and "
                             f"{type(other)} is not supported.")

        mul_ids = []
        for i in range(len(self.ids)):
            mul_id = self.evaluator.mul_plaintext(
                other.ids[i], self.ids[i], in_place)
            mul_ids.append(mul_id)

        if in_place:
            return other
        result_is_complex = self.is_complex or other.is_complex
        return CipherTensor(self.scheme, mul_ids, self.shape, self.on_shape, self.start, self.stride, self.stop, result_is_complex) 

    def __mul__(self, other):
        return self.mul(other, in_place=False)     

    def __imul__(self, other):
        return self.mul(other, in_place=True)
    
    def _check_valid(self, other):
        return 
        
    def get_ids(self):
        return self.ids
    
    def scale(self):
        return self.backend.GetPlaintextScale(self.ids[0])
    
    def set_scale(self, scale):
        for ptxt in self.ids:
            self.backend.SetPlaintextScale(ptxt, scale)

    def level(self):
        return self.backend.GetPlaintextLevel(self.ids[0])
    
    def slots(self):
        return self.backend.GetPlaintextSlots(self.ids[0])
    
    def min(self):
        return self.decode().min()
    
    def max(self):
        return self.decode().max()
    
    def moduli(self):
        return self.backend.GetModuliChain()
    
    def decode(self):
        return self.encoder.decode(self)
    

class CipherTensor:
    def __init__(self, scheme, ctxt_ids, shape, on_shape=None, start=None, stride=None, stop=None, is_complex=False):
        self.scheme = scheme
        self.backend = scheme.backend 
        self.encryptor = scheme.encryptor
        self.evaluator = scheme.evaluator
        self.bootstrapper = scheme.bootstrapper

        self.ids = [ctxt_ids] if isinstance(ctxt_ids, int) else ctxt_ids
        self._track_ids()
        self.shape = shape 
        self.on_shape = on_shape or shape
        self.start = start
        self.stride = stride
        self.stop = stop
        self.is_complex = is_complex

    # Release a backend ciphertext when the last CipherTensor wrapping its
    # id goes away. Ids are reference-counted because several tensors can
    # wrap the same id (a layer's output and the temporary it was built
    # from), so a plain destructor would free ids still in use. Before
    # this, every intermediate ciphertext of every forward stayed allocated
    # for the life of the process (measured on Cheddar: +4.5 GiB per
    # forward for HE-LRM at threshold 5,000,000, batch 2; the same growth
    # in host RSS on Lattigo). Plaintexts are not freed here: layers hold
    # diagonal plaintexts by raw id.
    _id_refs = {}  # ct_id -> number of live CipherTensors wrapping it

    def _track_ids(self):
        for idx in self.ids:
            CipherTensor._id_refs[idx] = CipherTensor._id_refs.get(idx, 0) + 1

    def __del__(self):
        if getattr(self, "scheme", None) is None:
            return
        if 'sys' not in globals() or not sys.modules:
            return
        try:
            for idx in self.ids:
                n = CipherTensor._id_refs.get(idx, 0) - 1
                if n > 0:
                    CipherTensor._id_refs[idx] = n
                else:
                    CipherTensor._id_refs.pop(idx, None)
                    self.backend.DeleteCiphertext(idx)
        except Exception:
            pass  # avoids errors for GC at program termination

    def __len__(self):
        return len(self.ids)
    
    def __str__(self):
        ptxt = self.decrypt()
        return str(ptxt.decode())
    
    #--------------#
    #  Operations  #
    #--------------#
    
    def __neg__(self):
        neg_ids = []
        for ctxt in self.ids:
            neg_id = self.evaluator.negate(ctxt)
            neg_ids.append(neg_id)

        return CipherTensor(self.scheme, neg_ids, self.shape, self.on_shape, self.start, self.stride, self.stop, self.is_complex)
    
    def add(self, other, in_place=False):
        self._check_valid(other)

        add_ids = []
        for i in range(len(self.ids)):
            if isinstance(other, (int, float)):
                add_id = self.evaluator.add_scalar(
                    self.ids[i], other, in_place)
            elif isinstance(other, PlainTensor):
                add_id = self.evaluator.add_plaintext(
                    self.ids[i], other.ids[i], in_place)
            elif isinstance(other, CipherTensor):
                add_id = self.evaluator.add_ciphertext(
                    self.ids[i], other.ids[i], in_place)
            else:
                raise ValueError(f"Addition between CipherTensor and "
                                 f"{type(other)} is not supported.")

            add_ids.append(add_id)

        if in_place:
            return self
        result_is_complex = self.is_complex or getattr(other, "is_complex", False)
        return CipherTensor(self.scheme, add_ids, self.shape, self.on_shape, self.start, self.stride, self.stop, result_is_complex)
    
    def __add__(self, other):
        return self.add(other, in_place=False)
    
    def __radd__(self, other):
        return self.add(other, in_place=False)

    def __iadd__(self, other):
        return self.add(other, in_place=True)
    
    def sub(self, other, in_place=False):
        self._check_valid(other)

        sub_ids = []
        for i in range(len(self.ids)):
            if isinstance(other, (int, float)):
                sub_id = self.evaluator.sub_scalar(
                    self.ids[i], other, in_place)
            elif isinstance(other, PlainTensor):
                sub_id = self.evaluator.sub_plaintext(
                    self.ids[i], other.ids[i], in_place)
            elif isinstance(other, CipherTensor):
                sub_id = self.evaluator.sub_ciphertext(
                    self.ids[i], other.ids[i], in_place)
            else:
                raise ValueError(f"Subtraction between CipherTensor and "
                                 f"{type(other)} is not supported.")

            sub_ids.append(sub_id)

        if in_place:
            return self
        result_is_complex = self.is_complex or getattr(other, "is_complex", False)
        return CipherTensor(self.scheme, sub_ids, self.shape, self.on_shape, self.start, self.stride, self.stop, result_is_complex)
    
    def __sub__(self, other):
        return self.sub(other, in_place=False)

    def __isub__(self, other):
        return self.sub(other, in_place=True)

    def __rsub__(self, other):
        return self.sub(other, in_place=False)
    
    def mul(self, other, in_place=False):
        self._check_valid(other)

        mul_ids = []
        for i in range(len(self.ids)):
            if isinstance(other, (int, float)):
                mul_id = self.evaluator.mul_scalar(
                    self.ids[i], other, in_place)
            elif isinstance(other, PlainTensor):
                mul_id = self.evaluator.mul_plaintext(
                    self.ids[i], other.ids[i], in_place)
            elif isinstance(other, CipherTensor):
                mul_id = self.evaluator.mul_ciphertext(
                    self.ids[i], other.ids[i], in_place)
            else:
                raise ValueError(f"Multiplication between CipherTensor and "
                                 f"{type(other)} is not supported.")
            
            mul_ids.append(mul_id)

        if in_place:
            return self
        result_is_complex = self.is_complex or getattr(other, "is_complex", False)
        return CipherTensor(self.scheme, mul_ids, self.shape, self.on_shape, self.start, self.stride, self.stop, result_is_complex)
    
    def __mul__(self, other):
        return self.mul(other, in_place=False)     

    def __imul__(self, other):
        return self.mul(other, in_place=True)

    def __rmul__(self, other):
        return self.mul(other, in_place=False)
    
    def roll(self, amount, in_place=False):
        rot_ids = []
        for ctxt in self.ids:
            rot_id = self.evaluator.rotate(ctxt, amount, in_place)
            rot_ids.append(rot_id)

        return CipherTensor(self.scheme, rot_ids, self.shape, self.on_shape, self.start, self.stride, self.stop, self.is_complex)
    
    def _check_valid(self, other):
        return
    
    #----------------------
    #
    #---------------------
    
    def scale(self):
        return self.backend.GetCiphertextScale(self.ids[0])
    
    def set_scale(self, scale):
        for ctxt in self.ids:
            self.backend.SetCiphertextScale(ctxt, scale)

    def level(self):
        return self.backend.GetCiphertextLevel(self.ids[0])
    
    def slots(self):
        return self.backend.GetCiphertextSlots(self.ids[0])
    
    def degree(self):
        return self.backend.GetCiphertextDegree(self.ids[0])
    
    def min(self):
        return self.decrypt().min()
    
    def max(self):
        return self.decrypt().max()
    
    def moduli(self):
        return self.backend.GetModuliChain()
    
    def bootstrap(self):
        elements = self.on_shape.numel()
        slots = 2 ** math.ceil(math.log2(elements))
        slots = int(min(self.slots(), slots)) # sparse bootstrapping
        
        btp_ids = []
        for ctxt in self.ids:
            btp_id = self.bootstrapper.bootstrap(ctxt, slots)
            btp_ids.append(btp_id)

        return CipherTensor(self.scheme, btp_ids, self.shape, self.on_shape, self.start, self.stride, self.stop, self.is_complex)
        
    def decrypt(self):
        return self.encryptor.decrypt(self)
