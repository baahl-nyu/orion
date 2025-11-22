import os 
import ctypes
import platform

import torch
import numpy as np


class LattigoFunction:
    """Helper to wrap ctypes functions with argument and return types."""
    def __init__(self, func, argtypes, restype):
        self.func = func
        self.func.argtypes = argtypes 
        self.func.restype = restype

    def __call__(self, *args):
        c_args = []
        for arg in args:
            curr_argtype = self.func.argtypes[len(c_args)]
            c_arg = self.convert_to_ctypes(arg, curr_argtype)
            if isinstance(c_arg, tuple):
                c_args.extend(c_arg)
            else:
                c_args.append(c_arg)
                
        c_result = self.func(*c_args)
        py_result = self.convert_from_ctypes(c_result)
        
        # If the result is a list, then we'll need to manually free the
        # memory we allocated for this list in Go with the below. We'll
        # defer freeing byte data (from serialization) until after that
        # data has been saved to HDF5.
        if isinstance(py_result, list):
            LattigoFunction.FreeCArray(
                ctypes.cast(c_result.Data, ctypes.c_void_p))

        return py_result

    @torch._dynamo.disable
    def convert_to_ctypes(self, arg, typ):
        if isinstance(arg, int) and typ == ctypes.c_int:
            return ctypes.c_int(arg)
        elif isinstance(arg, int) and typ == ctypes.c_ulong:
            return ctypes.c_ulong(arg)
        elif isinstance(arg, float):
            return ctypes.c_float(arg)
        elif isinstance(arg, str):
            return arg.encode('utf-8')
        elif (isinstance(arg, np.ndarray) and 
            arg.dtype == np.uint8 and 
            typ == ctypes.POINTER(ctypes.c_ubyte)):
            ptr = arg.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
            return (ptr, len(arg))
        elif isinstance(arg, list):
            if typ == ctypes.POINTER(ctypes.c_int):
                return ((ctypes.c_int * len(arg))(*arg), len(arg))
            elif typ == ctypes.POINTER(ctypes.c_float):
                return ((ctypes.c_float * len(arg))(*arg), len(arg))
            elif typ == ctypes.POINTER(ctypes.c_ulong):
                return ((ctypes.c_ulong * len(arg))(*arg), len(arg))
            elif typ == ctypes.POINTER(ctypes.c_ubyte):
                return ((ctypes.c_ubyte * len(arg))(*arg), len(arg))
            else:
                raise ValueError("Unexpected list type to convert.")
        else:
            return arg
            
    def convert_from_ctypes(self, res):
        if type(res) == ctypes.c_int:
            return int(res)
        elif type(res) == ctypes.c_float:
            return float(res)
        elif type(res) == ArrayResultFloat:
            return [float(res.Data[i]) for i in range(res.Length)]
        elif type(res) in (ArrayResultInt, ArrayResultUInt64):
            return [int(res.Data[i]) for i in range(res.Length)]
        elif type(res) == ArrayResultDouble:
            return [float(res.Data[i]) for i in range(res.Length)]
        elif type(res) == ArrayResultByte:
            # Create numpy array directly from the C buffer
            buffer = ctypes.cast(
                res.Data, 
                ctypes.POINTER(ctypes.c_ubyte * res.Length)
            ).contents
            array = np.frombuffer(buffer, dtype=np.uint8)
            return array, res.Data
        else:
            return res


class LattigoLibrary:
    """A class to manage loading and interfacing with Lattigo."""
    def __init__(self):
        self.lib = self._load_library()

    def _load_library(self):
        try:
            # Determine library name based on platform
            if platform.system() == "Linux":
                lib_name = "lattigo-linux.so"
            elif platform.system() == "Darwin":  # macOS
                if platform.machine().lower() in ("arm64", "aarch64"):
                    lib_name = "lattigo-mac-arm64.dylib"
                else:
                    lib_name = "lattigo-mac.dylib"
            elif platform.system() == "Windows":
                lib_name = "lattigo-windows.dll"
            else:
                raise RuntimeError("Unsupported platform")
                        
            # Standard path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            lib_path = os.path.join(current_dir, lib_name)            
            return ctypes.CDLL(lib_path)
            
        except OSError as e:
            raise RuntimeError(f"Failed to load Lattigo library: {e}")

    def _find_library(self, root_dir, lib_name):
        """Recursively search for the library file"""
        for root, _, files in os.walk(root_dir):
            if lib_name in files:
                return os.path.join(root, lib_name)
        raise FileNotFoundError(f"Library {lib_name} not found in {root_dir}")
                    
    def setup_bindings(self, orion_params):
        """
        Declares the functions from the Lattigo shared library and sets their
        argument and return types.
        """
        self.setup_scheme(orion_params)
        self.setup_tensor_binds()
        self.setup_key_generator()
        self.setup_encoder()
        self.setup_encryptor()
        self.setup_evaluator()
        self.setup_poly_evaluator()
        self.setup_lt_evaluator()
        self.setup_bootstrapper()

    def setup_scheme(self, orion_params):
        self.NewScheme = LattigoFunction(
            self.lib.NewScheme,
            argtypes=[
                ctypes.c_int, 
                ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_char_p,
                ctypes.c_char_p,
            ],
            restype=None
        )

        self.DeleteScheme = LattigoFunction(
            self.lib.DeleteScheme,
            argtypes=None,
            restype=None
        )

        self.FreeCArray = LattigoFunction(
            self.lib.FreeCArray,
            argtypes=[ctypes.c_void_p],
            restype=None
        )
        LattigoFunction.FreeCArray = self.FreeCArray

        logn = orion_params.get_logn()
        logq = orion_params.get_logq()
        logp = orion_params.get_logp()
        logscale = orion_params.get_logscale()
        h = orion_params.get_hamming_weight()
        ringtype = orion_params.get_ringtype()
        keys_path = orion_params.get_keys_path()
        io_mode = orion_params.get_io_mode()

        self.NewScheme(logn, logq, logp, logscale, h, ringtype, keys_path, io_mode)

    def setup_tensor_binds(self):
        self.DeletePlaintext = LattigoFunction(
            self.lib.DeletePlaintext,
            argtypes=[ctypes.c_int],
            restype=None
        )

        self.DeleteCiphertext = LattigoFunction(
            self.lib.DeleteCiphertext,
            argtypes=[ctypes.c_int],
            restype=None
        )

        self.GetPlaintextScale = LattigoFunction(
            self.lib.GetPlaintextScale,
            argtypes=[ctypes.c_int],
            restype=ctypes.c_ulong
        )

        self.GetCiphertextScale = LattigoFunction(
            self.lib.GetCiphertextScale,
            argtypes=[ctypes.c_int],
            restype=ctypes.c_ulong
        )

        self.SetPlaintextScale = LattigoFunction(
            self.lib.SetPlaintextScale,
            argtypes=[
                ctypes.c_int,
                ctypes.c_ulong,
            ],
            restype=None
        )

        self.SetCiphertextScale = LattigoFunction(
            self.lib.SetCiphertextScale,
            argtypes=[
                ctypes.c_int,
                ctypes.c_ulong,
            ],
            restype=None
        )

        self.GetPlaintextLevel = LattigoFunction(
            self.lib.GetPlaintextLevel,
            argtypes=[ctypes.c_int],
            restype=ctypes.c_int
        )
        
        self.GetCiphertextLevel = LattigoFunction(
            self.lib.GetCiphertextLevel,
            argtypes=[ctypes.c_int],
            restype=ctypes.c_int
        )

        self.GetPlaintextSlots = LattigoFunction(
            self.lib.GetPlaintextSlots,
            argtypes=[ctypes.c_int],
            restype=ctypes.c_int
        )
        
        self.GetCiphertextSlots = LattigoFunction(
            self.lib.GetCiphertextSlots,
            argtypes=[ctypes.c_int],
            restype=ctypes.c_int
        )

        self.GetCiphertextDegree = LattigoFunction(
            self.lib.GetCiphertextDegree,
            argtypes=[ctypes.c_int],
            restype=ctypes.c_int
        )

        self.GetModuliChain = LattigoFunction(
            self.lib.GetModuliChain,
            argtypes=None,
            restype=ArrayResultUInt64,
        )

        self.GetAuxModuliChain = LattigoFunction(
            self.lib.GetAuxModuliChain,
            argtypes=None,
            restype=ArrayResultUInt64,
        )

        self.GetLivePlaintexts = LattigoFunction(
            self.lib.GetLivePlaintexts,
            argtypes=None,
            restype=ArrayResultInt
        )

        self.GetLiveCiphertexts = LattigoFunction(
            self.lib.GetLiveCiphertexts,
            argtypes=None,
            restype=ArrayResultInt
        )

    def setup_key_generator(self):
        self.NewKeyGenerator = LattigoFunction(
            self.lib.NewKeyGenerator,
            argtypes=[],
            restype=None
        )

        self.GenerateSecretKey = LattigoFunction(
            self.lib.GenerateSecretKey,
            argtypes=[], 
            restype=None
        )

        self.GeneratePublicKey = LattigoFunction(
            self.lib.GeneratePublicKey,
            argtypes=[], 
            restype=None
        )

        self.GenerateRelinearizationKey = LattigoFunction(
            self.lib.GenerateRelinearizationKey,
            argtypes=[], 
            restype=None
        )

        self.GenerateEvaluationKeys = LattigoFunction(
            self.lib.GenerateEvaluationKeys,
            argtypes=[], 
            restype=None
        )

        self.SerializeSecretKey = LattigoFunction(
            self.lib.SerializeSecretKey,
            argtypes=[],
            restype=ArrayResultByte
        )

        self.LoadSecretKey = LattigoFunction(
            self.lib.LoadSecretKey,
            argtypes=[ctypes.POINTER(ctypes.c_ubyte), ctypes.c_ulong],
            restype=None
        )

    def setup_encoder(self):
        self.NewEncoder = LattigoFunction(
            self.lib.NewEncoder,
            argtypes=[],
            restype=None
        )

        self.Encode = LattigoFunction(
            self.lib.Encode,
            argtypes=[
                ctypes.POINTER(ctypes.c_float), ctypes.c_int,
                ctypes.c_int,
                ctypes.c_ulong,
            ],
            restype=ctypes.c_int
        )
        self.Decode = LattigoFunction(
            self.lib.Decode,
            argtypes=[ctypes.c_int],
            restype=ArrayResultFloat,
        )

    def setup_encryptor(self):
        self.NewEncryptor = LattigoFunction(
            self.lib.NewEncryptor,
            argtypes=[],
            restype=None
        )

        self.NewDecryptor = LattigoFunction(
            self.lib.NewDecryptor,
            argtypes=[],
            restype=None
        )

        self.Encrypt = LattigoFunction(
            self.lib.Encrypt,
            argtypes=[ctypes.c_int],
            restype=ctypes.c_int
        )
        self.Decrypt = LattigoFunction(
            self.lib.Decrypt,
            argtypes=[ctypes.c_int],
            restype=ctypes.c_int
        )

    def setup_evaluator(self):
        self.NewEvaluator = LattigoFunction(
            self.lib.NewEvaluator,
            argtypes=[],
            restype=None
        )

        self.AddRotationKey = LattigoFunction(
            self.lib.AddRotationKey,
            argtypes=[ctypes.c_int],
            restype=None
        )

        self.Negate = LattigoFunction(
            self.lib.Negate,
            argtypes=[ctypes.c_int],
            restype=ctypes.c_int
        )

        self.Rotate = LattigoFunction(
            self.lib.Rotate,
            argtypes=[
                ctypes.c_int,
                ctypes.c_int
            ],
            restype=ctypes.c_int
        )

        self.RotateNew = LattigoFunction(
            self.lib.RotateNew,
            argtypes=[
                ctypes.c_int,
                ctypes.c_int
            ],
            restype=ctypes.c_int
        )       

        self.Rescale = LattigoFunction(
            self.lib.Rescale,
            argtypes=[ctypes.c_int],
            restype=ctypes.c_int
        )

        self.RescaleNew = LattigoFunction(
            self.lib.RescaleNew,
            argtypes=[ctypes.c_int],
            restype=ctypes.c_int
        )

        self.AddScalar = LattigoFunction(
            self.lib.AddScalar,
            argtypes=[
                ctypes.c_int,
                ctypes.c_float
            ],
            restype=ctypes.c_int
        )

        self.AddScalarNew = LattigoFunction(
            self.lib.AddScalarNew,
            argtypes=[
                ctypes.c_int,
                ctypes.c_float
            ],
            restype=ctypes.c_int
        )

        self.SubScalar = LattigoFunction(
            self.lib.SubScalar,
            argtypes=[
                ctypes.c_int,
                ctypes.c_float
            ],
            restype=ctypes.c_int
        )

        self.SubScalarNew = LattigoFunction(
            self.lib.SubScalarNew,
            argtypes=[
                ctypes.c_int,
                ctypes.c_float
            ],
            restype=ctypes.c_int
        )

        self.MulScalarInt = LattigoFunction(
            self.lib.MulScalarInt,
            argtypes=[
                ctypes.c_int,
                ctypes.c_int
            ],
            restype=ctypes.c_int
        )

        self.MulScalarIntNew = LattigoFunction(
          self.lib.MulScalarIntNew,
            argtypes=[
                ctypes.c_int,
                ctypes.c_int
            ],
            restype=ctypes.c_int
        )

        self.MulScalarFloat = LattigoFunction(
            self.lib.MulScalarFloat,
            argtypes=[
                ctypes.c_int,
                ctypes.c_float
            ],
            restype=ctypes.c_int
        )

        self.MulScalarFloatNew = LattigoFunction(
          self.lib.MulScalarFloatNew,
            argtypes=[
                ctypes.c_int,
                ctypes.c_float
            ],
            restype=ctypes.c_int
        )

        self.AddPlaintext = LattigoFunction(
          self.lib.AddPlaintext,
            argtypes=[
                ctypes.c_int,
                ctypes.c_int
            ],
            restype=ctypes.c_int
        )

        self.AddPlaintextNew = LattigoFunction(
          self.lib.AddPlaintextNew,
            argtypes=[
                ctypes.c_int,
                ctypes.c_int
            ],
            restype=ctypes.c_int
        )

        self.SubPlaintext = LattigoFunction(
          self.lib.SubPlaintext,
            argtypes=[
                ctypes.c_int,
                ctypes.c_int
            ],
            restype=ctypes.c_int
        )

        self.SubPlaintextNew = LattigoFunction(
          self.lib.SubPlaintextNew,
            argtypes=[
                ctypes.c_int,
                ctypes.c_int
            ],
            restype=ctypes.c_int
        )

        self.MulPlaintext = LattigoFunction(
          self.lib.MulPlaintext,
            argtypes=[
                ctypes.c_int,
                ctypes.c_int
            ],
            restype=ctypes.c_int
        )

        self.MulPlaintextNew = LattigoFunction(
          self.lib.MulPlaintextNew,
            argtypes=[
                ctypes.c_int,
                ctypes.c_int
            ],
            restype=ctypes.c_int
        )

        self.AddCiphertext = LattigoFunction(
          self.lib.AddCiphertext,
            argtypes=[
                ctypes.c_int,
                ctypes.c_int
            ],
            restype=ctypes.c_int
        )

        self.AddCiphertextNew = LattigoFunction(
          self.lib.AddCiphertextNew,
            argtypes=[
                ctypes.c_int,
                ctypes.c_int
            ],
            restype=ctypes.c_int
        )

        self.SubCiphertext = LattigoFunction(
          self.lib.SubCiphertext,
            argtypes=[
                ctypes.c_int,
                ctypes.c_int
            ],
            restype=ctypes.c_int
        )

        self.SubCiphertextNew = LattigoFunction(
          self.lib.SubCiphertextNew,
            argtypes=[
                ctypes.c_int,
                ctypes.c_int
            ],
            restype=ctypes.c_int
        )

        self.MulRelinCiphertext = LattigoFunction(
          self.lib.MulRelinCiphertext,
            argtypes=[
                ctypes.c_int,
                ctypes.c_int
            ],
            restype=ctypes.c_int
        )

        self.MulRelinCiphertextNew = LattigoFunction(
          self.lib.MulRelinCiphertextNew,
            argtypes=[
                ctypes.c_int,
                ctypes.c_int
            ],
            restype=ctypes.c_int
        )

    def setup_poly_evaluator(self):
        self.NewPolynomialEvaluator = LattigoFunction(
            self.lib.NewPolynomialEvaluator,
            argtypes=[],
            restype=None
        )

        self.GenerateMonomial = LattigoFunction(
            self.lib.GenerateMonomial,
            argtypes=[ctypes.POINTER(ctypes.c_float), ctypes.c_int],
            restype=ctypes.c_int
        )

        self.GenerateChebyshev = LattigoFunction(
            self.lib.GenerateChebyshev,
            argtypes=[ctypes.POINTER(ctypes.c_float), ctypes.c_int],
            restype=ctypes.c_int
        )

        self.EvaluatePolynomial = LattigoFunction(
            self.lib.EvaluatePolynomial,
            argtypes=[
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_ulong,
            ],
            restype=ctypes.c_int
        )

        self.GenerateMinimaxSignCoeffs = LattigoFunction(
            self.lib.GenerateMinimaxSignCoeffs,
            argtypes=[
                ctypes.POINTER(ctypes.c_int), ctypes.c_int, # degrees
                ctypes.c_int, # prec 
                ctypes.c_int, # logalpha
                ctypes.c_int, # logerr
                ctypes.c_int, # debug
            ],
            restype=ArrayResultDouble
        )

    def setup_lt_evaluator(self):
        self.NewLinearTransformEvaluator = LattigoFunction(
            self.lib.NewLinearTransformEvaluator,
            argtypes=[],
            restype=None
        )

        self.GenerateLinearTransform = LattigoFunction(
            self.lib.GenerateLinearTransform,
            argtypes=[
                ctypes.POINTER(ctypes.c_int), ctypes.c_int, # diags_idxs
                ctypes.POINTER(ctypes.c_float), ctypes.c_int, # diags_data
                ctypes.c_int, # level
                ctypes.c_float, # bsgs_ratio
                ctypes.c_char_p, # io_mode
            ],
            restype=ctypes.c_int
        )

        self.EvaluateLinearTransform = LattigoFunction(
            self.lib.EvaluateLinearTransform,
            argtypes=[
                ctypes.c_int, # transform ID
                ctypes.c_int, # ctxt ID
            ],
            restype=ctypes.c_int
        )

        self.DeleteLinearTransform = LattigoFunction(
            self.lib.DeleteLinearTransform,
            argtypes=[ctypes.c_int],
            restype=None
        )

        self.GetLinearTransformRotationKeys = LattigoFunction(
            self.lib.GetLinearTransformRotationKeys,
            argtypes=[ctypes.c_int],
            restype=ArrayResultInt
        )

        self.GenerateLinearTransformRotationKey = LattigoFunction(
            self.lib.GenerateLinearTransformRotationKey,
            argtypes=[ctypes.c_int],
            restype=None
        )

        self.GenerateAndSerializeRotationKey = LattigoFunction(
            self.lib.GenerateAndSerializeRotationKey,
            argtypes=[ctypes.c_int],
            restype=ArrayResultByte
        )
        
        self.LoadRotationKey = LattigoFunction(
            self.lib.LoadRotationKey,
            argtypes=[
                ctypes.POINTER(ctypes.c_ubyte), ctypes.c_ulong,
                ctypes.c_ulong,
            ],
            restype=None
        )

        self.SerializeDiagonal = LattigoFunction(
            self.lib.SerializeDiagonal,
            argtypes=[
                ctypes.c_int, # transform id
                ctypes.c_int, # diag index
            ],
            restype=ArrayResultByte
        )

        self.LoadPlaintextDiagonal = LattigoFunction(
            self.lib.LoadPlaintextDiagonal,
            argtypes=[
                ctypes.POINTER(ctypes.c_ubyte), ctypes.c_ulong,
                ctypes.c_int,
                ctypes.c_ulong,
            ],
            restype=None
        )

        self.RemovePlaintextDiagonals = LattigoFunction(
            self.lib.RemovePlaintextDiagonals,
            argtypes=[ctypes.c_int],
            restype=None
        )

        self.RemoveRotationKeys = LattigoFunction(
            self.lib.RemoveRotationKeys,
            argtypes=[],
            restype=None,
        )

    def setup_bootstrapper(self):
        self.NewBootstrapper = LattigoFunction(
            self.lib.NewBootstrapper,
            argtypes=[
                ctypes.POINTER(ctypes.c_int), ctypes.c_int, # logPs
                ctypes.c_int, # slots
            ], 
            restype=None
        )

        self.Bootstrap = LattigoFunction(
            self.lib.Bootstrap,
            argtypes=[
                ctypes.c_int,
                ctypes.c_int,
            ],
            restype=ctypes.c_int
        )

        self.DeleteBootstrappers = LattigoFunction(
            self.lib.DeleteBootstrappers,
            argtypes=None,
            restype=None
        )


class ArrayResultInt(ctypes.Structure):
    _fields_ = [("Data", ctypes.POINTER(ctypes.c_int)), ("Length", ctypes.c_ulong)]

class ArrayResultFloat(ctypes.Structure):
    _fields_ = [("Data", ctypes.POINTER(ctypes.c_float)), ("Length", ctypes.c_ulong)]

class ArrayResultDouble(ctypes.Structure):
    _fields_ = [("Data", ctypes.POINTER(ctypes.c_double)), ("Length", ctypes.c_ulong)]

class ArrayResultUInt64(ctypes.Structure):
    _fields_ = [("Data", ctypes.POINTER(ctypes.c_ulong)), ("Length", ctypes.c_ulong)]

class ArrayResultByte(ctypes.Structure):
    _fields_ = [("Data", ctypes.POINTER(ctypes.c_char)), ("Length", ctypes.c_ulong)]