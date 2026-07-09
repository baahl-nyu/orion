import torch

from orion.core.orion import scheme


CONFIG = {
    "ckks_params": {
        "LogN": 13,
        "LogQ": [45, 30, 30, 30],
        "LogP": [50],
        "LogScale": 30,
        "H": 192,
        "RingType": "Standard",
    },
    "orion": {
        "margin": 2,
        "embedding_method": "hybrid",
        "backend": "lattigo",
        "fuse_modules": True,
        "debug": False,
        "io_mode": "none",
    },
}


def test_complex_encode_decode_round_trip():
    """Encoding/decoding a complex tensor should preserve both lanes."""
    scheme.init_scheme(CONFIG)

    try:
        values = torch.complex(
            torch.arange(8, dtype=torch.float64),
            torch.arange(8, dtype=torch.float64) * -1.0,
        )

        ptxt = scheme.encode(values)
        assert ptxt.is_complex

        decoded = scheme.decode(ptxt)
        assert torch.is_complex(decoded)
        torch.testing.assert_close(decoded, values, atol=1e-3, rtol=1e-3)
    finally:
        scheme.delete_scheme()


def test_complex_encrypt_decrypt_round_trip():
    """Encrypting/decrypting a complex plaintext should preserve both lanes."""
    scheme.init_scheme(CONFIG)

    try:
        real = torch.tensor([1.5, -2.25, 3.0, 0.0], dtype=torch.float64)
        imag = torch.tensor([0.5, 4.0, -1.0, 2.0], dtype=torch.float64)
        values = torch.complex(real, imag)

        ptxt = scheme.encode(values)
        ctxt = scheme.encrypt(ptxt)
        assert ctxt.is_complex

        recovered_ptxt = scheme.decrypt(ctxt)
        decoded = scheme.decode(recovered_ptxt)

        torch.testing.assert_close(decoded, values, atol=1e-2, rtol=1e-2)
    finally:
        scheme.delete_scheme()


def test_complex_ciphertext_times_real_plaintext():
    """A complex-packed ciphertext times a real-only plaintext must scale
    each lane independently (no cross terms), in both operand orders."""
    scheme.init_scheme(CONFIG)

    try:
        real = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        imag = torch.tensor([5.0, 6.0, 7.0, 8.0], dtype=torch.float64)
        matrix = torch.complex(real, imag)
        vector = torch.tensor([2.0, 0.5, -1.0, 3.0], dtype=torch.float64)

        ct_matrix = scheme.encrypt(scheme.encode(matrix))
        pt_vector = scheme.encode(vector)

        expected = matrix * vector

        # ciphertext * plaintext
        ct_result = ct_matrix * pt_vector
        assert ct_result.is_complex
        decoded = scheme.decode(scheme.decrypt(ct_result))
        torch.testing.assert_close(decoded, expected, atol=1e-2, rtol=1e-2)

        # plaintext * ciphertext (commuted order)
        ct_result_rev = pt_vector * ct_matrix
        assert ct_result_rev.is_complex
        decoded_rev = scheme.decode(scheme.decrypt(ct_result_rev))
        torch.testing.assert_close(decoded_rev, expected, atol=1e-2, rtol=1e-2)
    finally:
        scheme.delete_scheme()


def test_complex_ciphertext_times_real_ciphertext():
    """A complex-packed ciphertext times a real-only ciphertext (both
    encrypted) must scale each lane independently, in both operand orders."""
    scheme.init_scheme(CONFIG)

    try:
        real = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        imag = torch.tensor([5.0, 6.0, 7.0, 8.0], dtype=torch.float64)
        matrix = torch.complex(real, imag)
        vector = torch.tensor([2.0, 0.5, -1.0, 3.0], dtype=torch.float64)

        ct_matrix = scheme.encrypt(scheme.encode(matrix))
        ct_vector = scheme.encrypt(scheme.encode(vector))
        assert not ct_vector.is_complex

        expected = matrix * vector

        # complex ciphertext * real ciphertext
        ct_result = ct_matrix * ct_vector
        assert ct_result.is_complex
        decoded = scheme.decode(scheme.decrypt(ct_result))
        torch.testing.assert_close(decoded, expected, atol=1e-2, rtol=1e-2)

        # real ciphertext * complex ciphertext (commuted order)
        ct_result_rev = ct_vector * ct_matrix
        assert ct_result_rev.is_complex
        decoded_rev = scheme.decode(scheme.decrypt(ct_result_rev))
        torch.testing.assert_close(decoded_rev, expected, atol=1e-2, rtol=1e-2)
    finally:
        scheme.delete_scheme()


def test_real_encode_is_unaffected():
    """The existing real-only encode/decode path must stay unchanged."""
    scheme.init_scheme(CONFIG)

    try:
        values = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)

        ptxt = scheme.encode(values)
        assert not ptxt.is_complex

        decoded = scheme.decode(ptxt)
        assert not torch.is_complex(decoded)
        torch.testing.assert_close(decoded, values, atol=1e-3, rtol=1e-3)
    finally:
        scheme.delete_scheme()
