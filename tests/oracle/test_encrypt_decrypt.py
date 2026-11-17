"""Tests for encrypt/decrypt round-trip accuracy."""

import torch
import pytest
from fhe_test_utils import assert_fhe_close


class TestEncryptDecrypt:
    """Encrypt plaintexts into ciphertexts and decrypt them back."""

    def test_encrypt_decrypt_basic(self, scheme):
        """Basic encrypt → decrypt → decode round-trip."""
        values = [1.0, 2.0, 3.0, 4.0]
        ptxt = scheme.encode(values)
        ctxt = scheme.encrypt(ptxt)
        ptxt_out = scheme.decrypt(ctxt)
        result = scheme.decode(ptxt_out)
        assert_fhe_close(result, values, atol=1e-2, msg="basic encrypt/decrypt")

    def test_encrypt_decrypt_full_slots(self, scheme, slots):
        """Full-slot vector survives encrypt/decrypt."""
        values = torch.randn(slots)
        ptxt = scheme.encode(values)
        ctxt = scheme.encrypt(ptxt)
        ptxt_out = scheme.decrypt(ctxt)
        result = scheme.decode(ptxt_out)
        assert_fhe_close(result, values, atol=1e-2, msg="full-slot encrypt/decrypt")

    def test_encrypt_decrypt_negative_values(self, scheme):
        """Negative values survive encryption round-trip."""
        values = [-5.0, -0.5, -0.001, 0.0]
        ptxt = scheme.encode(values)
        ctxt = scheme.encrypt(ptxt)
        ptxt_out = scheme.decrypt(ctxt)
        result = scheme.decode(ptxt_out)
        assert_fhe_close(result, values, atol=1e-2, msg="negative encrypt/decrypt")

    def test_ciphertext_level_matches_plaintext(self, scheme, max_level):
        """Ciphertext level should equal the plaintext's encoding level."""
        values = [1.0, 2.0]
        level = max_level - 1
        ptxt = scheme.encode(values, level=level)
        ctxt = scheme.encrypt(ptxt)
        assert ctxt.level() == level, (
            f"Expected ctxt level {level}, got {ctxt.level()}")

    def test_ciphertext_slots(self, scheme, slots):
        """Ciphertext reports the correct number of slots."""
        values = [1.0, 2.0]
        ptxt = scheme.encode(values)
        ctxt = scheme.encrypt(ptxt)
        assert ctxt.slots() == slots, (
            f"Expected {slots} slots, got {ctxt.slots()}")

    def test_encrypt_decrypt_multi_ciphertext(self, scheme, slots):
        """Multi-plaintext tensor encrypts to multi-ciphertext tensor."""
        values = torch.randn(slots * 2)
        ptxt = scheme.encode(values)
        ctxt = scheme.encrypt(ptxt)
        assert len(ctxt) == 2, f"Expected 2 ciphertexts, got {len(ctxt)}"
        ptxt_out = scheme.decrypt(ctxt)
        result = scheme.decode(ptxt_out)
        assert_fhe_close(result, values, atol=1e-2, msg="multi-ciphertext")
