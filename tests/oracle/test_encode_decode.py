"""Tests for encode/decode round-trip accuracy."""

import torch
import pytest
from fhe_test_utils import assert_fhe_close


class TestEncodeDecode:
    """Encode values into CKKS plaintexts and decode them back."""

    def test_encode_decode_basic(self, scheme, slots):
        """Encode a short vector, decode, check values match."""
        values = [1.0, 2.0, 3.0, 4.0]
        ptxt = scheme.encode(values)
        result = scheme.decode(ptxt)
        assert_fhe_close(result, values, atol=1e-4, msg="basic encode/decode")

    def test_encode_decode_full_slots(self, scheme, slots):
        """Encode a vector that fills all slots exactly."""
        values = torch.randn(slots)
        ptxt = scheme.encode(values)
        result = scheme.decode(ptxt)
        assert_fhe_close(result, values, atol=1e-4, msg="full-slot encode/decode")

    def test_encode_decode_negative_values(self, scheme):
        """Negative values round-trip correctly."""
        values = [-3.5, -0.001, -100.0, 0.0]
        ptxt = scheme.encode(values)
        result = scheme.decode(ptxt)
        assert_fhe_close(result, values, atol=1e-4, msg="negative values")

    def test_encode_decode_large_values(self, scheme):
        """Large magnitude values."""
        values = [1e4, -1e4, 5e3, -5e3]
        ptxt = scheme.encode(values)
        result = scheme.decode(ptxt)
        assert_fhe_close(result, values, atol=1.0, msg="large values")

    def test_encode_decode_zeros(self, scheme, slots):
        """All-zero vector."""
        values = torch.zeros(slots)
        ptxt = scheme.encode(values)
        result = scheme.decode(ptxt)
        assert_fhe_close(result, values, atol=1e-5, msg="zeros")

    def test_encode_at_specific_level(self, scheme, max_level):
        """Encode at a specific level and verify the plaintext level."""
        values = [1.0, 2.0]
        level = max_level - 2
        ptxt = scheme.encode(values, level=level)
        assert ptxt.level() == level, (
            f"Expected plaintext level {level}, got {ptxt.level()}")
        result = scheme.decode(ptxt)
        assert_fhe_close(result, values, atol=1e-4, msg="specific level")

    def test_encode_decode_tensor_shape_preserved(self, scheme):
        """Output tensor shape matches input tensor shape."""
        values = torch.randn(8)
        ptxt = scheme.encode(values)
        result = scheme.decode(ptxt)
        assert result.shape == values.shape, (
            f"Shape mismatch: {result.shape} vs {values.shape}")

    def test_encode_decode_multi_plaintext(self, scheme, slots):
        """Vector larger than one slot-count produces multiple plaintexts."""
        values = torch.randn(slots * 2)
        ptxt = scheme.encode(values)
        assert len(ptxt) == 2, f"Expected 2 plaintexts, got {len(ptxt)}"
        result = scheme.decode(ptxt)
        assert_fhe_close(result, values, atol=1e-4, msg="multi-plaintext")
