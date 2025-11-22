from orion.core.orion import scheme


def test_aux_moduli_chain():
    """
    Test that the GetAuxModuliChain function correctly returns auxiliary primes
    (P primes) with the expected bit sizes as specified in LogP.
    """
    config = {
        "ckks_params": {
            "LogN": 14,
            "LogQ": [45, 30, 30, 30, 30, 45],
            "LogP": [50, 51, 52],  # Nontrivial choice of special primes
            "LogScale": 30,
            "H": 192,
            "RingType": "Standard"
        },
        "orion": {
            "margin": 2,
            "embedding_method": "hybrid",
            "backend": "lattigo",
            "fuse_modules": True,
            "debug": False,
            "io_mode": "none"
        }
    }

    scheme.init_scheme(config)

    try:
        aux_moduli = scheme.backend.GetAuxModuliChain()

        expected_count = len(config["ckks_params"]["LogP"])
        assert len(aux_moduli) == expected_count, (
            f"Expected {expected_count} auxiliary primes, got {len(aux_moduli)}"
        )

        # Verify each prime has the correct bit size
        for i, (prime, expected_logp) in enumerate(zip(aux_moduli, config["ckks_params"]["LogP"])):
            actual_bits = prime.bit_length()

            # In CKKS, primes should have exactly the specified bit size
            # (or be within 1 bit due to prime selection constraints)
            assert actual_bits == expected_logp or actual_bits == expected_logp + 1, (
                f"Auxiliary prime {i} has {actual_bits} bits, expected {expected_logp} "
                f"(prime value: {prime})"
            )

            print(f"Auxiliary prime {i}: {actual_bits} bits (expected {expected_logp})")

        print(f"\nSuccessfully verified {len(aux_moduli)} auxiliary primes")

    finally:
        scheme.delete_scheme()
