# Oracle Test Assumptions & Corrections

Assumptions that were wrong when initially writing the tests, and the corrections discovered by running against the Lattigo backend.

## 1. Rotation direction

**Wrong assumption:** Positive `k` in `Rotate(ct, k)` means right-shift (last elements wrap to front).

**Correct behavior:** Positive `k` = **left-shift**. `roll(1)` on `[1, 2, 3, 4]` gives `[2, 3, 4, 1]`. Negative `k` = right-shift.

## 2. Monomial coefficient ordering

**Wrong assumption:** `generate_monomial` takes ascending order `[a0, a1, ..., an]`.

**Correct behavior:** `generate_monomial` takes **descending** order `[an, ..., a1, a0]`. The Python wrapper internally reverses (`coeffs[::-1]`) before passing to Go, which expects ascending. So the user-facing API is highest-degree-first, matching numpy's `poly1d` convention.

Example: `f(x) = 2x + 1` → `generate_monomial([2.0, 1.0])`.
