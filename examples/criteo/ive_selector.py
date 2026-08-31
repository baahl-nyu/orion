"""IVE selector (Cheon/Jang/Kang/Rhee, arXiv:2606.03191) on Cheddar.

A faithful implementation of the paper's Algorithm 1 (Independent
Vector Evaluation) and the PCMM lookup that follows it, written against
the same Cheddar bindings HE-LRM uses so the two selectors can be timed
on one machine, one build, one clock.

Deliberately *not* using any of this branch's upload optimizations
beyond the ones IVE's own protocol specifies (Figure 1: seeded `a`,
level-0 encryption).

What IVE replaces
-----------------
HE-LRM's client sends a one-hot selector: p slots per index, and the
server pays nothing to use it. IVE's client sends one *scalar* per
index -- a point on the complex unit circle -- and the server derives a
p-coordinate vector from it homomorphically.

The vector is not the one-hot e_j. It is

    v(a) = sqrt(2/p) * [Re(a), .., Re(a^(p/2)),
                        Im(a), .., Im(a^(p/2 - 1)), 1/sqrt(2)]

evaluated at a_j = exp(2*pi*i * J_j), J_j = (-1)^j * (2j+1) / (4p).
These v_j are linearly independent but not orthogonal to the standard
basis, so the lookup cannot use M directly. The paper's Section 4.1
gives an orthogonal D with D v_j = e_j scaled appropriately; the server
instead multiplies by M @ L where L = D^-1, folded into the plaintext
table offline. That is what makes the whole thing free of any extra
online linear algebra: the basis change costs the server nothing at
inference time, only a one-time offline table transform.

Resource cost per VecGen call, from their Table 2, reproduced exactly
by `vecgen()` below:

    #Add = p - 1     #ptMult = p/2 - 1    #Mult = p/2 - 1
    #Conj = p/2      #PCMM = 1            depth = log p

Batching
--------
Algorithm 1 is written for a batch: the input ciphertext carries one
a_j per slot, so a ring-degree-N ciphertext processes N/2 independent
queries in one pass and the whole cost amortizes across them. At batch
1 the same work runs with a single live slot. Both are measured; the
gap between them is the entire IVE story.
"""

import math

import numpy as np


# --------------------------------------------------------------------
# Plaintext side: the vectors, the basis change, and the table transform
# --------------------------------------------------------------------

def alpha(j, p):
    """The scalar the client sends for index j (Section 4.2)."""
    return np.exp(2j * np.pi * ((-1) ** j) * (2 * j + 1) / (4 * p))


def v_of_alpha(a, p):
    """v(alpha) from equation (2): p real coordinates from one scalar."""
    n = p // 2
    powers = np.array([a ** k for k in range(1, n + 1)])
    return math.sqrt(2.0 / p) * np.concatenate(
        [powers.real, powers[:n - 1].imag, [1.0 / math.sqrt(2.0)]])


def basis_matrix(p):
    """D from Section 4.1, with D[:, j] = v_j. Orthogonal (their App. B).

    Built from its definition rather than by inverting a Vandermonde:
    the paper gives D in closed form and the closed form is what the
    orthogonality proof applies to.
    """
    D = np.zeros((p, p))
    for j in range(p):
        D[:, j] = v_of_alpha(alpha(j, p), p)
    return D


def transformed_table(M, p):
    """M @ L / sqrt(2p), the plaintext the server holds (Figure 1).

    M is (p, d) -- p table rows of dimension d. L = D^-1, and the
    sqrt(2p) divides out the constant factor Algorithm 1 leaves in its
    output (their note above Algorithm 1: the factor is absorbed here to
    save a level).
    """
    D = basis_matrix(p)
    L = np.linalg.inv(D)
    return (M.T @ L / math.sqrt(2.0 * p)).T


def folded_table(M, p):
    """The same lookup, refolded onto the powers so v is never built.

    Algorithm 1 materialises all p coordinates of v(alpha) and the PCMM
    then contracts them against M@L. But those coordinates are only
    Re(alpha^k) and Im(alpha^k), and both are linear in alpha^k and its
    conjugate, so the contraction can be applied to the *powers* and the
    Re/Im split deferred past it. Writing A = (M@L)[0:n], B = (M@L)[n:p-1]
    padded to length n, and z_k = alpha^(k+1):

        out_r = S_r + conj(S_r) + const_r,   S_r = sum_k (A - i B)[k,r] z_k

    and S_r splits into two real-constant accumulations, S_r = P_r + i Q_r,
    because multiplying a ciphertext by i is key-free.

    The multiply count is unchanged -- 2d accumulations over p/2 powers is
    the same d*p ciphertext-constant products as d accumulations over p
    coordinates. What changes is that Algorithm 1's p/2 conjugations
    become d, its p-1 adds become 3d, and only the p/2 powers are ever
    live instead of p/2 powers plus p coordinates.

    **This is not the paper's Algorithm 1.** Their Table 2 books
    #Conj = p/2, so they do materialise v. This is our refactoring of
    their scheme, and is labelled as such wherever it is measured.

    Returns (W, const): W is (2d, p/2) with row 2r the weights for P_r
    and row 2r+1 those for Q_r; const is (d,), the contribution of v's
    constant last coordinate.
    """
    ML = transformed_table(M, p)
    n, d = p // 2, ML.shape[1]
    W = np.zeros((2 * d, n))
    for r in range(d):
        W[2 * r] = ML[0:n, r]
        W[2 * r + 1, :n - 1] = -ML[n:p - 1, r]
    return W, ML[p - 1, :] * math.sqrt(2.0)


# --------------------------------------------------------------------
# Ciphertext side: Algorithm 1
# --------------------------------------------------------------------

class Counter:
    """Operation tally, checked against the paper's Table 2."""

    def __init__(self):
        self.add = self.ptmult = self.mult = self.conj = self.pcmm = 0

    def as_dict(self):
        return dict(add=self.add, ptmult=self.ptmult, mult=self.mult,
                    conj=self.conj, pcmm=self.pcmm)

    def expected(self, p):
        return dict(add=p - 1, ptmult=p // 2 - 1, mult=p // 2 - 1,
                    conj=p // 2, pcmm=self.pcmm)


def make_const(backend, like_ct):
    """Enc(sqrt(2)) at another ciphertext's level and scale (line 16)."""
    lvl = backend.GetCiphertextLevel(like_ct)
    scale = backend.GetCiphertextScale(like_ct)
    n_slots = backend.GetCiphertextSlots(like_ct)
    pt = backend.Encode([math.sqrt(2.0)] * n_slots, lvl, scale)
    ct = backend.Encrypt(pt)
    backend.DeletePlaintext(pt)
    return ct


def power_tree(backend, ct_id, p, counter=None, last_stage=True):
    """Algorithm 1's lines 3-9 on their own: alpha^1 .. alpha^(p/2).

    Split out because both selector variants need exactly this and
    nothing else in common. Returns (prime, owned, counter): `prime[k]`
    encrypts alpha^(k+1) -- `prime[0]` is `ct_id` itself, not a copy --
    and `owned` holds the level-alignment temporaries the caller must
    free.

    With `last_stage=False` the tree stops one stage early, having built
    alpha^1 .. alpha^(n/2) -- exactly the powers that are ever used as
    multiplicands. See `top_powers`.
    """
    c = counter or Counter()
    n = p // 2
    limit = n if last_stage else n // 2
    owned = []

    # Powers of alpha by a multiplication tree.
    # prime[k] encrypts alpha^(k+1). The tree schedule is the paper's:
    # at stage d the anchor is alpha^i for i = 2^d, multiplied into the
    # t = min(i, n - i) powers already built.
    prime = [None] * n
    prime[0] = ct_id
    d = 0
    while (1 << d) < limit:
        i = 1 << d
        t = min(i, n - i)
        for j in range(1, t + 1):
            a, b = prime[i - 1], prime[j - 1]
            # CKKS multiplies only at a common level; the tree's two
            # operands sit at different depths whenever i != j, so the
            # higher one walks down. This is the paper's #ptMult column
            # -- LevelDown is a Mult+Rescale per level in Cheddar.
            la, lb = (backend.GetCiphertextLevel(a),
                      backend.GetCiphertextLevel(b))
            mine_a = mine_b = False
            if la != lb:
                target = min(la, lb)
                if la > target:
                    a, mine_a = backend._align_lt_input(a, target)
                else:
                    b, mine_b = backend._align_lt_input(b, target)
                c.ptmult += 1
            prod = backend.MulRelinCiphertextNew(a, b)
            c.mult += 1
            prime[i + j - 1] = backend.RescaleNew(prod)
            backend.DeleteCiphertext(prod)
            # The aligned copy feeds exactly one multiply. Holding it to
            # the end of the tree, as this did, keeps O(p) of them alive
            # at once -- for p = 32768 that was tens of GiB of nothing.
            if mine_a:
                backend.DeleteCiphertext(a)
            if mine_b:
                backend.DeleteCiphertext(b)
        d += 1

    return prime, owned, c


def top_powers(backend, prime, n, js, counter=None):
    """The final tree stage, for the given j: alpha^(n/2 + j).

    Every anchor in the schedule is alpha^(2^d) with 2^d <= n/2, so the
    powers this stage produces -- alpha^(n/2+1) .. alpha^n, half of all
    of them -- are the only ones never read again. Producing them a
    chunk at a time and contracting each chunk into the running PCMM
    sums halves the peak live set, which is what bounds the reachable p.

    Returns (products, temporaries); both are the caller's to free.
    """
    c = counter or Counter()
    i = n // 2
    out, owned = [], []
    for j in js:
        a, b = prime[i - 1], prime[j - 1]
        la, lb = (backend.GetCiphertextLevel(a),
                  backend.GetCiphertextLevel(b))
        mine_a = mine_b = False
        if la != lb:
            target = min(la, lb)
            if la > target:
                a, mine_a = backend._align_lt_input(a, target)
            else:
                b, mine_b = backend._align_lt_input(b, target)
            c.ptmult += 1
        prod = backend.MulRelinCiphertextNew(a, b)
        c.mult += 1
        out.append(backend.RescaleNew(prod))
        backend.DeleteCiphertext(prod)
        if mine_a:
            backend.DeleteCiphertext(a)
        if mine_b:
            backend.DeleteCiphertext(b)
    return out, owned


def vecgen(backend, ct_id, p, counter=None, keep=None, const_ct=None):
    """Algorithm 1. Returns p ciphertext ids encrypting sqrt(2p)*v_j.

    `ct_id` encrypts alpha_j in every slot that carries a query. The
    returned ciphertexts are owned by the caller; intermediates are
    freed here except those `keep` asks to retain.
    """
    c = counter or Counter()
    n = p // 2
    prime, owned, c = power_tree(backend, ct_id, p, c)

    # Stage 2 (lines 10-18): split each power into 2*Re and 2*Im by
    # conjugation. z + conj(z) = 2Re(z); (z - conj(z))/sqrt(-1) = 2Im(z),
    # and the divide is key-free (MulImaginaryUnit + Negate), which is
    # why line 14 costs no level.
    out = [None] * p
    for k in range(n):
        src = prime[k]
        conj = backend.ConjugateNew(src)
        c.conj += 1
        out[k] = backend.AddCiphertextNew(src, conj)
        c.add += 1
        if k < n - 1:
            diff = backend.SubCiphertextNew(src, conj)
            c.add += 1
            backend.MulImaginaryUnit(diff)
            out[n + k] = backend.Negate(diff)
            backend.DeleteCiphertext(diff)
        backend.DeleteCiphertext(conj)

    # Line 16: the last coordinate is the constant sqrt(2) -- v(alpha)'s
    # final entry is 1/sqrt(2) and Algorithm 1 outputs sqrt(2p)*v, so it
    # carries no dependence on the query whatsoever. `const_ct` lets the
    # caller supply it once instead of re-encrypting it per call: it is
    # query-independent, so any implementation hoists it, and encoding
    # 2^15 slots per call would otherwise dominate VecGen at small p.
    out[p - 1] = const_ct if const_ct is not None else make_const(backend, out[0])

    for k in range(n):
        if prime[k] != ct_id and (keep is None or prime[k] not in keep):
            backend.DeleteCiphertext(prime[k])
    for h in owned:
        backend.DeleteCiphertext(h)
    return out, c
