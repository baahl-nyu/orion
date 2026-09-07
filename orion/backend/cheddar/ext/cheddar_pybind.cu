// Copyright 2026 The HE-LRM authors
// pybind11 module exposing Cheddar's CKKS API to Orion's backend layer.
//
// Not yet implemented: see bindings.py for the current list. Extending
// these is future work, not a permanent scope limit.
//
// Word size is uint64_t: Orion LogQ/LogP bit sizes map to one prime per
// level, same regime as the lattigo/desilo backends. The Python side
// generates the actual primes (Cheddar's Parameter takes explicit prime
// lists) and passes them in setup_scheme.
//
// Cheddar's AssertTrue calls std::exit on failure, which would kill the
// pytest process on a level/scale mismatch. Every op below pre-checks
// the conditions Cheddar asserts (same NP, same scale) and throws
// std::runtime_error instead, so misuse surfaces as a Python exception.

#include <Random.h>
#include <UserInterface.h>
#include <core/Context.h>
#include <core/MemoryPool.h>
#include <extension/BootContext.h>
#include <extension/EvalPoly.h>
#include <extension/Hoist.h>

#include <common/Basic.cuh>

#include <thrust/execution_policy.h>
#include <thrust/fill.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

// Dual-word build: the same source compiles into _cheddar_native
// (word=uint64_t, Orion LogQ/LogP bit-list chains) and _cheddar_native32
// (word=uint32_t, pregenerated rational-rescaling chains passed as
// explicit LevelConfig/TerPrimes -- Cheddar's native 25-30-prime design
// point, ~half the memory footprint per object). bindings.py picks the
// module at import time via ORION_CHEDDAR_WORD32.
#ifdef ORION_CHEDDAR_WORD32
using word = uint32_t;
#define ORION_CHEDDAR_MODULE_NAME _cheddar_native32
#else
using word = uint64_t;
#define ORION_CHEDDAR_MODULE_NAME _cheddar_native
#endif
using Ct = cheddar::Ciphertext<word>;
using Pt = cheddar::Plaintext<word>;
using Param = cheddar::Parameter<word>;
using Iface = cheddar::UserInterface<word>;
using Complex = std::complex<double>;

// ---------------------------------------------------------------------------
// Backend state -- single instance per process.
// ---------------------------------------------------------------------------

struct BackendState {
    // param_ must outlive context (Context holds a const Parameter&).
    std::unique_ptr<Param> param;
    cheddar::ContextPtr<word> context;
    std::unique_ptr<Iface> iface;

    // Rotation distances with a prepared key (normalized to [0, slots)).
    std::set<int> rot_keys;

    // Bootstrap. BootContext extends Context and is built against the
    // same param_ as the regular context above -- ciphertexts aren't
    // tied to a specific Context instance, so ops on either context
    // freely interoperate. Built lazily on first NewBootstrapper call
    // (not every scheme needs bootstrap). One BootContext/BootParameter
    // pair per scheme; PrepareEvalSpecialFFT is repeated per distinct
    // slot count (boot_prepared_slots tracks which have run).
    std::unique_ptr<cheddar::BootParameter> boot_param;
    std::shared_ptr<cheddar::BootContext<word>> boot_context;
    std::set<int> boot_prepared_slots;
    // EvalMod precompute is slot-independent and built once (lazily, on the
    // first NewBootstrapper call) into boot_context; this tracks that.
    bool eval_mod_prepared = false;
    // min_ks (minimum key-switching): far fewer boot rotation keys at the
    // cost of slower Boot. Set at setup_scheme from the Orion config; the
    // key generation (AddRequiredRotations) and Boot must agree on it.
    // Default false (full key set, fastest); worth enabling for workloads
    // that keep many boot circuits + linear-transform keys resident at once
    // (e.g. ResNet's several slot counts), where the full set OOMs a 24GB
    // card.
    bool boot_min_ks = false;

    double default_scale = 0.0;
    int max_level = 0;
    int slot_count = 0;
    // Explicit-chain (rational-rescaling) mode: set when setup_scheme
    // received a pregenerated LevelConfig instead of synthesizing the flat
    // one-prime-per-level chain. Rescaling then divides by multi-prime
    // products that only approximate 2^LogScale, so the exact scale a
    // plaintext must carry at level L follows the chain's canonical
    // trajectory (Parameter::GetScale) rather than the constant default
    // scale -- see canonical_scale() below.
    bool canonical_scales = false;

    // ID registries. next_*_id starts at 1; 0 is reserved for "invalid".
    std::map<int, std::unique_ptr<Pt>> plaintexts;
    std::map<int, std::unique_ptr<Ct>> ciphertexts;
    int next_pt_id = 1;
    int next_ct_id = 1;

    // Double-hoisted linear transforms: native HoistHandler per LT block
    // (see GenerateHoistLT below). pt_level recorded per handler because
    // HoistHandler asserts its input arrives at exactly that level.
    struct HoistLT {
        std::unique_ptr<cheddar::HoistHandler<word>> handler;
        int pt_level;
    };
    std::map<int, HoistLT> hoist_lts;
    int next_hoist_lt_id = 1;

    // Broadcast scalars in RNS form. One word per prime, against a
    // Plaintext's N words per prime -- see MulConstantAccumNew. Kept in
    // their own registry so a table of them can be encoded once, offline,
    // and reused across evaluations the way plaintexts are.
    struct StoredConst {
        std::unique_ptr<cheddar::Constant<word>> value;
        int level;
        double scale;
    };
    std::map<int, StoredConst> constants;
    int next_const_id = 1;

    // A d x p table of constants, repacked prime-major and contiguous so
    // the PC-MM kernel reads one array instead of chasing d*p device
    // pointers. Built once, offline -- see PackConstantTableNew.
    struct StoredConstTable {
        cheddar::DeviceVector<word> data;
        int rows;
        int cols;
        int qsize;
        int level;
        double scale;
    };
    std::map<int, StoredConstTable> const_tables;
    int next_table_id = 1;

    // The slot-varying twin: a rows x cols table of full plaintexts,
    // recorded as device pointers (the plaintexts stay in their own
    // registry). Used when several fields share one ciphertext, so a
    // table entry cannot be a broadcast scalar.
    struct StoredPtTable {
        const word** ptrs;
        int rows;
        int cols;
        int qsize;
        int level;
        double scale;
        int num_slots;
    };
    std::map<int, StoredPtTable> pt_tables;
    int next_pt_table_id = 1;

    // Device-side descriptor arrays for the PC-MM kernel (pointers and
    // aux offsets). Grown on demand and reused, so a per-call PC-MM does
    // not pay a cudaMalloc.
    void* scratch = nullptr;
    size_t scratch_bytes = 0;

    void* ensure_scratch(size_t bytes) {
        if (bytes > scratch_bytes) {
            if (scratch) cudaFree(scratch);
            if (cudaMalloc(&scratch, bytes) != cudaSuccess) {
                scratch = nullptr;
                scratch_bytes = 0;
                throw std::runtime_error("ensure_scratch: cudaMalloc failed.");
            }
            scratch_bytes = bytes;
        }
        return scratch;
    }

    void free_scratch() {
        if (scratch) cudaFree(scratch);
        scratch = nullptr;
        scratch_bytes = 0;
    }

    int put_pt(std::unique_ptr<Pt> pt) {
        int id = next_pt_id++;
        plaintexts[id] = std::move(pt);
        return id;
    }
    int put_ct(std::unique_ptr<Ct> ct) {
        int id = next_ct_id++;
        ciphertexts[id] = std::move(ct);
        return id;
    }
    Pt& pt(int id) {
        auto it = plaintexts.find(id);
        if (it == plaintexts.end())
            throw std::out_of_range("Plaintext id " + std::to_string(id) +
                                    " not found");
        return *it->second;
    }
    Ct& ct(int id) {
        auto it = ciphertexts.find(id);
        if (it == ciphertexts.end())
            throw std::out_of_range("Ciphertext id " + std::to_string(id) +
                                    " not found");
        return *it->second;
    }
    int put_const(std::unique_ptr<cheddar::Constant<word>> c, int level,
                  double scale) {
        int id = next_const_id++;
        constants[id] = StoredConst{std::move(c), level, scale};
        return id;
    }
    StoredConst& konst(int id) {
        auto it = constants.find(id);
        if (it == constants.end())
            throw std::out_of_range("Constant id " + std::to_string(id) +
                                    " not found");
        return it->second;
    }
};

static BackendState g_state;

void ensure_setup() {
    if (!g_state.context)
        throw std::runtime_error(
            "Cheddar backend not initialised. Call setup_scheme() first.");
}

// In-place op support: move the ciphertext computed at `from` into `id`'s
// map slot (destroying whatever was previously there) and drop the now-
// empty `from` entry. Callers compute a result via the existing *New path
// into a fresh id, then fold it back onto the id the caller expects to
// keep -- same pattern desilo's _replace uses, just at the map level since
// Cheddar's ids live in g_state rather than a Python-side registry.
void replace_ct(int id, int from) {
    g_state.ciphertexts[id] = std::move(g_state.ciphertexts.at(from));
    g_state.ciphertexts.erase(from);
}

template <typename Container>
int level_of(const Container& c) {
    return g_state.param->NPToLevel(c.GetNP());
}

// The scale fresh encodes should carry at `level`. Flat 64-bit chains:
// the constant default scale (2^LogScale), today's behavior, unchanged.
// Explicit rational-rescaling chains: the canonical per-level trajectory
// -- Cheddar maintains GetScale(l) = GetScale(l+1)^2 / rescale_prod(l+1)
// anchored at GetScale(0) == base_scale, and its ops assert scales
// near-exactly, so encoding at the nominal 2^LogScale (up to ~4% off
// trajectory mid-chain) would either trip those asserts or silently bias
// values. Mirrors upstream's own Testbed::DetermineScale.
double canonical_scale(int level) {
    if (!g_state.canonical_scales) return g_state.default_scale;
    if (level <= g_state.param->default_encryption_level_)
        return g_state.param->GetScale(level);
    return g_state.param->GetRescalePrimeProd(level);
}

// Cheddar's Add/Sub assert scale equality at 1e-12 relative and
// std::exit on failure. The kernels produce operands whose scales
// drift apart by ~1e-7..1e-6 relative (2^LogScale-vs-actual-prime
// drift, e.g. BMM3's identity-path vs masked-path chunks). Lattigo
// absorbs that drift silently (result takes op0's scale) and HEonGPU
// only ever tracks the nominal scale, so for cross-backend parity we
// absorb it too: within kDriftTolerance the second operand is
// relabeled to the first operand's scale (value bias == the relative
// drift, orders below the oracle tolerance). Bigger mismatches are
// real kernel bugs and still throw.
constexpr double kDriftTolerance = 1e-4;

// Returns b (possibly retagged via a temp copy) with scale matched to
// `scale`. Throws on mismatch beyond kDriftTolerance.
const Ct* absorb_scale_drift(const Ct& b, double scale,
                             std::unique_ptr<Ct>& tmp, const char* op) {
    double diff = std::abs(scale - b.GetScale());
    if (diff < 1e-12 * scale) return &b;
    if (!(diff < kDriftTolerance * scale)) {
        throw std::runtime_error(
            std::string(op) + ": scale mismatch beyond drift tolerance (" +
            std::to_string(scale) + " vs " + std::to_string(b.GetScale()) +
            ").");
    }
    if (!tmp) {
        tmp = std::make_unique<Ct>();
        g_state.context->Copy(*tmp, b);
    }
    tmp->SetScale(scale);
    return tmp.get();
}

// Align `c` down to target_level via LevelDown on a temp copy when
// needed. Returns the ciphertext to use (either &c or tmp.get()).
const Ct* align_down(const Ct& c, int target_level,
                     std::unique_ptr<Ct>& tmp) {
    if (level_of(c) <= target_level) return &c;
    tmp = std::make_unique<Ct>();
    g_state.context->LevelDown(*tmp, c, target_level);
    return tmp.get();
}

// Explicit caller-facing LevelDown: drop ct_id to target_level, returning
// a NEW ciphertext id (input untouched). Lets the Python LT evaluator
// align its input ONCE before the baby-step rotations instead of letting
// MulPlaintextAccumNew re-align every rotated operand -- LevelDown is a
// per-level Mult+Rescale walk in cheddar (Context::LevelDown), so the
// per-operand policy cost O(num_operands * level_gap) rescales per
// transform where O(level_gap) suffices. Rotating below the walk also
// shrinks every rotation's ModUp/ModDown to the target level's limb
// count. No-op gap (already at target) still returns a fresh copy so
// callers own the result unconditionally.
int AlignToLevelNew(int ct_id, int target_level) {
    const Ct& c = g_state.ct(ct_id);
    if (level_of(c) < target_level)
        throw std::runtime_error(
            "AlignToLevelNew: input level " + std::to_string(level_of(c)) +
            " below target " + std::to_string(target_level) + ".");
    auto out = std::make_unique<Ct>();
    if (level_of(c) == target_level) {
        g_state.context->Copy(*out, c);
    } else {
        g_state.context->LevelDown(*out, c, target_level);
    }
    return g_state.put_ct(std::move(out));
}

// ---------------------------------------------------------------------------
// Setup / teardown
// ---------------------------------------------------------------------------

void delete_scheme();  // forward decl (setup tears down prior state)

// Generation counter guarding against the bench runner's rebuild
// pattern: it constructs the NEW scheme before Python GC finalizes the
// OLD one, and the old scheme's DeleteScheme would otherwise wipe the
// global state the new scheme just set up (the exact failure mode that
// forces heongpu into one-process-per-shape). setup_scheme returns the
// new generation; delete_scheme_if only tears down when that
// generation is still the live one.
static int g_generation = 0;

int setup_scheme(py::dict params) {
    // Full teardown of any prior scheme first -- also releases the
    // previous shape's pool so back-to-back shapes don't stack 12GB
    // contexts on the card.
    delete_scheme();

    int logN = params["LogN"].cast<int>();
    int logScale = params["LogScale"].cast<int>();
    auto main_primes = params["MainPrimes"].cast<std::vector<word>>();
    auto aux_primes = params["AuxPrimes"].cast<std::vector<word>>();

    // Explicit chain (rational rescaling): Python passes the full
    // pregenerated structure -- LevelConfig ((num_main, num_ter) per
    // level), TerPrimes, AdditionalBase -- from a chain file like
    // Cheddar's parameters/bootparam_40.json, instead of this wrapper
    // synthesizing a flat one-prime-per-level chain from LogQ bit sizes.
    // Required for word32 (a ~2^40 scale cannot be one ~30-bit prime per
    // level) and enables canonical per-level scale handling everywhere.
    std::vector<word> ter_primes;
    std::pair<int, int> additional_base{0, 0};
    std::vector<std::pair<int, int>> level_config;
    int num_levels;
    g_state.canonical_scales = params.contains("LevelConfig");
    if (g_state.canonical_scales) {
        for (auto item : params["LevelConfig"].cast<py::list>()) {
            auto pair = item.cast<std::vector<int>>();
            if (pair.size() != 2)
                throw std::runtime_error(
                    "setup_scheme: LevelConfig entries must be "
                    "(num_main, num_ter) pairs.");
            level_config.emplace_back(pair[0], pair[1]);
        }
        num_levels = static_cast<int>(level_config.size());
        if (params.contains("TerPrimes"))
            ter_primes = params["TerPrimes"].cast<std::vector<word>>();
        if (params.contains("AdditionalBase")) {
            auto ab = params["AdditionalBase"].cast<std::vector<int>>();
            if (ab.size() != 2)
                throw std::runtime_error(
                    "setup_scheme: AdditionalBase must be a pair.");
            additional_base = {ab[0], ab[1]};
        }
    } else {
        num_levels = static_cast<int>(main_primes.size());
        // Flat chain: level i uses main primes 0..i, no terminal primes.
        // Orion's level convention (level = index into LogQ) maps directly.
        level_config.reserve(num_levels);
        for (int i = 0; i < num_levels; ++i)
            level_config.emplace_back(i + 1, 0);
    }

    double base_scale = std::pow(2.0, logScale);

    // default_encryption_level = the top of the *usable* chain, where
    // fresh ciphertexts are encrypted and where Orion's level accounting
    // tops out. For bootstrap-enabled schemes this is strictly below
    // num_levels-1: the topmost (num_cts + eval_mod) primes are reserved
    // for the boot circuit (CoeffToSlot + EvalMod), and BootContext
    // asserts default_encryption_level == GetStCStartLevel() == max_level
    // - num_cts_levels - GetNumEvalModLevels(). When bootstrap is not
    // configured, Python passes num_levels-1 (whole chain usable) and
    // no primes are reserved.
    int default_enc_level = params.contains("DefaultEncryptionLevel")
        ? params["DefaultEncryptionLevel"].cast<int>()
        : num_levels - 1;

    g_state.param = std::make_unique<Param>(
        logN, base_scale, default_enc_level,
        level_config, main_primes, aux_primes, ter_primes, additional_base);

    // Hamming weights (secret sparsity). Chain files carry the values the
    // chain's security analysis assumed; the flat path keeps Parameter's
    // defaults, unchanged.
    if (params.contains("DenseHammingWeight"))
        g_state.param->SetDenseHammingWeight(
            params["DenseHammingWeight"].cast<int>());
    if (params.contains("SparseHammingWeight"))
        g_state.param->SetSparseHammingWeight(
            params["SparseHammingWeight"].cast<int>());

    // When bootstrap is configured (Python passes the boot level knobs),
    // build a BootContext and use it as *the* context -- BootContext
    // derives from Context, so all regular ops go through it unchanged.
    // Keeping a single context (rather than a separate regular Context +
    // BootContext) roughly halves resident memory, which is what lets
    // full-slot LogN=16 bootstrap fit on a 24GB card. The heavy,
    // slot-specific precompute (EvalMod / special FFT) stays lazy in
    // NewBootstrapper; only the light constant tables are built here.
    if (params.contains("BootNumCtsLevels")) {
        int nc = params["BootNumCtsLevels"].cast<int>();
        int ns = params["BootNumStcLevels"].cast<int>();
        int lmr = params.contains("BootLogMessageRatio")
            ? params["BootLogMessageRatio"].cast<int>() : 5;
        g_state.boot_param = std::make_unique<cheddar::BootParameter>(
            num_levels - 1, nc, ns, lmr);
        g_state.boot_context = cheddar::BootContext<word>::Create(
            *g_state.param, *g_state.boot_param);
        g_state.eval_mod_prepared = false;
        g_state.boot_min_ks = params.contains("BootMinKs")
            ? params["BootMinKs"].cast<bool>() : false;
        g_state.context = g_state.boot_context;  // shared: one context
    } else {
        g_state.context = cheddar::Context<word>::Create(*g_state.param);
    }

    g_state.default_scale = base_scale;
    g_state.max_level = num_levels - 1;
    g_state.slot_count = (1 << logN) / 2;
    return ++g_generation;
}

void delete_scheme() {
    g_state.plaintexts.clear();
    g_state.ciphertexts.clear();
    // Constants hold device memory like the other containers; leaving
    // them to static destruction runs their frees after the CUDA context
    // is gone, which segfaults at exit (seen as exit=139 after a clean
    // run has already written its results).
    g_state.constants.clear();
    g_state.const_tables.clear();
    for (auto& kv : g_state.pt_tables)
        if (kv.second.ptrs) cudaFree(kv.second.ptrs);
    g_state.pt_tables.clear();
    g_state.free_scratch();
    g_state.hoist_lts.clear();
    g_state.rot_keys.clear();
    g_state.boot_prepared_slots.clear();
    g_state.eval_mod_prepared = false;
    // boot_context aliases context (shared_ptr): resetting it here drops
    // one ref; context.reset() below drops the last and destroys the
    // single underlying (Boot)Context.
    g_state.boot_context.reset();
    g_state.boot_param.reset();
    g_state.iface.reset();
    g_state.context.reset();
    g_state.param.reset();
}

void delete_scheme_if(int generation) {
    if (generation == g_generation) delete_scheme();
}

// ---------------------------------------------------------------------------
// Key generation
// ---------------------------------------------------------------------------
// UserInterface's constructor samples the secrets and generates the basic
// evaluation keys (mult / conj / dense-to-sparse / sparse-to-dense), so
// the Orion keygen steps collapse into constructing it once.

void NewKeyGenerator() { ensure_setup(); }
void GenerateSecretKey() {
    ensure_setup();
    if (!g_state.iface)
        g_state.iface = std::make_unique<Iface>(g_state.context);
}
void GeneratePublicKey() { ensure_setup(); }
void GenerateRelinearizationKey() { ensure_setup(); }
void GenerateEvaluationKeys() {
    // Rotation keys are generated per-shift via AddRotationKey.
}

void ensure_keys() {
    ensure_setup();
    if (!g_state.iface)
        throw std::runtime_error(
            "Cheddar keys not generated. Call GenerateSecretKey() first.");
}

// ---------------------------------------------------------------------------
// Encryptor / Decryptor -- no-ops, UserInterface covers both.
// ---------------------------------------------------------------------------

void NewEncryptor() { ensure_keys(); }
void NewDecryptor() { ensure_keys(); }

// ---------------------------------------------------------------------------
// Encode / Decode
// ---------------------------------------------------------------------------

int Encode(std::vector<double> values, int level, double scale) {
    ensure_setup();
    double s = (scale > 0.0) ? scale : canonical_scale(level);
    // Explicit-chain mode: the canonical trajectory is the unique
    // convention under which both ct-ct and ct-pt mult-then-rescale stay
    // scale-exact (canonical(l)^2 / rescale_prod(l) == canonical(l-1)),
    // so every plaintext at level l must sit at exactly canonical(l).
    // Callers request it only approximately -- the nominal 2^LogScale
    // (encoder.py, tests) or the moduli-chain entry (Orion's errorless-
    // rescale idiom; GetModuliChain reports rounded canonical scales in
    // this mode) -- so snap any request within the chain's wobble range
    // (rescale products stray up to ~13% off nominal) onto the exact
    // value. Requests further out are honored as passed.
    if (g_state.canonical_scales) {
        double canon = canonical_scale(level);
        if (std::abs(s - canon) < 0.15 * canon) s = canon;
    }
    std::vector<Complex> msg(values.size());
    for (size_t i = 0; i < values.size(); ++i) msg[i] = Complex(values[i], 0.0);
    auto pt = std::make_unique<Pt>();
    g_state.context->encoder_.Encode(*pt, level, s, msg);
    return g_state.put_pt(std::move(pt));
}

// Complex-lane variant: `interleaved` is [re0, im0, re1, im1, ...].
// Cheddar's encoder is natively complex, so this is the same call as Encode minus the
// imaginary-part zeroing.
int EncodeComplex(std::vector<double> interleaved, int level, double scale) {
    ensure_setup();
    if (interleaved.size() % 2 != 0)
        throw std::runtime_error(
            "EncodeComplex: expected interleaved [re, im, ...] pairs.");
    double s = (scale > 0.0) ? scale : canonical_scale(level);
    if (g_state.canonical_scales) {
        double canon = canonical_scale(level);
        if (std::abs(s - canon) < 0.15 * canon) s = canon;
    }
    std::vector<Complex> msg(interleaved.size() / 2);
    for (size_t i = 0; i < msg.size(); ++i)
        msg[i] = Complex(interleaved[2 * i], interleaved[2 * i + 1]);
    auto pt = std::make_unique<Pt>();
    g_state.context->encoder_.Encode(*pt, level, s, msg);
    return g_state.put_pt(std::move(pt));
}

std::vector<double> Decode(int pt_id) {
    ensure_setup();
    std::vector<Complex> msg;
    g_state.context->encoder_.Decode(msg, g_state.pt(pt_id));
    std::vector<double> out(msg.size());
    for (size_t i = 0; i < msg.size(); ++i) out[i] = msg[i].real();
    return out;
}

std::vector<double> DecodeComplex(int pt_id) {
    ensure_setup();
    std::vector<Complex> msg;
    g_state.context->encoder_.Decode(msg, g_state.pt(pt_id));
    std::vector<double> out(2 * msg.size());
    for (size_t i = 0; i < msg.size(); ++i) {
        out[2 * i] = msg[i].real();
        out[2 * i + 1] = msg[i].imag();
    }
    return out;
}

// Re-encode an existing plaintext at a new level/scale (decode+encode).
// Offline-path helper used by the ct-pt level/scale alignment below.
std::unique_ptr<Pt> reencode(const Pt& src, int level, double scale) {
    std::vector<Complex> msg;
    g_state.context->encoder_.Decode(msg, src);
    // Decode returns values already divided by src's scale, so encoding
    // at `scale` reproduces the same logical message.
    auto pt = std::make_unique<Pt>();
    g_state.context->encoder_.Encode(*pt, level, scale, msg);
    return pt;
}

// ---------------------------------------------------------------------------
// Encrypt / Decrypt
// ---------------------------------------------------------------------------

int Encrypt(int pt_id) {
    ensure_keys();
    auto ct = std::make_unique<Ct>();
    g_state.iface->Encrypt(*ct, g_state.pt(pt_id));
    return g_state.put_ct(std::move(ct));
}

// ---------------------------------------------------------------------------
// Seeded ("compressed") ciphertexts
//
// A fresh RLWE ciphertext is (b, a) with a uniform and independent of
// the message and secret, so a client can derive a from a seed and send
// only (b, seed) -- halving upload. The server rebuilds a with
// ExpandSeededAx. Seeds must be fresh per ciphertext; reusing one a
// across two messages under one key leaks their difference.
// ---------------------------------------------------------------------------

int EncryptSeeded(int pt_id, std::uint64_t seed) {
    ensure_keys();
    auto ct = std::make_unique<Ct>();
    // Seeding here fixes the `a` draw that Encrypt makes first; the
    // error term drawn afterwards is then also deterministic, which is
    // fine because only the client ever sees it.
    cheddar::Random::SetSeed(seed);
    g_state.iface->Encrypt(*ct, g_state.pt(pt_id));
    return g_state.put_ct(std::move(ct));
}

// Emulates "only b crossed the wire": drop a, keeping b and metadata.
void DropAx(int ct_id) {
    ensure_setup();
    Ct& c = g_state.ct(ct_id);
    thrust::fill(thrust::device, c.ax_.begin(), c.ax_.end(),
                 static_cast<word>(0));
}

void ExpandSeededAx(int ct_id, std::uint64_t seed) {
    ensure_keys();
    g_state.iface->ExpandSeededAx(g_state.ct(ct_id), seed);
}

// True wire size of a ciphertext, measured from the polynomial buffers
// rather than assumed from its level. The level -> limb-count map is a
// table (Parameter::level_config_), not level+1, so on the word32
// rational-rescaling chain a level can carry several primes.
double GetCiphertextBytes(int ct_id) {
    ensure_setup();
    const Ct& c = g_state.ct(ct_id);
    std::size_t words = c.bx_.size() + c.ax_.size();
    if (c.HasRx()) words += c.rx_.size();
    return static_cast<double>(words) * sizeof(word);
}

// Just the b polynomial -- what a seeded upload actually transmits.
double GetCiphertextBxBytes(int ct_id) {
    ensure_setup();
    return static_cast<double>(g_state.ct(ct_id).bx_.size()) * sizeof(word);
}

int Decrypt(int ct_id) {
    ensure_keys();
    auto pt = std::make_unique<Pt>();
    const Ct& c = g_state.ct(ct_id);
    if (c.HasRx()) {
        // Defensive: kernels relinearize before decrypt, but if one
        // slips through, relinearize a temp instead of failing inside
        // UserInterface.
        Ct tmp;
        g_state.context->Relinearize(tmp, c,
                                     g_state.iface->GetMultiplicationKey());
        g_state.iface->Decrypt(*pt, tmp);
    } else {
        g_state.iface->Decrypt(*pt, c);
    }
    return g_state.put_pt(std::move(pt));
}

// ---------------------------------------------------------------------------
// ct ± ct
// ---------------------------------------------------------------------------

int AddCiphertextNew(int a, int b) {
    ensure_setup();
    const Ct& ca = g_state.ct(a);
    const Ct& cb = g_state.ct(b);
    int target = std::min(level_of(ca), level_of(cb));
    std::unique_ptr<Ct> ta, tb;
    const Ct* ua = align_down(ca, target, ta);
    const Ct* ub = align_down(cb, target, tb);
    ub = absorb_scale_drift(*ub, ua->GetScale(), tb, "AddCiphertextNew");
    auto out = std::make_unique<Ct>();
    g_state.context->Add(*out, *ua, *ub);
    return g_state.put_ct(std::move(out));
}

int AddCiphertext(int a, int b) {
    replace_ct(a, AddCiphertextNew(a, b));
    return a;
}

int SubCiphertextNew(int a, int b) {
    ensure_setup();
    const Ct& ca = g_state.ct(a);
    const Ct& cb = g_state.ct(b);
    int target = std::min(level_of(ca), level_of(cb));
    std::unique_ptr<Ct> ta, tb;
    const Ct* ua = align_down(ca, target, ta);
    const Ct* ub = align_down(cb, target, tb);
    ub = absorb_scale_drift(*ub, ua->GetScale(), tb, "SubCiphertextNew");
    auto out = std::make_unique<Ct>();
    g_state.context->Sub(*out, *ua, *ub);
    return g_state.put_ct(std::move(out));
}

int SubCiphertext(int a, int b) {
    replace_ct(a, SubCiphertextNew(a, b));
    return a;
}

int MulNoRelinCiphertextNew(int a, int b) {
    ensure_setup();
    const Ct& ca = g_state.ct(a);
    const Ct& cb = g_state.ct(b);
    if (ca.HasRx() || cb.HasRx())
        throw std::runtime_error(
            "MulNoRelinCiphertextNew: operand is degree-2; relinearize "
            "before multiplying again.");
    int target = std::min(level_of(ca), level_of(cb));
    std::unique_ptr<Ct> ta, tb;
    const Ct* ua = align_down(ca, target, ta);
    const Ct* ub = align_down(cb, target, tb);
    // Tensor-only product: result carries rx (degree-2) until an
    // explicit Relinearize -- Cheddar's native lazy-relin model.
    auto out = std::make_unique<Ct>();
    g_state.context->Mult(*out, *ua, *ub);
    return g_state.put_ct(std::move(out));
}

int RelinearizeNew(int a) {
    ensure_keys();
    const Ct& c = g_state.ct(a);
    auto out = std::make_unique<Ct>();
    if (c.HasRx()) {
        g_state.context->Relinearize(*out, c,
                                     g_state.iface->GetMultiplicationKey());
    } else {
        // Kernel tail-end relin runs even when the product was already
        // relinearized; make it a clone in that case.
        g_state.context->Copy(*out, c);
    }
    return g_state.put_ct(std::move(out));
}

int MulRelinCiphertextNew(int a, int b) {
    ensure_keys();
    int prod = MulNoRelinCiphertextNew(a, b);
    int out = RelinearizeNew(prod);
    g_state.ciphertexts.erase(prod);
    return out;
}

int MulRelinCiphertext(int a, int b) {
    replace_ct(a, MulRelinCiphertextNew(a, b));
    return a;
}

// ---------------------------------------------------------------------------
// ct x pt (THOR / BMM-3 hot path)
// ---------------------------------------------------------------------------
// Cheddar's ct-pt ops assert same NP. The kernels encode masks one level
// below the input at times (lattigo/desilo auto-align), so we LevelDown
// the ct to the pt's level when needed. The reverse case (pt above ct)
// re-encodes the pt at the ct's level -- offline path, rare.

const Pt* align_pt(const Ct& c, const Pt& p, std::unique_ptr<Pt>& tmp,
                   double scale_override = 0.0) {
    int ct_level = level_of(c);
    double scale = (scale_override > 0.0) ? scale_override : p.GetScale();
    if (level_of(p) == ct_level && scale == p.GetScale()) return &p;
    tmp = reencode(p, ct_level, scale);
    return tmp.get();
}

int MulPlaintextNew(int ct_id, int pt_id) {
    ensure_setup();
    const Ct& c = g_state.ct(ct_id);
    const Pt& p = g_state.pt(pt_id);
    std::unique_ptr<Ct> tc;
    std::unique_ptr<Pt> tp;
    // ct above pt: drop ct (cheap). pt above ct: re-encode pt.
    const Ct* uc = align_down(c, std::min(level_of(c), level_of(p)), tc);
    const Pt* up = (level_of(p) > level_of(*uc))
                       ? align_pt(*uc, p, tp)
                       : &p;
    auto out = std::make_unique<Ct>();
    g_state.context->Mult(*out, *uc, *up);  // PMult covers rx (degree-2)
    return g_state.put_ct(std::move(out));
}

int MulPlaintext(int ct_id, int pt_id) {
    replace_ct(ct_id, MulPlaintextNew(ct_id, pt_id));
    return ct_id;
}

// Fused MAC: out = sum_i ct_i * pt_i in one ElementWiseHandler::PAccum
// call instead of N ct-pt Mults + (N-1) Adds -- each of those is a
// separate memory-bound elementwise pass over full ciphertexts, so
// fusing them into one kernel (CPAccumWorker chunks internally by 8,
// carrying partial sums through the trailing extra-ct slot, so any N
// works) removes 2N-2 full-ciphertext memory round trips. Built purely
// from public Context members (elem_handler_), same as
// RotateBatchHoistedNew -- no cheddar source patch needed. Mirrors
// HoistHandler::EvaluateSingleAccum's result setup (Hoist.cu).
int MulPlaintextAccumNew(std::vector<int> ct_ids, std::vector<int> pt_ids) {
    ensure_setup();
    if (ct_ids.empty() || ct_ids.size() != pt_ids.size())
        throw std::runtime_error(
            "MulPlaintextAccumNew: ct_ids/pt_ids must be equal-length and "
            "non-empty.");
    if (ct_ids.size() == 1) return MulPlaintextNew(ct_ids[0], pt_ids[0]);

    // Align everything to the lowest operand level (same policy as
    // MulPlaintextNew: drop cts, re-encode pts).
    int target = 1 << 30;
    for (int id : ct_ids) target = std::min(target, level_of(g_state.ct(id)));
    for (int id : pt_ids) target = std::min(target, level_of(g_state.pt(id)));

    size_t n = ct_ids.size();
    std::vector<std::unique_ptr<Ct>> tmp_cts(n);
    std::vector<std::unique_ptr<Pt>> tmp_pts(n);
    std::vector<const Ct*> cts(n);
    std::vector<const Pt*> pts(n);
    for (size_t i = 0; i < n; i++) {
        const Ct& c = g_state.ct(ct_ids[i]);
        if (c.HasRx())
            throw std::runtime_error(
                "MulPlaintextAccumNew: operand is degree-2; relinearize "
                "first.");
        cts[i] = align_down(c, target, tmp_cts[i]);
        const Pt& p = g_state.pt(pt_ids[i]);
        pts[i] = (level_of(p) > target) ? align_pt(*cts[i], p, tmp_pts[i])
                                        : &p;
    }

    const cheddar::NPInfo np = cts[0]->GetNP();
    const double out_scale = cts[0]->GetScale() * pts[0]->GetScale();
    int num_slots = 0;
    for (size_t i = 0; i < n; i++) {
        if (cts[i]->GetNP().GetNumQ() != np.GetNumQ() ||
            cts[i]->GetNP().num_aux_ != np.num_aux_)
            throw std::runtime_error(
                "MulPlaintextAccumNew: ciphertext NP mismatch after level "
                "alignment.");
        double s = cts[i]->GetScale() * pts[i]->GetScale();
        if (std::abs(s - out_scale) > 1e-9 * out_scale)
            throw std::runtime_error(
                "MulPlaintextAccumNew: inconsistent ct*pt scale products "
                "across pairs.");
        num_slots = std::max(
            num_slots, std::max(cts[i]->GetNumSlots(), pts[i]->GetNumSlots()));
    }

    auto out = std::make_unique<Ct>();
    out->RemoveRx();
    out->ModifyNP(np);
    out->SetScale(out_scale);
    out->SetNumSlots(num_slots);
    std::vector<cheddar::DvView<word>> out_view = out->ViewVector();

    std::vector<std::vector<cheddar::DvConstView<word>>> ct_views;
    std::vector<cheddar::DvConstView<word>> pt_views;
    ct_views.reserve(n);
    pt_views.reserve(n);
    for (size_t i = 0; i < n; i++) {
        ct_views.push_back(cts[i]->ConstViewVector());
        pt_views.push_back(pts[i]->ConstView());
    }
    g_state.context->elem_handler_.PAccum(out_view, np, ct_views, pt_views);
    return g_state.put_ct(std::move(out));
}

// Constant-weighted accumulation: sum_i scalars[i] * ct_ids[i], fused.
//
// Same shape as MulPlaintextAccumNew, but each weight is a broadcast
// scalar rather than a slot vector, so it rides cheddar's CAccum (the
// path EvalPoly and BootContext already use) instead of PAccum. That
// matters for memory, not just speed: a Constant<word> is one word per
// prime, while the equivalent Plaintext is N words per prime -- 65536x
// larger for the same information. IVE's PCMM multiplies p coordinate
// ciphertexts by d*p table scalars, and materialising those as full
// plaintexts is what put p >= 8192 out of reach on a 96 GB card.
//
// Scale follows MulScalarFloat: each constant is encoded at the
// ciphertext's own scale, so the result's scale is squared and the
// caller must Rescale once -- identical to the plaintext path it
// replaces.
cheddar::Constant<word> encode_const(double number, int level, double scale);

// Broadcast-scalar table entries, encoded once and reused.
//
// A Constant<word> holds one word per prime; the Plaintext that encodes
// the same broadcast value holds N words per prime -- 65536x more for
// identical arithmetic. IVE's PCMM multiplies p coordinate ciphertexts
// by d*p table scalars, and materialising those as plaintexts is what
// put p >= 8192 out of reach on a 96 GB card.
int EncodeConstantNew(double scalar, int level, double scale) {
    ensure_setup();
    auto c = std::make_unique<cheddar::Constant<word>>(
        encode_const(scalar, level, scale));
    return g_state.put_const(std::move(c), level, scale);
}

void DeleteConstant(int id) {
    g_state.constants.erase(id);
}

// sum_i const_ids[i] * ct_ids[i], fused through cheddar's CAccum -- the
// path EvalPoly and BootContext already use. Structurally the same as
// MulPlaintextAccumNew; the weights are broadcast scalars rather than
// slot vectors.
//
// Scale follows MulScalarFloat: constants are encoded at the
// ciphertext's scale, so the result's scale is squared and the caller
// Rescales once, exactly as with the plaintext path it replaces.
//
// Constants carry no level-alignment path (there is no Constant analogue
// of align_pt), so every constant must already sit at the level the
// ciphertexts align down to. That is the natural calling pattern: the
// table is encoded at VecGen's output level and the coordinates arrive
// there.
int MulConstantAccumNew(std::vector<int> ct_ids, std::vector<int> const_ids) {
    ensure_setup();
    if (ct_ids.empty() || ct_ids.size() != const_ids.size())
        throw std::runtime_error(
            "MulConstantAccumNew: ct_ids/const_ids must be equal-length and "
            "non-empty.");

    int target = 1 << 30;
    for (int id : ct_ids) target = std::min(target, level_of(g_state.ct(id)));

    size_t n = ct_ids.size();
    std::vector<std::unique_ptr<Ct>> tmp_cts(n);
    std::vector<const Ct*> cts(n);
    for (size_t i = 0; i < n; i++) {
        const Ct& c = g_state.ct(ct_ids[i]);
        if (c.HasRx())
            throw std::runtime_error(
                "MulConstantAccumNew: operand is degree-2; relinearize "
                "first.");
        cts[i] = align_down(c, target, tmp_cts[i]);
    }

    const cheddar::NPInfo np = cts[0]->GetNP();
    const double ct_scale = cts[0]->GetScale();
    const double const_scale = g_state.konst(const_ids[0]).scale;
    int num_slots = 0;
    for (size_t i = 0; i < n; i++) {
        if (cts[i]->GetNP().GetNumQ() != np.GetNumQ() ||
            cts[i]->GetNP().num_aux_ != np.num_aux_)
            throw std::runtime_error(
                "MulConstantAccumNew: ciphertext NP mismatch after level "
                "alignment.");
        if (std::abs(cts[i]->GetScale() - ct_scale) > 1e-9 * ct_scale)
            throw std::runtime_error(
                "MulConstantAccumNew: inconsistent ciphertext scales.");
        if (std::abs(g_state.konst(const_ids[i]).scale - const_scale) >
            1e-9 * const_scale)
            throw std::runtime_error(
                "MulConstantAccumNew: inconsistent constant scales.");
        num_slots = std::max(num_slots, cts[i]->GetNumSlots());
        if (g_state.konst(const_ids[i]).level != target)
            throw std::runtime_error(
                "MulConstantAccumNew: constant " +
                std::to_string(const_ids[i]) + " is at level " +
                std::to_string(g_state.konst(const_ids[i]).level) +
                " but the ciphertexts align to level " +
                std::to_string(target) + "; re-encode the table there.");
    }

    auto out = std::make_unique<Ct>();
    out->RemoveRx();
    out->ModifyNP(np);
    out->SetScale(ct_scale * const_scale);
    out->SetNumSlots(num_slots);
    std::vector<cheddar::DvView<word>> out_view = out->ViewVector();

    std::vector<std::vector<cheddar::DvConstView<word>>> ct_views;
    std::vector<cheddar::DvConstView<word>> const_views;
    ct_views.reserve(n);
    const_views.reserve(n);
    for (size_t i = 0; i < n; i++) {
        ct_views.push_back(cts[i]->ConstViewVector());
        const_views.push_back(g_state.konst(const_ids[i]).value->ConstView());
    }
    g_state.context->elem_handler_.CAccum(out_view, np, ct_views, const_views);
    return g_state.put_ct(std::move(out));
}

// ---------------------------------------------------------------------------
// PC-MM in one pass over the ciphertext matrix
// ---------------------------------------------------------------------------
// Bae-Cheon-Hanrot-Park-Stehle (CRYPTO'24, IVE's reference [5]) observe
// that multiplying a plaintext matrix U into ciphertexts whose rows are
// separate RLWE elements is, coefficient-wise, nothing but two plaintext
// matrix products: writing the ciphertexts' polynomials as rows of A and
// B, U*A and U*B are again the a- and b-parts of the answer. Their point
// is that a PC-MM should therefore run at the speed of a PP-MM -- one
// GEMM, each input read once -- rather than as a sequence of scalar
// multiply-accumulates.
//
// MulConstantAccumNew already computes U*A and U*B; it just computes them
// one output row at a time, so the p input ciphertexts are streamed from
// HBM once per output row. For IVE's d x p table that is d-fold redundant
// traffic (and, past cheddar's 8-way accumulation chunk, a recursive
// temp-buffer walk on top). ConstMatMulKernel below is the GEMM form: one
// pass over the inputs, all d accumulators live in registers.
//
// Their remaining tricks do not transfer. The floating-point PP-MM of
// Section 5 buys nothing here (a GPU's modular multiplier is not the
// bottleneck; memory traffic is), and the shared-a format of Section 3
// does not apply to VecGen's outputs, which are products and conjugates
// with independent a-parts.

constexpr int kMatMulBlock = 256;
constexpr int kMatMulKTile = 64;

// out[j][i] = in[i][j]: the d*p constants, each one word per prime,
// repacked so the online kernel indexes one contiguous array per prime
// instead of chasing d*p device pointers. Runs once, offline.
template <typename W>
__global__ void GatherConstTableKernel(W* out, const W* const* in, int n,
                                       int tsize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n * tsize) return;
    int i = tid / tsize;
    int j = tid - i * tsize;
    out[static_cast<size_t>(j) * n + i] = in[i][j];
}

// One RNS coefficient position per thread; D output rows accumulated at
// once. The k-tile stages the table entries and source pointers for the
// current chunk of input ciphertexts in shared memory, so the inner loop
// is a register-resident FMA chain over streamed source words.
template <typename W, int D>
__global__ void ConstMatMulKernel(
    W* const* __restrict__ dst_ptrs, const W* const* __restrict__ src_ptrs,
    const int* __restrict__ src_extra, const W* __restrict__ table,
    int table_stride, int table_qsize, int row0, int rows,
    const W* __restrict__ primes,
    const cheddar::make_signed_t<W>* __restrict__ inv_primes, int num_q_primes,
    int log_degree, int P) {
    using SW = cheddar::make_signed_t<W>;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int prime_index = i >> log_degree;
    const W prime = primes[prime_index];
    const SW inv_prime = inv_primes[prime_index];
    // Aux primes sit behind the container's own q-part, which may be
    // longer than this NP's -- the same offset CPAccum applies.
    const bool aux = (prime_index >= num_q_primes);
    const int common_index =
        aux ? prime_index + (table_qsize - num_q_primes) : prime_index;
    const W* trow = table + static_cast<size_t>(common_index) * table_stride;

    __shared__ W s_const[D * kMatMulKTile];
    __shared__ const W* s_src[2 * kMatMulKTile];
    __shared__ int s_extra[kMatMulKTile];

    W acc[D][2];
#pragma unroll
    for (int r = 0; r < D; r++) {
        acc[r][0] = 0;
        acc[r][1] = 0;
    }

    for (int k0 = 0; k0 < P; k0 += kMatMulKTile) {
        const int kt = min(kMatMulKTile, P - k0);
        __syncthreads();
        for (int t = threadIdx.x; t < D * kt; t += blockDim.x) {
            const int r = t / kt;
            const int kk = t - r * kt;
            s_const[r * kMatMulKTile + kk] =
                (r < rows)
                    ? trow[static_cast<size_t>(row0 + r) * P + k0 + kk]
                    : W{0};
        }
        for (int t = threadIdx.x; t < kt; t += blockDim.x) {
            s_src[2 * t] = src_ptrs[2 * (k0 + t)];
            s_src[2 * t + 1] = src_ptrs[2 * (k0 + t) + 1];
            s_extra[t] = src_extra[k0 + t];
        }
        __syncthreads();

        for (int kk = 0; kk < kt; kk++) {
            const int off = aux ? s_extra[kk] : 0;
            const W v0 =
                cheddar::basic::StreamingLoad(s_src[2 * kk] + i + off);
            const W v1 =
                cheddar::basic::StreamingLoad(s_src[2 * kk + 1] + i + off);
#pragma unroll
            for (int r = 0; r < D; r++) {
                const W c = s_const[r * kMatMulKTile + kk];
                acc[r][0] = cheddar::basic::Add(
                    acc[r][0],
                    cheddar::basic::MultMontgomery(c, v0, prime, inv_prime),
                    prime);
                acc[r][1] = cheddar::basic::Add(
                    acc[r][1],
                    cheddar::basic::MultMontgomery(c, v1, prime, inv_prime),
                    prime);
            }
        }
    }

#pragma unroll
    for (int r = 0; r < D; r++) {
        if (r < rows) {
            dst_ptrs[2 * r][i] = acc[r][0];
            dst_ptrs[2 * r + 1][i] = acc[r][1];
        }
    }
}

// Pack a d x p table of already-encoded constants (row-major, entry
// (r, k) at const_ids[r * p + k]) into the layout ConstMatMulKernel
// reads. Offline work: the IVE table is query-independent.
int PackConstantTableNew(std::vector<int> const_ids, int rows) {
    ensure_setup();
    if (rows <= 0 || const_ids.empty() ||
        const_ids.size() % static_cast<size_t>(rows) != 0)
        throw std::runtime_error(
            "PackConstantTableNew: const_ids must be a non-empty row-major "
            "rows x cols table.");
    const int n = static_cast<int>(const_ids.size());
    const int cols = n / rows;

    const auto& first = g_state.konst(const_ids[0]);
    const int level = first.level;
    const double scale = first.scale;
    const int qsize = first.value->ConstView().QSize();
    const int tsize = first.value->ConstView().TotalSize();

    std::vector<const word*> hptr(n);
    for (int i = 0; i < n; i++) {
        const auto& k = g_state.konst(const_ids[i]);
        if (k.level != level)
            throw std::runtime_error(
                "PackConstantTableNew: all constants must share one level.");
        if (std::abs(k.scale - scale) > 1e-9 * scale)
            throw std::runtime_error(
                "PackConstantTableNew: all constants must share one scale.");
        auto view = k.value->ConstView();
        if (view.QSize() != qsize || view.TotalSize() != tsize)
            throw std::runtime_error(
                "PackConstantTableNew: inconsistent constant layout.");
        hptr[i] = view.data();
    }

    const word** dptr = nullptr;
    if (cudaMalloc(&dptr, sizeof(const word*) * n) != cudaSuccess)
        throw std::runtime_error(
            "PackConstantTableNew: cudaMalloc for the pointer array failed.");
    cudaMemcpy(dptr, hptr.data(), sizeof(const word*) * n,
               cudaMemcpyHostToDevice);

    cheddar::DeviceVector<word> packed(n * tsize);
    const int total = n * tsize;
    GatherConstTableKernel<word>
        <<<(total + 255) / 256, 256>>>(packed.data(), dptr, n, tsize);
    cudaError_t err = cudaDeviceSynchronize();
    cudaFree(dptr);
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("PackConstantTableNew: ") +
                                 cudaGetErrorString(err));

    int id = g_state.next_table_id++;
    g_state.const_tables.emplace(
        id, BackendState::StoredConstTable{std::move(packed), rows, cols,
                                          qsize, level, scale});
    return id;
}

void DeleteConstantTable(int id) { g_state.const_tables.erase(id); }

// U * (the p input ciphertexts), all d output rows in one pass. Returns
// d new ciphertext ids, in table row order. Scale follows the CAccum
// path: constants carry the ciphertexts' own scale, so each output's
// scale is squared and the caller Rescales once.
std::vector<int> MulConstantMatMulNew(std::vector<int> ct_ids, int table_id) {
    ensure_setup();
    auto it = g_state.const_tables.find(table_id);
    if (it == g_state.const_tables.end())
        throw std::out_of_range("Constant table id " +
                                std::to_string(table_id) + " not found");
    const auto& tab = it->second;
    if (static_cast<int>(ct_ids.size()) != tab.cols)
        throw std::runtime_error(
            "MulConstantMatMulNew: expected " + std::to_string(tab.cols) +
            " ciphertexts, got " + std::to_string(ct_ids.size()) + ".");

    int target = 1 << 30;
    for (int id : ct_ids) target = std::min(target, level_of(g_state.ct(id)));
    if (tab.level != target)
        throw std::runtime_error(
            "MulConstantMatMulNew: table is at level " +
            std::to_string(tab.level) + " but the ciphertexts align to " +
            std::to_string(target) + "; re-encode the table there.");

    const size_t n = ct_ids.size();
    std::vector<std::unique_ptr<Ct>> tmp_cts(n);
    std::vector<const Ct*> cts(n);
    for (size_t i = 0; i < n; i++) {
        const Ct& c = g_state.ct(ct_ids[i]);
        if (c.HasRx())
            throw std::runtime_error(
                "MulConstantMatMulNew: operand is degree-2; relinearize "
                "first.");
        cts[i] = align_down(c, target, tmp_cts[i]);
    }

    const cheddar::NPInfo np = cts[0]->GetNP();
    const double ct_scale = cts[0]->GetScale();
    int num_slots = 0;
    for (size_t i = 0; i < n; i++) {
        if (cts[i]->GetNP().GetNumQ() != np.GetNumQ() ||
            cts[i]->GetNP().num_aux_ != np.num_aux_)
            throw std::runtime_error(
                "MulConstantMatMulNew: ciphertext NP mismatch after level "
                "alignment.");
        if (std::abs(cts[i]->GetScale() - ct_scale) > 1e-9 * ct_scale)
            throw std::runtime_error(
                "MulConstantMatMulNew: inconsistent ciphertext scales.");
        num_slots = std::max(num_slots, cts[i]->GetNumSlots());
    }

    const auto& param = g_state.context->param_;
    const int degree = param.degree_;
    const int q_size = np.GetNumQ() * degree;

    std::vector<const word*> hsrc(2 * n);
    std::vector<int> hextra(n);
    for (size_t i = 0; i < n; i++) {
        auto views = cts[i]->ConstViewVector();
        if (views.size() != 2)
            throw std::runtime_error(
                "MulConstantMatMulNew: expected degree-1 ciphertexts.");
        hsrc[2 * i] = views[0].data();
        hsrc[2 * i + 1] = views[1].data();
        hextra[i] = views[0].QSize() - q_size;
    }

    std::vector<std::unique_ptr<Ct>> outs(tab.rows);
    std::vector<word*> hdst(2 * tab.rows);
    for (int r = 0; r < tab.rows; r++) {
        outs[r] = std::make_unique<Ct>();
        outs[r]->RemoveRx();
        outs[r]->ModifyNP(np);
        outs[r]->SetScale(ct_scale * tab.scale);
        outs[r]->SetNumSlots(num_slots);
        auto views = outs[r]->ViewVector();
        hdst[2 * r] = views[0].data();
        hdst[2 * r + 1] = views[1].data();
    }

    // One H2D transfer for the three descriptor arrays: d*2 output
    // pointers, p*2 source pointers, p aux offsets.
    const size_t ptr_bytes = sizeof(void*) * (hdst.size() + hsrc.size());
    const size_t int_bytes = sizeof(int) * hextra.size();
    std::vector<unsigned char> host(ptr_bytes + int_bytes);
    std::memcpy(host.data(), hdst.data(), sizeof(void*) * hdst.size());
    std::memcpy(host.data() + sizeof(void*) * hdst.size(), hsrc.data(),
                sizeof(void*) * hsrc.size());
    std::memcpy(host.data() + ptr_bytes, hextra.data(), int_bytes);
    unsigned char* dev =
        static_cast<unsigned char*>(g_state.ensure_scratch(host.size()));
    cudaMemcpy(dev, host.data(), host.size(), cudaMemcpyHostToDevice);

    word* const* d_dst = reinterpret_cast<word* const*>(dev);
    const word* const* d_src = reinterpret_cast<const word* const*>(
        dev + sizeof(void*) * hdst.size());
    const int* d_extra = reinterpret_cast<const int*>(dev + ptr_bytes);

    const word* primes = param.GetPrimesPtr(np);
    const cheddar::make_signed_t<word>* inv_primes =
        param.GetInvPrimesPtr(np);
    if (degree % kMatMulBlock != 0)
        throw std::runtime_error(
            "one-pass PC-MM: ring degree must be a multiple of " +
            std::to_string(kMatMulBlock) + ".");
    const int grid = np.GetNumTotal() * degree / kMatMulBlock;
    // 16 accumulator pairs fit comfortably in registers for 32-bit
    // words; the 64-bit build halves the tile to stay off local memory.
    constexpr int kDTile = sizeof(word) == 4 ? 16 : 8;
    for (int row0 = 0; row0 < tab.rows; row0 += kDTile) {
        const int rows = std::min(kDTile, tab.rows - row0);
        ConstMatMulKernel<word, kDTile><<<grid, kMatMulBlock>>>(
            d_dst + 2 * row0, d_src, d_extra, tab.data.data(),
            tab.rows * tab.cols, tab.qsize, row0, rows, primes, inv_primes,
            np.GetNumQ(), param.log_degree_, tab.cols);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("MulConstantMatMulNew: ") +
                                 cudaGetErrorString(err));

    std::vector<int> ids(tab.rows);
    for (int r = 0; r < tab.rows; r++)
        ids[r] = g_state.put_ct(std::move(outs[r]));
    return ids;
}

// The slot-varying twin of ConstMatMulKernel. When several fields share
// one ciphertext each table entry is a full plaintext rather than a
// broadcast scalar, so the multipliers cannot be staged in shared
// memory -- but the one-pass structure still removes the d-fold re-read
// of the coordinate ciphertexts, which is the larger of the two streams
// whenever the ciphertexts outweigh the table.
template <typename W, int D>
__global__ void PlaintextMatMulKernel(
    W* const* __restrict__ dst_ptrs, const W* const* __restrict__ src_ptrs,
    const int* __restrict__ src_extra,
    const W* const* __restrict__ pt_ptrs, int pt_extra, int row0, int rows,
    const W* __restrict__ primes,
    const cheddar::make_signed_t<W>* __restrict__ inv_primes, int num_q_primes,
    int log_degree, int P) {
    using SW = cheddar::make_signed_t<W>;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int prime_index = i >> log_degree;
    const W prime = primes[prime_index];
    const SW inv_prime = inv_primes[prime_index];
    const bool aux = (prime_index >= num_q_primes);
    const int poff = aux ? pt_extra : 0;

    W acc[D][2];
#pragma unroll
    for (int r = 0; r < D; r++) {
        acc[r][0] = 0;
        acc[r][1] = 0;
    }

    for (int k = 0; k < P; k++) {
        const int off = aux ? src_extra[k] : 0;
        const W v0 = cheddar::basic::StreamingLoad(src_ptrs[2 * k] + i + off);
        const W v1 =
            cheddar::basic::StreamingLoad(src_ptrs[2 * k + 1] + i + off);
#pragma unroll
        for (int r = 0; r < D; r++) {
            if (r < rows) {
                const W c = cheddar::basic::StreamingLoad(
                    pt_ptrs[(row0 + r) * P + k] + i + poff);
                acc[r][0] = cheddar::basic::Add(
                    acc[r][0],
                    cheddar::basic::MultMontgomery(c, v0, prime, inv_prime),
                    prime);
                acc[r][1] = cheddar::basic::Add(
                    acc[r][1],
                    cheddar::basic::MultMontgomery(c, v1, prime, inv_prime),
                    prime);
            }
        }
    }

#pragma unroll
    for (int r = 0; r < D; r++) {
        if (r < rows) {
            dst_ptrs[2 * r][i] = acc[r][0];
            dst_ptrs[2 * r + 1][i] = acc[r][1];
        }
    }
}

// Pin a rows x cols plaintext table (row-major, entry (r, k) at
// pt_ids[r * cols + k]) for MulPlaintextMatMulNew. Only the pointers
// are recorded, so the plaintexts must outlive the table and stay at
// the level they were encoded at; delete and repack otherwise.
int PackPlaintextTableNew(std::vector<int> pt_ids, int rows) {
    ensure_setup();
    if (rows <= 0 || pt_ids.empty() ||
        pt_ids.size() % static_cast<size_t>(rows) != 0)
        throw std::runtime_error(
            "PackPlaintextTableNew: pt_ids must be a non-empty row-major "
            "rows x cols table.");
    const int n = static_cast<int>(pt_ids.size());
    const int cols = n / rows;

    const Pt& first = g_state.pt(pt_ids[0]);
    const int level = level_of(first);
    const double scale = first.GetScale();
    const int qsize = first.ConstView().QSize();
    int num_slots = 0;

    std::vector<const word*> hptr(n);
    for (int i = 0; i < n; i++) {
        const Pt& p = g_state.pt(pt_ids[i]);
        if (level_of(p) != level)
            throw std::runtime_error(
                "PackPlaintextTableNew: all plaintexts must share one level.");
        if (std::abs(p.GetScale() - scale) > 1e-9 * scale)
            throw std::runtime_error(
                "PackPlaintextTableNew: all plaintexts must share one scale.");
        auto view = p.ConstView();
        if (view.QSize() != qsize)
            throw std::runtime_error(
                "PackPlaintextTableNew: inconsistent plaintext layout.");
        num_slots = std::max(num_slots, p.GetNumSlots());
        hptr[i] = view.data();
    }

    const word** dptr = nullptr;
    if (cudaMalloc(&dptr, sizeof(const word*) * n) != cudaSuccess)
        throw std::runtime_error(
            "PackPlaintextTableNew: cudaMalloc for the pointer array failed.");
    cudaMemcpy(dptr, hptr.data(), sizeof(const word*) * n,
               cudaMemcpyHostToDevice);

    int id = g_state.next_pt_table_id++;
    g_state.pt_tables[id] = BackendState::StoredPtTable{
        dptr, rows, cols, qsize, level, scale, num_slots};
    return id;
}

void DeletePlaintextTable(int id) {
    auto it = g_state.pt_tables.find(id);
    if (it == g_state.pt_tables.end()) return;
    if (it->second.ptrs) cudaFree(it->second.ptrs);
    g_state.pt_tables.erase(it);
}

// The plaintext-table PC-MM, all d rows in one pass. Returns d new
// ciphertext ids; each output's scale is ct_scale * pt_scale, so the
// caller Rescales once -- same contract as MulPlaintextAccumNew.
std::vector<int> MulPlaintextMatMulNew(std::vector<int> ct_ids,
                                       int table_id) {
    ensure_setup();
    auto it = g_state.pt_tables.find(table_id);
    if (it == g_state.pt_tables.end())
        throw std::out_of_range("Plaintext table id " +
                                std::to_string(table_id) + " not found");
    const auto& tab = it->second;
    if (static_cast<int>(ct_ids.size()) != tab.cols)
        throw std::runtime_error(
            "MulPlaintextMatMulNew: expected " + std::to_string(tab.cols) +
            " ciphertexts, got " + std::to_string(ct_ids.size()) + ".");

    int target = 1 << 30;
    for (int id : ct_ids) target = std::min(target, level_of(g_state.ct(id)));
    if (tab.level != target)
        throw std::runtime_error(
            "MulPlaintextMatMulNew: table is at level " +
            std::to_string(tab.level) + " but the ciphertexts align to " +
            std::to_string(target) + "; re-encode the table there.");

    const size_t n = ct_ids.size();
    std::vector<std::unique_ptr<Ct>> tmp_cts(n);
    std::vector<const Ct*> cts(n);
    for (size_t i = 0; i < n; i++) {
        const Ct& c = g_state.ct(ct_ids[i]);
        if (c.HasRx())
            throw std::runtime_error(
                "MulPlaintextMatMulNew: operand is degree-2; relinearize "
                "first.");
        cts[i] = align_down(c, target, tmp_cts[i]);
    }

    const cheddar::NPInfo np = cts[0]->GetNP();
    const double ct_scale = cts[0]->GetScale();
    int num_slots = tab.num_slots;
    for (size_t i = 0; i < n; i++) {
        if (cts[i]->GetNP().GetNumQ() != np.GetNumQ() ||
            cts[i]->GetNP().num_aux_ != np.num_aux_)
            throw std::runtime_error(
                "MulPlaintextMatMulNew: ciphertext NP mismatch after level "
                "alignment.");
        if (std::abs(cts[i]->GetScale() - ct_scale) > 1e-9 * ct_scale)
            throw std::runtime_error(
                "MulPlaintextMatMulNew: inconsistent ciphertext scales.");
        num_slots = std::max(num_slots, cts[i]->GetNumSlots());
    }

    const auto& param = g_state.context->param_;
    const int degree = param.degree_;
    const int q_size = np.GetNumQ() * degree;

    std::vector<const word*> hsrc(2 * n);
    std::vector<int> hextra(n);
    for (size_t i = 0; i < n; i++) {
        auto views = cts[i]->ConstViewVector();
        if (views.size() != 2)
            throw std::runtime_error(
                "MulPlaintextMatMulNew: expected degree-1 ciphertexts.");
        hsrc[2 * i] = views[0].data();
        hsrc[2 * i + 1] = views[1].data();
        hextra[i] = views[0].QSize() - q_size;
    }

    std::vector<std::unique_ptr<Ct>> outs(tab.rows);
    std::vector<word*> hdst(2 * tab.rows);
    for (int r = 0; r < tab.rows; r++) {
        outs[r] = std::make_unique<Ct>();
        outs[r]->RemoveRx();
        outs[r]->ModifyNP(np);
        outs[r]->SetScale(ct_scale * tab.scale);
        outs[r]->SetNumSlots(num_slots);
        auto views = outs[r]->ViewVector();
        hdst[2 * r] = views[0].data();
        hdst[2 * r + 1] = views[1].data();
    }

    const size_t ptr_bytes = sizeof(void*) * (hdst.size() + hsrc.size());
    const size_t int_bytes = sizeof(int) * hextra.size();
    std::vector<unsigned char> host(ptr_bytes + int_bytes);
    std::memcpy(host.data(), hdst.data(), sizeof(void*) * hdst.size());
    std::memcpy(host.data() + sizeof(void*) * hdst.size(), hsrc.data(),
                sizeof(void*) * hsrc.size());
    std::memcpy(host.data() + ptr_bytes, hextra.data(), int_bytes);
    unsigned char* dev =
        static_cast<unsigned char*>(g_state.ensure_scratch(host.size()));
    cudaMemcpy(dev, host.data(), host.size(), cudaMemcpyHostToDevice);

    word* const* d_dst = reinterpret_cast<word* const*>(dev);
    const word* const* d_src = reinterpret_cast<const word* const*>(
        dev + sizeof(void*) * hdst.size());
    const int* d_extra = reinterpret_cast<const int*>(dev + ptr_bytes);

    const word* primes = param.GetPrimesPtr(np);
    const cheddar::make_signed_t<word>* inv_primes =
        param.GetInvPrimesPtr(np);
    if (degree % kMatMulBlock != 0)
        throw std::runtime_error(
            "one-pass PC-MM: ring degree must be a multiple of " +
            std::to_string(kMatMulBlock) + ".");
    const int grid = np.GetNumTotal() * degree / kMatMulBlock;
    constexpr int kDTile = sizeof(word) == 4 ? 16 : 8;
    for (int row0 = 0; row0 < tab.rows; row0 += kDTile) {
        const int rows = std::min(kDTile, tab.rows - row0);
        PlaintextMatMulKernel<word, kDTile><<<grid, kMatMulBlock>>>(
            d_dst + 2 * row0, d_src, d_extra, tab.ptrs,
            tab.qsize - q_size, row0, rows, primes, inv_primes,
            np.GetNumQ(), param.log_degree_, tab.cols);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("MulPlaintextMatMulNew: ") +
                                 cudaGetErrorString(err));

    std::vector<int> ids(tab.rows);
    for (int r = 0; r < tab.rows; r++)
        ids[r] = g_state.put_ct(std::move(outs[r]));
    return ids;
}

// ---------------------------------------------------------------------------
// Double-hoisted linear transform (native HoistHandler)
// ---------------------------------------------------------------------------
// Wraps cheddar's HoistHandler (extension/Hoist.h, the Bossuat et al.
// EUROCRYPT'21 double-hoisting BSGS): baby-step rotations share one ModUp
// AND stay fused (BSFusedKeyMult: KeyMult+MAC+automorphism in one kernel
// when beta fits), giant-step accumulation runs GSFusedPAccum +
// PermuteAccum, and the final ModDown folds the rescale in for free
// (EvaluateFinalModDown -> ModDownAndRescale). The PlainHoistMap is built
// directly from Orion's diagonals -- LinearTransform's square/power-of-two
// factory constraint does NOT apply to HoistHandler itself.
//
// Conventions (match the Python BSGS path exactly): with HRot(ct,k)[i] =
// ct[i+k], Evaluate computes res[i] = sum_gs sum_bs pt[gs][bs][i+gs] *
// input[i+gs+bs]; we need res[i] = sum_k diag_k[i]*input[i+k] for
// k = gs+bs, so pt[gs][bs] = roll(diag_k, +gs) -- the same giant-step
// pre-roll GenerateLinearTransform bakes into its cached plaintexts.
//
// The output is ALREADY rescaled (level pt_level-1, scale
// input_scale*pt_scale/q[pt_level]) -- callers must skip their usual
// post-LT rescale (bindings.py advertises this via lt_output_prerescaled).
//
// Key set: read back from the constructed handler via
// AddRequiredRotations, never recomputed externally -- the constructor
// can swap bs/gs for single-giant-group maps (kOptimizeAutomorphism),
// changing which distances it needs.

int GenerateHoistLT(std::vector<int> diags_idxs, std::vector<double> diags_data,
                    int level, int bs) {
    ensure_setup();
    const size_t num_diags = diags_idxs.size();
    if (num_diags == 0 || diags_data.size() % num_diags != 0)
        throw std::runtime_error(
            "GenerateHoistLT: empty or ragged diagonal data.");
    const size_t width = diags_data.size() / num_diags;
    if (bs < 1) throw std::runtime_error("GenerateHoistLT: bs must be >= 1.");

    cheddar::PlainHoistMap hoist_map;
    for (size_t i = 0; i < num_diags; i++) {
        int k = diags_idxs[i];
        if (k < 0 || static_cast<size_t>(k) >= width)
            throw std::runtime_error(
                "GenerateHoistLT: diagonal index out of [0, width).");
        int b = k % bs;
        int g = k - b;
        cheddar::Message msg(width, Complex(0.0, 0.0));
        const double* src = diags_data.data() + i * width;
        // pt[g][b][j'] = diag_k[j' - g]  (roll by +g)
        for (size_t j = 0; j < width; j++)
            msg[(j + g) % width] = Complex(src[j], 0.0);
        hoist_map[g].emplace(b, std::move(msg));
    }

    BackendState::HoistLT entry;
    entry.handler = std::make_unique<cheddar::HoistHandler<word>>(
        g_state.context, hoist_map, level, canonical_scale(level));
    entry.pt_level = level;
    int id = g_state.next_hoist_lt_id++;
    g_state.hoist_lts[id] = std::move(entry);
    return id;
}

std::vector<int> GetHoistLTRotationKeys(int hlt_id) {
    auto it = g_state.hoist_lts.find(hlt_id);
    if (it == g_state.hoist_lts.end())
        throw std::out_of_range("HoistLT id " + std::to_string(hlt_id) +
                                " not found");
    cheddar::EvkRequest req;
    it->second.handler->AddRequiredRotations(req, /*min_ks=*/false);
    std::vector<int> keys;
    keys.reserve(req.size());
    for (const auto& [rot, lvl] : req) keys.push_back(rot);
    return keys;
}

int EvaluateHoistLT(int hlt_id, int ct_id) {
    ensure_keys();
    auto it = g_state.hoist_lts.find(hlt_id);
    if (it == g_state.hoist_lts.end())
        throw std::out_of_range("HoistLT id " + std::to_string(hlt_id) +
                                " not found");
    const Ct& c = g_state.ct(ct_id);
    if (c.HasRx())
        throw std::runtime_error(
            "EvaluateHoistLT: input is degree-2; relinearize first.");
    std::unique_ptr<Ct> tmp;
    const Ct* input = align_down(c, it->second.pt_level, tmp);
    if (level_of(*input) != it->second.pt_level)
        throw std::runtime_error(
            "EvaluateHoistLT: input level below the transform's pt_level.");
    auto out = std::make_unique<Ct>();
    // Shared max-level key map: BSFusedKeyMult handles key level >=
    // input level fine (key_extra/num_accum_offset alignment) once the
    // fused-dispatch-beta1 patch is in -- verified empirically at
    // level 3 with max-level keys. No per-level key material needed.
    it->second.handler->Evaluate(g_state.context, *out, *input,
                                 g_state.iface->GetEvkMap(),
                                 /*min_ks=*/false);
    return g_state.put_ct(std::move(out));
}

void DeleteHoistLT(int hlt_id) { g_state.hoist_lts.erase(hlt_id); }

int AddPlaintextNew(int ct_id, int pt_id) {
    ensure_setup();
    const Ct& c = g_state.ct(ct_id);
    const Pt& p = g_state.pt(pt_id);
    std::unique_ptr<Ct> tc;
    std::unique_ptr<Pt> tp;
    const Ct* uc = align_down(c, std::min(level_of(c), level_of(p)), tc);
    // Cheddar asserts equal scales on ct+pt. Lattigo semantics: the pt
    // is meant to represent its values at the ct's scale -- re-encode
    // when they differ (offline path).
    const Pt* up = &p;
    double diff = std::abs(uc->GetScale() - p.GetScale());
    if (level_of(p) != level_of(*uc) || !(diff < 1e-12 * uc->GetScale()))
        up = align_pt(*uc, p, tp, uc->GetScale());
    auto out = std::make_unique<Ct>();
    g_state.context->Add(*out, *uc, *up);
    return g_state.put_ct(std::move(out));
}

int AddPlaintext(int ct_id, int pt_id) {
    replace_ct(ct_id, AddPlaintextNew(ct_id, pt_id));
    return ct_id;
}

int SubPlaintextNew(int ct_id, int pt_id) {
    ensure_setup();
    const Ct& c = g_state.ct(ct_id);
    const Pt& p = g_state.pt(pt_id);
    std::unique_ptr<Ct> tc;
    std::unique_ptr<Pt> tp;
    const Ct* uc = align_down(c, std::min(level_of(c), level_of(p)), tc);
    const Pt* up = &p;
    double diff = std::abs(uc->GetScale() - p.GetScale());
    if (level_of(p) != level_of(*uc) || !(diff < 1e-12 * uc->GetScale()))
        up = align_pt(*uc, p, tp, uc->GetScale());
    auto out = std::make_unique<Ct>();
    g_state.context->Sub(*out, *uc, *up);
    return g_state.put_ct(std::move(out));
}

int SubPlaintext(int ct_id, int pt_id) {
    replace_ct(ct_id, SubPlaintextNew(ct_id, pt_id));
    return ct_id;
}

// ---------------------------------------------------------------------------
// ct - scalar arithmetic
// ---------------------------------------------------------------------------
// Add/Sub encode the constant at the ciphertext's own scale (Mult's result
// scale is ct.scale * const.scale, so this keeps Add/Sub scale-neutral, no
// rescale needed -- matching evaluator.py, which never rescales after
// add_scalar/sub_scalar). MulScalarFloat encodes at the ciphertext's scale
// too, so the product's scale is squared and needs one Rescale (evaluator.py
// always rescales after the float path). MulScalarInt encodes the constant
// at scale=1.0 instead, so the product keeps the ciphertext's original
// scale unchanged -- matching evaluator.py, which does NOT rescale after
// the int path.

cheddar::Constant<word> encode_const(double number, int level, double scale) {
    cheddar::Constant<word> c;
    g_state.context->encoder_.EncodeConstant(c, level, scale, number);
    return c;
}

int AddScalarNew(int ct_id, double scalar) {
    ensure_setup();
    const Ct& c = g_state.ct(ct_id);
    auto k = encode_const(scalar, level_of(c), c.GetScale());
    auto out = std::make_unique<Ct>();
    g_state.context->Add(*out, c, k);
    return g_state.put_ct(std::move(out));
}

int AddScalar(int ct_id, double scalar) {
    replace_ct(ct_id, AddScalarNew(ct_id, scalar));
    return ct_id;
}

int SubScalarNew(int ct_id, double scalar) {
    ensure_setup();
    const Ct& c = g_state.ct(ct_id);
    auto k = encode_const(scalar, level_of(c), c.GetScale());
    auto out = std::make_unique<Ct>();
    g_state.context->Sub(*out, c, k);
    return g_state.put_ct(std::move(out));
}

int SubScalar(int ct_id, double scalar) {
    replace_ct(ct_id, SubScalarNew(ct_id, scalar));
    return ct_id;
}

int MulScalarFloatNew(int ct_id, double scalar) {
    ensure_setup();
    const Ct& c = g_state.ct(ct_id);
    auto k = encode_const(scalar, level_of(c), c.GetScale());
    auto out = std::make_unique<Ct>();
    g_state.context->Mult(*out, c, k);
    return g_state.put_ct(std::move(out));
}

int MulScalarFloat(int ct_id, double scalar) {
    replace_ct(ct_id, MulScalarFloatNew(ct_id, scalar));
    return ct_id;
}

int MulScalarIntNew(int ct_id, int scalar) {
    ensure_setup();
    const Ct& c = g_state.ct(ct_id);
    auto k = encode_const(static_cast<double>(scalar), level_of(c), 1.0);
    auto out = std::make_unique<Ct>();
    g_state.context->Mult(*out, c, k);
    return g_state.put_ct(std::move(out));
}

int MulScalarInt(int ct_id, int scalar) {
    replace_ct(ct_id, MulScalarIntNew(ct_id, scalar));
    return ct_id;
}

// ---------------------------------------------------------------------------
// Negate
// ---------------------------------------------------------------------------

int Negate(int ct_id) {
    ensure_setup();
    auto out = std::make_unique<Ct>();
    g_state.context->Neg(*out, g_state.ct(ct_id));
    return g_state.put_ct(std::move(out));
}

// Homomorphic complex conjugation (slot-wise conj). The conjugation key
// is prepared unconditionally by UserInterface's constructor
// (PrepareBasicEvks), so unlike rotations there is no per-distance key
// to add first.
int ConjugateNew(int ct_id) {
    ensure_keys();
    const Ct& in = g_state.ct(ct_id);
    if (in.HasRx())
        throw std::runtime_error(
            "ConjugateNew: input is degree-2; relinearize first.");
    auto out = std::make_unique<Ct>();
    g_state.context->HConj(*out, in, g_state.iface->GetConjugationKey());
    return g_state.put_ct(std::move(out));
}

int Conjugate(int ct_id) {
    replace_ct(ct_id, ConjugateNew(ct_id));
    return ct_id;
}

// Slot-wise multiply by sqrt(-1). Key-free (a plain polynomial
// permutation), consumes no level.
int MulImaginaryUnitNew(int ct_id) {
    ensure_setup();
    auto out = std::make_unique<Ct>();
    g_state.context->MultImaginaryUnit(*out, g_state.ct(ct_id));
    return g_state.put_ct(std::move(out));
}

int MulImaginaryUnit(int ct_id) {
    replace_ct(ct_id, MulImaginaryUnitNew(ct_id));
    return ct_id;
}

// ---------------------------------------------------------------------------
// Rescale
// ---------------------------------------------------------------------------

int RescaleNew(int ct_id) {
    ensure_setup();
    const Ct& c = g_state.ct(ct_id);
    if (level_of(c) <= 0)
        throw std::runtime_error("RescaleNew: ciphertext already at level 0.");
    auto out = std::make_unique<Ct>();
    g_state.context->Rescale(*out, c);  // native HasRx() path for degree-2
    return g_state.put_ct(std::move(out));
}

int Rescale(int ct_id) {
    replace_ct(ct_id, RescaleNew(ct_id));
    return ct_id;
}

// ---------------------------------------------------------------------------
// Rotations
// ---------------------------------------------------------------------------
// Convention check (unittest/BasicTest.cpp HRot): res[i] = a[i + k], the
// same left-rotation lattigo uses. Distances are normalized to
// [0, slots) -- rotating by -k equals rotating by slots - k.

int normalize_shift(int k) {
    int slots = g_state.slot_count;
    int kn = k % slots;
    if (kn < 0) kn += slots;
    return kn;
}

void AddRotationKey(int k) {
    ensure_keys();
    int kn = normalize_shift(k);
    if (kn == 0 || g_state.rot_keys.count(kn)) return;
    // Always pass an explicit level, matching upstream's own usage
    // (BasicTest / the EvkRequest path). Do NOT rely on the documented
    // "-1 -> max" default: in the implementation -1 is the internal
    // sentinel for the dense-to-sparse key shape (GetNPForEvk(-1)),
    // which yields a beta-1 key and "Beta mismatch" at HRot.
    g_state.iface->PrepareRotationKey(kn, g_state.max_level);
    g_state.rot_keys.insert(kn);
}

int RotateNew(int ct_id, int k) {
    ensure_keys();
    int kn = normalize_shift(k);
    auto out = std::make_unique<Ct>();
    if (kn == 0) {
        // rot by 0 == clone; kernels use rot_batch(ct, [0]) as a cheap
        // clone (see bmm3_cipher.py).
        g_state.context->Copy(*out, g_state.ct(ct_id));
        return g_state.put_ct(std::move(out));
    }
    if (!g_state.rot_keys.count(kn))
        AddRotationKey(kn);
    g_state.context->HRot(*out, g_state.ct(ct_id),
                          g_state.iface->GetRotationKey(kn), kn);
    return g_state.put_ct(std::move(out));
}

// Single-hoisted N-way rotation: pay the ModUp of the input's ax (and
// the PseudoModUp of bx) ONCE for the whole batch, then per shift only
// KeyMult + MAC + ModDown + Permute. This mirrors the non-fused branch
// of Cheddar's own HoistHandler::EvaluateBabyStep (src/extension/
// Hoist.cu) followed by the ModDown tail of Context::MultKey -- built
// entirely from public Context members (MultKeyNoModDown,
// mod_switch_handlers_, p_prod_, elem_handler_, Permute), so this
// builds against plain upstream Cheddar with no patches.
//
// Semantics are identical to the per-shift HRot loop: HRot ==
// MultKey (ModUp+KeyMult+ModDown) + Permute with the same per-distance
// keys; hoisting only reorders when the shift-independent ModUp runs.
std::vector<int> RotateBatchHoistedNew(const Ct& input,
                                       const std::vector<int>& kns) {
    using Dv = cheddar::DeviceVector<word>;
    const auto& param = *g_state.param;
    auto& ctx = *g_state.context;

    if (input.HasRx())
        throw std::runtime_error(
            "RotateBatchNew: input is degree-2; relinearize first.");
    cheddar::NPInfo np = input.GetNP();
    if (np.num_aux_ != 0)
        throw std::runtime_error("RotateBatchNew: input has aux primes.");

    const int level = param.NPToLevel(np);
    const int num_q = np.GetNumQ();
    const int alpha = param.alpha_;
    const int degree = param.degree_;
    const int prime_offset = param.GetMaxNumTer() - np.num_ter_;
    const int beta = (num_q + prime_offset + alpha - 1) / alpha;
    const auto& mod_switcher = ctx.mod_switch_handlers_.at(level);

    // 1. ModUp of ax -- the expensive shift-independent half, done once.
    std::vector<Dv> modup;
    std::vector<cheddar::DvView<word>> modup_view;
    for (int i = 0; i < beta; i++) {
        modup.emplace_back((num_q + alpha) * degree);
        modup_view.push_back(modup[i].View(alpha * degree));
    }
    mod_switcher.ModUp(modup_view, input.AxConstView());

    // 2. P*bx once (added into each accumulator before its ModDown,
    //    replacing the per-call CAccum of the plain MultKey path).
    cheddar::DvConstView<word> p_prod_view(ctx.p_prod_.data() + prime_offset,
                                           num_q);
    Dv bx_pseudo(num_q * degree);
    cheddar::DvView<word> bx_pseudo_view = bx_pseudo.View();
    mod_switcher.PseudoModUp(bx_pseudo_view, input.BxConstView(), p_prod_view);
    bx_pseudo.ZeroExtend(alpha * degree);

    Ct accum;
    Ct dropped;
    std::vector<int> out_ids;
    out_ids.reserve(kns.size());
    for (int k : kns) {
        const auto& key = g_state.iface->GetRotationKey(k);
        // Shift-dependent half: digits x key_k, still in extended basis.
        ctx.MultKeyNoModDown(accum, modup, input, key);
        // MAC: accum.bx (q-part) += P*bx.
        cheddar::DvView<word> accum_bx_q_view(accum.bx_.data(),
                                              num_q * degree, 0);
        std::vector<cheddar::DvView<word>> mac_res = {accum_bx_q_view};
        std::vector<cheddar::DvConstView<word>> mac_a = {accum_bx_q_view};
        ctx.elem_handler_.Add(mac_res, np, mac_a, {bx_pseudo.ConstView()});
        // ModDown back to the q basis (the MultKey tail).
        dropped.RemoveRx();
        dropped.ModifyNP(np);
        dropped.SetScale(input.GetScale());
        dropped.SetNumSlots(input.GetNumSlots());
        auto dropped_bx = dropped.BxView();
        auto dropped_ax = dropped.AxView();
        mod_switcher.ModDown(dropped_bx, accum.BxConstView());
        mod_switcher.ModDown(dropped_ax, accum.AxConstView());
        // Automorphism last, matching HRot = MultKey + Permute.
        auto out = std::make_unique<Ct>();
        ctx.Permute(*out, dropped, k);
        out_ids.push_back(g_state.put_ct(std::move(out)));
    }
    return out_ids;
}

// hoist_mode: 0 = none (per-shift HRot loop), 1 = single (shared ModUp
// across the batch).
std::vector<int> RotateBatchNew(int ct_id, std::vector<int> shifts,
                                int hoist_mode) {
    ensure_keys();
    if (hoist_mode != 1 || shifts.size() < 2) {
        std::vector<int> out_ids;
        out_ids.reserve(shifts.size());
        for (int k : shifts) out_ids.push_back(RotateNew(ct_id, k));
        return out_ids;
    }

    // Normalize shifts; peel off zeros (clones) so the hoisted core only
    // sees real rotations, then reassemble in caller order.
    const Ct& input = g_state.ct(ct_id);
    std::vector<int> kns;
    kns.reserve(shifts.size());
    for (int k : shifts) {
        int kn = normalize_shift(k);
        if (kn != 0) {
            AddRotationKey(kn);
            kns.push_back(kn);
        }
    }

    std::map<int, std::vector<int>> hoisted_ids;  // kn -> ids (dup-safe)
    if (!kns.empty()) {
        std::vector<int> ids = RotateBatchHoistedNew(input, kns);
        for (size_t i = 0; i < kns.size(); ++i)
            hoisted_ids[kns[i]].push_back(ids[i]);
    }

    std::vector<int> out_ids;
    out_ids.reserve(shifts.size());
    for (int k : shifts) {
        int kn = normalize_shift(k);
        if (kn == 0) {
            auto out = std::make_unique<Ct>();
            g_state.context->Copy(*out, input);
            out_ids.push_back(g_state.put_ct(std::move(out)));
        } else {
            auto& ids = hoisted_ids[kn];
            out_ids.push_back(ids.back());
            if (ids.size() > 1) ids.pop_back();
        }
    }
    return out_ids;
}

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------
// Cheddar exposes bootstrap as raw building blocks (BootContext) rather
// than lattigo/desilo's single opaque call, so this orchestrates the
// sequence documented in BootContext.h: prepare EvalMod once, prepare
// the special FFT per slot count, discover+generate the rotation keys
// the circuit needs (folded into the same EvkMap/UserInterface every
// other op already shares), then Boot().
//
// BootContext is built against the scheme's existing param_ (same NTT/
// RNS machinery as the regular Context), so ciphertexts created via
// g_state.context interoperate with it directly -- no re-encryption or
// separate key material needed, matching lattigo's bootstrapping.
// Evaluator being a wrapper around the same scheme.Params/SecretKey.

void NewBootstrapper(int /*num_cts_levels*/, int /*num_stc_levels*/,
                     int /*log_message_ratio*/, int slots) {
    ensure_keys();
    // Cheddar's EvalSpecialFFT hard-asserts num_slots >= 256 (its FFT
    // tables aren't built for anything sparser). Narrow layers -- e.g.
    // HELRM's 16-wide embedding_features bootstrap -- request fewer slots
    // than that. Clamp what we *prepare* to 256; BootContext::Boot()
    // already rounds a ciphertext's real (smaller) GetNumSlots() up to
    // the next slot count it finds prepared (GetBootEnabledNumSlots in
    // BootContext.cpp), so preparing 256 here is sufficient for Boot() to
    // pick it up transparently for any narrower ciphertext. The unused
    // slots beyond the ciphertext's real width are already zero, so this
    // only spends extra (wasted) FFT work, not incorrect results.
    slots = std::max(slots, 256);
    // boot_context is built at setup_scheme time (as the shared context)
    // whenever boot_params are present. If it is missing, the scheme was
    // created without boot level knobs -- surface that clearly rather than
    // segfaulting on a null context.
    if (!g_state.boot_context)
        throw std::runtime_error(
            "Cheddar: scheme was not built with bootstrap parameters. Set "
            "boot_params num_cts_levels/num_stc_levels in the Orion config.");
    if (!g_state.eval_mod_prepared) {
        g_state.boot_context->PrepareEvalMod();
        g_state.eval_mod_prepared = true;
    }
    if (g_state.boot_prepared_slots.count(slots)) return;

    // ORION_CHEDDAR_BOOT_EVICT=1: keep at most one slot count's
    // CoeffToSlot/SlotToCoeff data (BootContext's eval_fft_) resident at a
    // time, evicting every other prepared slot count before preparing this
    // one. Rotation keys are never evicted here (they live in the shared
    // EvkMap, slot-count-independent), so this never frees a key another
    // slot count still needs.
    // Orion's placement generates all of a network's boot slot counts
    // during compile(), before inference starts, so without this a network
    // needing N slot counts keeps all N resident simultaneously -- fine
    // for small N, but ResNet20's 4 distinct slot counts (32768 lazily,
    // plus the 3 placement generates) is enough to OOM a 24GB card once
    // num_stc_levels shrinks (bigger StC BSGS blocks -> bigger eval_fft_
    // per slot count). Safe for any network whose slot count only shrinks
    // over the forward pass (true for ResNet-style feedforward nets); a
    // network that revisits an evicted slot count just regenerates it
    // (extra latency, not incorrect -- PrepareEvalSpecialFFT is
    // idempotent/deterministic).
    static const bool evict_boot_circuits = [] {
        const char *v = std::getenv("ORION_CHEDDAR_BOOT_EVICT");
        return v && std::string(v) != "" && std::string(v) != "0";
    }();
    if (evict_boot_circuits) {
        for (int prev : g_state.boot_prepared_slots)
            g_state.boot_context->RemoveEvalSpecialFFT(prev);
        g_state.boot_prepared_slots.clear();
    }

    g_state.boot_context->PrepareEvalSpecialFFT(slots);

    // min_ks=false: generate the full rotation-key set for the fastest
    // Boot execution. With the single-context design above, full-slot
    // LogN=16 boot peaks ~15GB (matching upstream's own boot_test), so
    // the larger key set fits comfortably on a 24GB card. Boot() below
    // must use the same min_ks value the keys were generated for.
    cheddar::EvkRequest req;
    g_state.boot_context->AddRequiredRotations(req, slots, g_state.boot_min_ks);
    g_state.iface->PrepareRotationKey(req);

    g_state.boot_prepared_slots.insert(slots);
}

int Bootstrap(int ct_id, int slots) {
    ensure_keys();
    // Orion's placement generates bootstrappers for the slot counts it
    // decides on, but runtime ops can request a count that was never
    // placed (observed: ResNet20's conv1 requests logslots-15 when
    // placement only generated 14/13/12). Unlike desilo's slot-agnostic
    // key, Cheddar's FFT tables + rotation keys are per-slot-count, so
    // prepare on demand here rather than surfacing the placement gap as
    // a crash. NewBootstrapper is idempotent (no-ops if already
    // prepared) and throws its own clear error if boot_context is
    // missing entirely.
    //
    // Deliberately ignore the caller's `slots` argument for this check
    // and use the ciphertext's own GetNumSlots() instead. Orion's
    // Python-side "slots" is a heuristic computed independently from
    // tensor shape metadata (see tensors.py's bootstrap()); it can
    // diverge from what the native ciphertext actually carries once
    // enough ops have propagated through Cheddar's own NumSlots tracking
    // (see e.g. EvalPoly.cpp/Hoist.cu taking Max() of operand NumSlots).
    // BootContext::Boot() itself only ever looks at input.GetNumSlots()
    // (never the slots this wrapper receives), so preparing based on
    // anything else can silently miss the real requirement -- observed
    // on ResNet: Python requested slots=16384 (already prepared), but
    // the ciphertext's actual GetNumSlots() was 32768 (never prepared),
    // and Boot() crashed on that real value with no lazy-prepare chance.
    int actual_slots = g_state.ct(ct_id).GetNumSlots();
    if (!g_state.boot_prepared_slots.count(actual_slots))
        NewBootstrapper(0, 0, 0, actual_slots);

    // Scale snapping. Cheddar's Boot derives its EvalMod scaleup from the
    // scheme's fixed base_scale, i.e. it assumes the input sits at exactly
    // base_scale. But exact-prime rescaling drifts the scale away from
    // base_scale -- and because MulScalar-then-Rescale squares the scale,
    // the per-prime offset compounds, reaching ~1% after a dozen rescales.
    // Bootstrap precision is empirically capped near
    // -log2(|scale/base_scale - 1|) (a 1% drift caps it at ~7 bits vs the
    // ~19 achievable on an undrifted input). So: snap the input to
    // base_scale with a pure scale relabel (which rescales the
    // *represented* message by r = scale/base_scale), boot the snapped
    // copy, then divide the result by r to recover the true message. Costs
    // one level for the correction multiply; a fresh, undrifted input
    // (r == 1, e.g. a bare encode) skips the whole dance.
    const double base = g_state.default_scale;
    const double r = g_state.ct(ct_id).GetScale() / base;

    if (std::abs(r - 1.0) <= 1e-9) {
        auto out = std::make_unique<Ct>();
        g_state.boot_context->Boot(*out, g_state.ct(ct_id),
                                   g_state.iface->GetEvkMap(), g_state.boot_min_ks);
        return g_state.put_ct(std::move(out));
    }

    Ct snapped;
    g_state.context->Copy(snapped, g_state.ct(ct_id));
    snapped.SetScale(base);

    auto booted = std::make_unique<Ct>();
    g_state.boot_context->Boot(*booted, snapped,
                               g_state.iface->GetEvkMap(), g_state.boot_min_ks);
    int boot_id = g_state.put_ct(std::move(booted));

    // Undo the r factor: MulScalarFloatNew encodes 1/r at the (base) scale
    // and RescaleNew brings the squared scale back down -- same pattern the
    // scalar-mul path uses elsewhere, leaving a clean ~base_scale result.
    int scaled = MulScalarFloatNew(boot_id, 1.0 / r);
    g_state.ciphertexts.erase(boot_id);
    int corrected = RescaleNew(scaled);
    g_state.ciphertexts.erase(scaled);
    return corrected;
}

// ---------------------------------------------------------------------------
// Polynomial evaluation (native EvalPoly)
// ---------------------------------------------------------------------------
// Cheddar's own EvalPoly (extension/EvalPoly.h) already implements a
// log-depth (BSGS-tree) polynomial evaluator -- the same one BootContext
// uses internally for EvalMod. bindings.py's GenerateMonomial /
// GenerateChebyshev previously only fed a pure-Python Horner's-method
// fallback (O(degree) levels: one Ct-Ct multiply + rescale per
// coefficient), which silently diverged from the ceil(log2(degree+1))
// depth orion/nn/activation.py assumes when placing bootstraps -- fine
// for the tiny (degree <= 3) iterative-sign polynomials, but the
// composite Sign/ReLU minimax polynomials default to degrees [15, 15, 27],
// consuming 15/15/27 levels via Horner vs. the 4/4/5 EvalPoly (and the
// depth model) actually expect -- exceeding the entire level budget
// (l_eff=10) inside a single polynomial evaluation, making the network
// unplaceable no matter how bootstraps are scheduled. EvalPoly requires
// degree >= 2 (asserts otherwise), so bindings.py keeps the Horner path
// for degree < 2.
int EvaluatePolynomialNative(int ct_id, std::vector<double> coeffs,
                             double target_scale, bool chebyshev) {
    ensure_keys();
    const Ct& c = g_state.ct(ct_id);
    if (c.HasRx())
        throw std::runtime_error(
            "EvaluatePolynomialNative: relinearize before evaluating a "
            "polynomial.");

    // Explicit-chain mode: a nominal (2^LogScale) target would strand the
    // output off the canonical trajectory by up to the chain's per-level
    // wobble (~2^0.1), far beyond the add-path drift tolerance. The output
    // level is input_level - ceil(log2(degree+1)) (EvalPoly's BSGS-tree
    // depth -- the same model orion/nn/activation.py schedules with);
    // snap the target to that level's canonical scale and verify the
    // depth model held afterwards.
    int predicted_level = -1;
    bool snapped = false;
    if (g_state.canonical_scales) {
        int degree = static_cast<int>(coeffs.size()) - 1;
        int depth = static_cast<int>(std::ceil(std::log2(degree + 1)));
        predicted_level = level_of(c) - depth;
        if (predicted_level < 0)
            throw std::runtime_error(
                "EvaluatePolynomialNative: not enough levels for degree.");
        // Callers pass either the nominal default scale or the output
        // level's (rounded) canonical scale -- snap both onto the exact
        // canonical value, same rationale as Encode above.
        double canon = canonical_scale(predicted_level);
        if (std::abs(target_scale - canon) < 0.15 * canon ||
            std::abs(target_scale - g_state.default_scale) <
                0.15 * g_state.default_scale) {
            snapped = true;
            target_scale = canon;
        }
    }

    // `chebyshev` interprets coeffs as sum_j c_j * T_j(x) and evaluates
    // in Chebyshev basis natively (same BSGS tree/depth model). This is
    // the only numerically viable route for high-degree approximations
    // (cheb2poly blows up past degree ~30).
    cheddar::EvalPoly<word> poly(coeffs, level_of(c), c.GetScale(),
                                 target_scale, chebyshev);
    poly.Compile(g_state.context);

    auto out = std::make_unique<Ct>();
    poly.Evaluate(g_state.context, *out, c,
                 g_state.iface->GetMultiplicationKey());
    if (snapped && level_of(*out) != predicted_level)
        throw std::runtime_error(
            "EvaluatePolynomialNative: output level " +
            std::to_string(level_of(*out)) + " != predicted " +
            std::to_string(predicted_level) +
            "; canonical target scale was chosen for the wrong level.");
    return g_state.put_ct(std::move(out));
}

void DeleteBootstrappers() {
    // No-op: the BootContext is the scheme's single shared context, so it
    // cannot be torn down independently -- doing so would destroy the
    // context regular ops still use (or, if another ref keeps it alive,
    // leave boot_prepared_slots inconsistent with the still-live
    // precompute). All boot state is released by delete_scheme() at scheme
    // teardown. This is also generation-safe: a stale bootstrapper
    // finalizer running after a new scheme is set up won't disturb it.
}

// Per-level rescale divisors, indexed by level 0..default_encryption_level
// (index 0 holds base_scale as a placeholder -- there is no rescale from
// level 0). This is what Orion's errorless-rescale idiom needs from
// GetModuliChain: encode a plaintext at exactly the amount the next
// rescale divides by, so mult-then-rescale returns the ciphertext to its
// prior scale. On flat chains each entry is just that level's prime; on
// rational-rescaling chains it is the (possibly non-integer) ratio of
// consecutive level bases, which no prime list can express.
std::vector<double> GetRescalePrimeProds() {
    ensure_setup();
    const int top = g_state.param->default_encryption_level_;
    std::vector<double> out;
    out.reserve(top + 1);
    out.push_back(g_state.param->base_scale_);
    for (int l = 1; l <= top; ++l)
        out.push_back(g_state.param->GetRescalePrimeProd(l));
    return out;
}

// Canonical per-level scale trajectory (diagnostics / scale planning).
std::vector<double> GetCanonicalScales() {
    ensure_setup();
    const int top = g_state.param->default_encryption_level_;
    std::vector<double> out;
    out.reserve(top + 1);
    for (int l = 0; l <= top; ++l)
        out.push_back(g_state.param->GetScale(l));
    return out;
}

// Number of levels EvalMod consumes in the boot circuit. Fixed by the
// vendored BootParameter's mod_coefficients_ table + num_double_angle_
// (independent of max_level / num_cts / num_stc), so Python can call it
// once to size the reserved boot-prime region without duplicating the
// magic constant. Constructed with dummy level args since
// GetNumEvalModLevels() ignores them.
int BootNumEvalModLevels() {
    return cheddar::BootParameter(/*max_level=*/64, /*num_cts=*/1,
                                  /*num_stc=*/1).GetNumEvalModLevels();
}

// ---------------------------------------------------------------------------
// Lifecycle: ID deletion
// ---------------------------------------------------------------------------

void DeleteCiphertext(int ct_id) { g_state.ciphertexts.erase(ct_id); }
void DeletePlaintext(int pt_id) { g_state.plaintexts.erase(pt_id); }

// ---------------------------------------------------------------------------
// Metadata getters
// ---------------------------------------------------------------------------

// Exact resident evaluation-key memory (MB): every key in the
// UserInterface's EvkMap (per-distance rotation keys + the 4 basic
// keys), summed over their device buffers. Matches Negar's separately
// reported CPU key-memory number; benches call it after warmup, when
// the shape's full key set exists.
double GetKeyMemoryMB() {
    ensure_keys();
    size_t words = 0;
    for (const auto& [idx, key] : g_state.iface->GetEvkMap()) {
        for (const auto& dv : key.bx_) words += dv.size();
        for (const auto& dv : key.ax_) words += dv.size();
    }
    return static_cast<double>(words) * sizeof(word) / (1024.0 * 1024.0);
}

// Peak bytes ever allocated through the RMM pool backing the current
// scheme (see MemoryPool.h) -- includes keys, ciphertexts and every other
// device buffer, not just evaluation keys like GetKeyMemoryMB. Measures
// the true footprint regardless of whether ORION_CHEDDAR_MANAGED_MEMORY
// is paging part of it to host RAM.
double GetPeakDeviceMemoryMB() {
    return static_cast<double>(cheddar::GetPeakDeviceBytes()) / (1024.0 * 1024.0);
}

// Working set: bytes users hold through the pool at an instant, and its
// peak. GetPeakDeviceMemoryMB above is the pool's footprint (what it took
// from the upstream, including per-bin preallocation), an upper bound.
double GetPeakLiveMemoryMB() {
    return static_cast<double>(cheddar::GetPeakLiveBytes()) / (1024.0 * 1024.0);
}
double GetLiveMemoryMB() {
    return static_cast<double>(cheddar::GetLiveBytes()) / (1024.0 * 1024.0);
}

int GetCiphertextLevel(int ct_id) { return level_of(g_state.ct(ct_id)); }
int GetPlaintextLevel(int pt_id) { return level_of(g_state.pt(pt_id)); }
// Real per-object widths (these were stubs returning the global
// slot_count, which masked the enc_sparse full-width encode behind the
// thr500 precision regression during debugging).
int GetCiphertextSlots(int ct_id) { return g_state.ct(ct_id).GetNumSlots(); }
int GetPlaintextSlots(int pt_id) { return g_state.pt(pt_id).GetNumSlots(); }

// Narrowing-only slot retag. ONLY valid when the caller guarantees the
// ciphertext's slot values are already periodic with period `slots`
// (e.g. after the hybrid method's rotate-and-add output fold, whose
// result is genuinely slots/2^r-periodic across the full width) -- a
// periodic dense ciphertext IS bit-for-bit a valid sparse (narrow)
// ciphertext, so this is pure metadata, no homomorphic work. Retagging
// a NON-periodic ciphertext silently corrupts rotation wraparound and
// bootstrap (the failure mode that killed the Phase-2 boot-resize
// attempt), hence the divisibility/narrowing guards.
void SetCiphertextSlots(int ct_id, int slots) {
    auto &ct = g_state.ct(ct_id);
    int cur = ct.GetNumSlots();
    if (slots <= 0 || slots > cur || cur % slots != 0)
        throw std::invalid_argument(
            "SetCiphertextSlots: target must be a positive divisor of the "
            "current width (narrowing only); got " + std::to_string(slots) +
            " from " + std::to_string(cur));
    ct.SetNumSlots(slots);
}

double GetCiphertextScale(int ct_id) { return g_state.ct(ct_id).GetScale(); }
double GetPlaintextScale(int pt_id) { return g_state.pt(pt_id).GetScale(); }
void SetCiphertextScale(int ct_id, double scale) {
    g_state.ct(ct_id).SetScale(scale);
}
void SetPlaintextScale(int pt_id, double scale) {
    g_state.pt(pt_id).SetScale(scale);
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// pybind11 module
// ---------------------------------------------------------------------------

PYBIND11_MODULE(ORION_CHEDDAR_MODULE_NAME, m) {
    m.doc() = "Cheddar CKKS backend bindings for Orion.";

    // Lifecycle
    m.def("setup_scheme", &setup_scheme, py::arg("params"));
    m.def("delete_scheme", &delete_scheme);
    m.def("delete_scheme_if", &delete_scheme_if, py::arg("generation"));

    // Key generation
    m.def("NewKeyGenerator", &NewKeyGenerator);
    m.def("GenerateSecretKey", &GenerateSecretKey);
    m.def("GeneratePublicKey", &GeneratePublicKey);
    m.def("GenerateRelinearizationKey", &GenerateRelinearizationKey);
    m.def("GenerateEvaluationKeys", &GenerateEvaluationKeys);

    // Encryptor / Decryptor
    m.def("NewEncryptor", &NewEncryptor);
    m.def("NewDecryptor", &NewDecryptor);

    // Encode / Decode
    m.def("Encode", &Encode, py::arg("values"), py::arg("level"),
          py::arg("scale") = 0.0);
    m.def("Decode", &Decode, py::arg("pt_id"));
    m.def("EncodeComplex", &EncodeComplex, py::arg("interleaved"),
          py::arg("level"), py::arg("scale") = 0.0);
    m.def("DecodeComplex", &DecodeComplex, py::arg("pt_id"));

    // Encrypt / Decrypt
    m.def("Encrypt", &Encrypt, py::arg("pt_id"));
    m.def("Decrypt", &Decrypt, py::arg("ct_id"));

    // ct - ct
    m.def("AddCiphertextNew", &AddCiphertextNew);
    m.def("AddCiphertext", &AddCiphertext);
    m.def("SubCiphertextNew", &SubCiphertextNew);
    m.def("SubCiphertext", &SubCiphertext);
    m.def("MulRelinCiphertextNew", &MulRelinCiphertextNew);
    m.def("MulRelinCiphertext", &MulRelinCiphertext);
    m.def("MulNoRelinCiphertextNew", &MulNoRelinCiphertextNew);
    m.def("RelinearizeNew", &RelinearizeNew);

    // ct - pt
    m.def("MulPlaintextNew", &MulPlaintextNew);
    m.def("MulPlaintext", &MulPlaintext);
    m.def("EncodeConstantNew", &EncodeConstantNew, py::arg("scalar"),
          py::arg("level"), py::arg("scale"));
    m.def("DeleteConstant", &DeleteConstant, py::arg("const_id"));
    m.def("MulConstantAccumNew", &MulConstantAccumNew, py::arg("ct_ids"),
          py::arg("const_ids"));
    m.def("PackConstantTableNew", &PackConstantTableNew,
          py::arg("const_ids"), py::arg("rows"));
    m.def("DeleteConstantTable", &DeleteConstantTable, py::arg("table_id"));
    m.def("MulConstantMatMulNew", &MulConstantMatMulNew, py::arg("ct_ids"),
          py::arg("table_id"));
    m.def("PackPlaintextTableNew", &PackPlaintextTableNew,
          py::arg("pt_ids"), py::arg("rows"));
    m.def("DeletePlaintextTable", &DeletePlaintextTable, py::arg("table_id"));
    m.def("MulPlaintextMatMulNew", &MulPlaintextMatMulNew, py::arg("ct_ids"),
          py::arg("table_id"));
    m.def("MulPlaintextAccumNew", &MulPlaintextAccumNew, py::arg("ct_ids"),
          py::arg("pt_ids"));

    // Double-hoisted linear transform
    m.def("GenerateHoistLT", &GenerateHoistLT, py::arg("diags_idxs"),
          py::arg("diags_data"), py::arg("level"), py::arg("bs"));
    m.def("GetHoistLTRotationKeys", &GetHoistLTRotationKeys,
          py::arg("hlt_id"));
    m.def("EvaluateHoistLT", &EvaluateHoistLT, py::arg("hlt_id"),
          py::arg("ct_id"));
    m.def("DeleteHoistLT", &DeleteHoistLT, py::arg("hlt_id"));
    m.def("AddPlaintextNew", &AddPlaintextNew);
    m.def("AddPlaintext", &AddPlaintext);
    m.def("SubPlaintextNew", &SubPlaintextNew);
    m.def("SubPlaintext", &SubPlaintext);

    // ct - scalar
    m.def("AddScalarNew", &AddScalarNew, py::arg("ct_id"), py::arg("scalar"));
    m.def("AddScalar", &AddScalar, py::arg("ct_id"), py::arg("scalar"));
    m.def("SubScalarNew", &SubScalarNew, py::arg("ct_id"), py::arg("scalar"));
    m.def("SubScalar", &SubScalar, py::arg("ct_id"), py::arg("scalar"));
    m.def("MulScalarFloatNew", &MulScalarFloatNew, py::arg("ct_id"),
          py::arg("scalar"));
    m.def("MulScalarFloat", &MulScalarFloat, py::arg("ct_id"),
          py::arg("scalar"));
    m.def("MulScalarIntNew", &MulScalarIntNew, py::arg("ct_id"),
          py::arg("scalar"));
    m.def("MulScalarInt", &MulScalarInt, py::arg("ct_id"), py::arg("scalar"));

    // Negate
    m.def("Negate", &Negate, py::arg("ct_id"));
    m.def("GetCiphertextBytes", &GetCiphertextBytes, py::arg("ct_id"));
    m.def("GetCiphertextBxBytes", &GetCiphertextBxBytes, py::arg("ct_id"));
    m.def("EncryptSeeded", &EncryptSeeded, py::arg("pt_id"), py::arg("seed"));
    m.def("DropAx", &DropAx, py::arg("ct_id"));
    m.def("ExpandSeededAx", &ExpandSeededAx, py::arg("ct_id"), py::arg("seed"));
    m.def("Conjugate", &Conjugate, py::arg("ct_id"));
    m.def("ConjugateNew", &ConjugateNew, py::arg("ct_id"));
    m.def("MulImaginaryUnit", &MulImaginaryUnit, py::arg("ct_id"));
    m.def("MulImaginaryUnitNew", &MulImaginaryUnitNew, py::arg("ct_id"));

    // Rescale
    m.def("RescaleNew", &RescaleNew);
    m.def("Rescale", &Rescale);

    // Rotations
    m.def("AddRotationKey", &AddRotationKey, py::arg("k"));
    m.def("AlignToLevelNew", &AlignToLevelNew, py::arg("ct_id"),
          py::arg("target_level"));
    m.def("RotateNew", &RotateNew, py::arg("ct_id"), py::arg("k"));
    m.def("RotateBatchNew", &RotateBatchNew, py::arg("ct_id"),
          py::arg("shifts"), py::arg("hoist_mode") = 0);

    // Bootstrap
    m.def("NewBootstrapper", &NewBootstrapper, py::arg("num_cts_levels"),
          py::arg("num_stc_levels"), py::arg("log_message_ratio"),
          py::arg("slots"));
    m.def("Bootstrap", &Bootstrap, py::arg("ct_id"), py::arg("slots"));
    m.def("DeleteBootstrappers", &DeleteBootstrappers);
    m.def("BootNumEvalModLevels", &BootNumEvalModLevels);
    m.def("GetRescalePrimeProds", &GetRescalePrimeProds);
    m.def("GetCanonicalScales", &GetCanonicalScales);

    // Polynomial evaluation
    m.def("EvaluatePolynomialNative", &EvaluatePolynomialNative,
          py::arg("ct_id"), py::arg("coeffs"), py::arg("target_scale"),
          py::arg("chebyshev") = false);

    // Lifecycle: ID deletion
    m.def("DeleteCiphertext", &DeleteCiphertext, py::arg("ct_id"));
    m.def("DeletePlaintext", &DeletePlaintext, py::arg("pt_id"));

    // Metadata
    m.def("GetKeyMemoryMB", &GetKeyMemoryMB);
    m.def("GetPeakDeviceMemoryMB", &GetPeakDeviceMemoryMB);
    m.def("GetPeakLiveMemoryMB", &GetPeakLiveMemoryMB);
    m.def("GetLiveMemoryMB", &GetLiveMemoryMB);
    m.def("GetCiphertextLevel", &GetCiphertextLevel, py::arg("ct_id"));
    m.def("GetPlaintextLevel", &GetPlaintextLevel, py::arg("pt_id"));
    m.def("GetCiphertextSlots", &GetCiphertextSlots, py::arg("ct_id"));
    m.def("SetCiphertextSlots", &SetCiphertextSlots, py::arg("ct_id"),
          py::arg("slots"));
    m.def("GetPlaintextSlots", &GetPlaintextSlots, py::arg("pt_id"));
    m.def("GetCiphertextScale", &GetCiphertextScale, py::arg("ct_id"));
    m.def("GetPlaintextScale", &GetPlaintextScale, py::arg("pt_id"));
    m.def("SetCiphertextScale", &SetCiphertextScale, py::arg("ct_id"),
          py::arg("scale"));
    m.def("SetPlaintextScale", &SetPlaintextScale, py::arg("pt_id"),
          py::arg("scale"));
}
