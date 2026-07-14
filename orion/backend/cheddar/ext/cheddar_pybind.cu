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

#include <UserInterface.h>
#include <core/Context.h>
#include <extension/BootContext.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

using word = uint64_t;
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

    // ID registries. next_*_id starts at 1; 0 is reserved for "invalid".
    std::map<int, std::unique_ptr<Pt>> plaintexts;
    std::map<int, std::unique_ptr<Ct>> ciphertexts;
    int next_pt_id = 1;
    int next_ct_id = 1;

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

    int num_levels = static_cast<int>(main_primes.size());
    // Flat chain: level i uses main primes 0..i, no terminal primes.
    // Orion's level convention (level = index into LogQ) maps directly.
    std::vector<std::pair<int, int>> level_config;
    level_config.reserve(num_levels);
    for (int i = 0; i < num_levels; ++i) level_config.emplace_back(i + 1, 0);

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
        level_config, main_primes, aux_primes);

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
    double s = (scale > 0.0) ? scale : g_state.default_scale;
    std::vector<Complex> msg(values.size());
    for (size_t i = 0; i < values.size(); ++i) msg[i] = Complex(values[i], 0.0);
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
    if (!g_state.boot_context || !g_state.boot_prepared_slots.count(slots))
        throw std::runtime_error(
            "Cheddar: no bootstrapper prepared for slots=" +
            std::to_string(slots) + ". Call NewBootstrapper first.");

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

void DeleteBootstrappers() {
    // No-op: the BootContext is the scheme's single shared context, so it
    // cannot be torn down independently -- doing so would destroy the
    // context regular ops still use (or, if another ref keeps it alive,
    // leave boot_prepared_slots inconsistent with the still-live
    // precompute). All boot state is released by delete_scheme() at scheme
    // teardown. This is also generation-safe: a stale bootstrapper
    // finalizer running after a new scheme is set up won't disturb it.
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

int GetCiphertextLevel(int ct_id) { return level_of(g_state.ct(ct_id)); }
int GetPlaintextLevel(int pt_id) { return level_of(g_state.pt(pt_id)); }
int GetCiphertextSlots(int /*ct_id*/) { return g_state.slot_count; }
int GetPlaintextSlots(int /*pt_id*/) { return g_state.slot_count; }

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

PYBIND11_MODULE(_cheddar_native, m) {
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

    // Rescale
    m.def("RescaleNew", &RescaleNew);
    m.def("Rescale", &Rescale);

    // Rotations
    m.def("AddRotationKey", &AddRotationKey, py::arg("k"));
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

    // Lifecycle: ID deletion
    m.def("DeleteCiphertext", &DeleteCiphertext, py::arg("ct_id"));
    m.def("DeletePlaintext", &DeletePlaintext, py::arg("pt_id"));

    // Metadata
    m.def("GetKeyMemoryMB", &GetKeyMemoryMB);
    m.def("GetCiphertextLevel", &GetCiphertextLevel, py::arg("ct_id"));
    m.def("GetPlaintextLevel", &GetPlaintextLevel, py::arg("pt_id"));
    m.def("GetCiphertextSlots", &GetCiphertextSlots, py::arg("ct_id"));
    m.def("GetPlaintextSlots", &GetPlaintextSlots, py::arg("pt_id"));
    m.def("GetCiphertextScale", &GetCiphertextScale, py::arg("ct_id"));
    m.def("GetPlaintextScale", &GetPlaintextScale, py::arg("pt_id"));
    m.def("SetCiphertextScale", &SetCiphertextScale, py::arg("ct_id"),
          py::arg("scale"));
    m.def("SetPlaintextScale", &SetPlaintextScale, py::arg("pt_id"),
          py::arg("scale"));
}
