/* ═══════════════════════════════════════════════════════════════════════
 * OICOS — OCTIV INTEGRATED CIRCUIT OPERATING SYSTEM
 * Formal Microcode Specification v0.2.0
 * ═══════════════════════════════════════════════════════════════════════
 *
 * AXIOMS
 *   I.   There is exactly one data structure: the Snap (32 bytes).
 *   II.  There is exactly one operation: snap_step(snap, trit).
 *   III. There is exactly one execution model: the Cascade.
 *   IV.  All OS services are topological patterns of Snaps.
 *   V.   Rank is monotonically non-increasing. Security is algebraic.
 *
 * SELF-SIMILARITY
 *   At every level of abstraction, OICOS entities are addressed,
 *   configured, and manipulated through the Snap interface.
 *   A Snap's edges connect to other Snaps, forming SNAGs that are
 *   themselves reachable through a Snap. The Snap is both the atom
 *   and the molecule — processes, memory, IPC, scheduling, and the
 *   kernel itself are all Snap topologies.
 *
 * CRITICAL PATH
 *   snap_step() requires exactly:
 *     2 bitfield extractions (quark selection)
 *     2 ROM lookups          (81-byte endomorphism table)
 *     1 array index          (edge routing)
 *     0 branches
 *   Hot-path footprint: 113 bytes (81 ROM + 32 Snap).
 *   Fits in 2 cache lines.
 *
 * MICROCODE ROM BUDGET
 *   ENDO[27][3]         81 B    Endomorphism transition matrix
 *   ENDO_RANK[27]       27 B    Endomorphism → image size
 *   OIC_ROUTE[3][2][7]  42 B    Rank × domain × cell → rank
 *   DTRIT[6][6]         36 B    Domain × phase → input trit
 *   ALU_S3[6][6]        36 B    S₃ Cayley composition table
 *   ALU_CHIRAL[3][3]     9 B    Chirality composition
 *   VALID[3][4]         12 B    Rank × quark_rank → bool
 *   CELL_RANK[7]         7 B    Cell → natural rank
 *   CANON_SQ[7]         14 B    Cell → canonical state quarks
 *   S3_TO_ENDO[6]        6 B    Dense S₃ index → endo index
 *   ───────────────────────────
 *   Total ROM:         270 bytes
 *
 * ENCODING CONVENTION
 *   Endomorphism index: i = f(0) + 3·f(1) + 9·f(2)
 *   Base-3 little-endian in state-space.
 *   e0={0,0,0} constant-0    e13={1,1,1} constant-1
 *   e21={0,1,2} IDENTITY     e26={2,2,2} constant-2
 *   e7={1,2,0} ROTATE CW     e11={2,0,1} ROTATE CCW
 *   e5={2,1,0} SWAP(0,2)     e15={0,2,1} SWAP(1,2)
 *   e19={1,0,2} SWAP(0,1)
 *
 * DERIVATION
 *   All constants derived from the OIC formal verification:
 *   7 cells from Myhill-Nerode collapse of 387,420,489 particles.
 *   Routing matrix verified by SMT solver (Z3/CVC4).
 *   Endomorphism closure proven UNSAT for boundary violations.
 *   S₃ ALU is a perfect Latin square (100% informational efficiency).
 *   2,163 frozen laws, zero exceptions, 100% singularity.
 *
 * ═══════════════════════════════════════════════════════════════════════ */

#ifndef OICOS_H
#define OICOS_H

#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#define OICOS_VERSION "0.2.0"

/* ═══════════════════════════════════════════════════════════════════════
 * I. CONSTANTS
 * ═══════════════════════════════════════════════════════════════════════ */

/* The Seven Cells (3 bits, 0-6).
 * Myhill-Nerode irreducible: 387,420,489 particles collapse to exactly 7
 * dynamical equivalence classes. These are the only "opcodes" OICOS needs.
 *
 *   Rank 1 (absorber):     NULL
 *   Rank 2 (lossy):        COLLAPSE, SPLIT, OSCILLATOR
 *   Rank 3 (surjective):   IDENTITY, SWAP, ROTATE
 */
#define CELL_NULL   0   /* Absorber:   rank 1, spin 0 — halt/free/ground    */
#define CELL_CLPS   1   /* Collapse:   rank 2, spin 0, 1 basin — join/lock  */
#define CELL_SPLT   2   /* Split:      rank 2, spin 0, 2 basins — fork      */
#define CELL_OSC    3   /* Oscillator: rank 2, spin 2 — timer/clock          */
#define CELL_ID     4   /* Identity:   rank 3, spin 0 — buffer/pipe/relay    */
#define CELL_SWP    5   /* Swap:       rank 3, spin 2 — exchange/switch      */
#define CELL_ROT    6   /* Rotate:     rank 3, spin 3 — schedule/cycle       */

/* The Rank Lattice (2 bits, 0-2).
 * Bus state = privilege level. MONOTONICALLY NON-INCREASING.
 * Once rank drops, it never rises. This IS the security model.
 *
 *   RANK_P ──→ RANK_S ──→ RANK_K
 *   kernel      user        dead
 *   S₃ closed   lossy OK    absorber
 */
#define RANK_K   0   /* Constant:     dead / information destroyed         */
#define RANK_S   1   /* Semi:         user mode / lossy gates permitted    */
#define RANK_P   2   /* Permutative:  kernel mode / S₃ group closure      */

/* Clock Domains (3 bits, 0-4). */
#define DOM_CYC  0   /* Cyclic:      [0,1,2,…]  g = f₂∘f₁∘f₀              */
#define DOM_CON  1   /* Constant:    [0,0,0,…]  g = f₀³                    */
#define DOM_ALT  2   /* Alternating: [0,1,0,…]  alternating f₀, f₁         */
#define DOM_REV  3   /* Reversed:    [2,1,0,…]  g = f₀∘f₁∘f₂              */
#define DOM_BIN  4   /* Binary:      application-defined                    */
#define DOM_IO   5   /* I/O:         channel-routed input/output            */

/* Snap Status (2 bits, 0-3). */
#define STAT_FREE   0   /* On free list — reclaimable                      */
#define STAT_IDLE   1   /* Alive, not scheduled — dormant                  */
#define STAT_LIVE   2   /* In active or pending wave — executing           */
#define STAT_LOCK   3   /* Blocked in synchronization primitive            */

/* Sentinel: null edge destination. */
#define SNAP_NIL    UINT32_MAX

/* ═══════════════════════════════════════════════════════════════════════
 * II. THE SNAP (32 bytes)
 * ═══════════════════════════════════════════════════════════════════════
 *
 * The universal and only data structure of OICOS.
 *
 * ┌────────┬──────┬──────────┬──────────────────────────────────────┐
 * │ Offset │ Size │ Field    │ Layout                               │
 * ├────────┼──────┼──────────┼──────────────────────────────────────┤
 * │   0    │  2   │ sq       │ f₀[4:0] | f₁[9:5] | f₂[14:10] | 0  │
 * │   2    │  2   │ oq       │ o₀[4:0] | o₁[9:5] | o₂[14:10] | 0  │
 * │   4    │  1   │ hdr      │ cell[7:5] | rank[4:3] | dom[2:0]    │
 * │   5    │  1   │ sr       │ st[7:6] | out[5:4] | stat[3:2] | 00 │
 * │   6    │  2   │ gen      │ 16-bit epoch counter                 │
 * │   8    │  12  │ e[3]     │ trit-routed edge destinations        │
 * │  20    │  4   │ back     │ parent (live) / next free (free)     │
 * │  24    │  8   │ data     │ 64-bit payload                       │
 * └────────┴──────┴──────────┴──────────────────────────────────────┘
 *
 * 2 Snaps per 64-byte cache line.
 * 32,768 Snaps per 1 MB.
 * 4,294,967,295 max addressable Snaps (32-bit edges).
 */
typedef struct {
    uint16_t sq;        /* state quarks:  f₀|f₁|f₂ packed 5:5:5:1         */
    uint16_t oq;        /* output quarks: o₀|o₁|o₂ packed 5:5:5:1         */
    uint8_t  hdr;       /* cell:3 | rank:2 | domain:3                      */
    uint8_t  sr;        /* state:2 | output:2 | status:2 | reserved:2      */
    uint16_t gen;       /* generation / epoch / cascade dedup marker        */
    uint32_t e[3];      /* edge[trit] → destination snap index              */
    uint32_t back;      /* parent snap (LIVE) / next free snap (FREE)       */
    uint64_t data;      /* payload: pointer, value, instruction, hash       */
} __attribute__((packed, aligned(32))) snap_t;

_Static_assert(sizeof(snap_t) == 32, "Snap must be exactly 32 bytes");

/* ═══════════════════════════════════════════════════════════════════════
 * III. FIELD ACCESS (all branchless, O(1))
 * ═══════════════════════════════════════════════════════════════════════ */

/* --- Quark packing --- */
#define PACK_Q(f0,f1,f2)  ((uint16_t)( \
    ((uint16_t)(f0)) | (((uint16_t)(f1)) << 5U) | (((uint16_t)(f2)) << 10U)))

static inline uint8_t snap_q(uint16_t packed, uint8_t trit) {
    return (uint8_t)((packed >> (trit * 5U)) & 0x1FU);
}

/* --- Header: cell[7:5] | rank[4:3] | dom[2:0] --- */
#define PACK_HDR(cell,rank,dom)  ((uint8_t)( \
    (((unsigned)(cell)) << 5U) | (((unsigned)(rank)) << 3U) | ((unsigned)(dom))))

static inline uint8_t snap_cell(const snap_t *s)  { return (uint8_t)(s->hdr >> 5U);          }
static inline uint8_t snap_rank(const snap_t *s)  { return (uint8_t)((s->hdr >> 3U) & 0x03U); }
static inline uint8_t snap_dom(const snap_t *s)   { return (uint8_t)(s->hdr & 0x07U);         }

/* --- State register: st[7:6] | out[5:4] | stat[3:2] | rsvd[1:0] --- */
#define PACK_SR(st,out,stat)  ((uint8_t)( \
    (((unsigned)(st)) << 6U) | (((unsigned)(out)) << 4U) | (((unsigned)(stat)) << 2U)))

static inline uint8_t snap_st(const snap_t *s)    { return (uint8_t)(s->sr >> 6U);            }
static inline uint8_t snap_out(const snap_t *s)   { return (uint8_t)((s->sr >> 4U) & 0x03U);  }
static inline uint8_t snap_stat(const snap_t *s)  { return (uint8_t)((s->sr >> 2U) & 0x03U);  }

static inline void snap_set_fsm(snap_t *s, uint8_t st, uint8_t out) {
    s->sr = (uint8_t)(((unsigned)st << 6U) | ((unsigned)out << 4U) | (s->sr & 0x0FU));
}

static inline void snap_set_stat(snap_t *s, uint8_t stat) {
    s->sr = (uint8_t)((s->sr & 0xF3U) | ((unsigned)stat << 2U));
}

/* --- SR reserved bits [1:0] — flag field --- */
#define SNAP_FLAG_BCAST  0x01U  /* bit 0: broadcast mode — fan-out to ALL edges */

static inline bool snap_bcast(const snap_t *s) { return (s->sr & SNAP_FLAG_BCAST) != 0U; }
static inline void snap_set_bcast(snap_t *s, bool on) {
    s->sr = (uint8_t)((s->sr & 0xFCU) | (on ? SNAP_FLAG_BCAST : 0U));
}

/* --- I/O channel convention in data field ---
 *   data[7:0]  = input channel  (0-255, 0 = stdin)
 *   data[15:8] = output channel (0-255, 0 = stdout)
 */
#define SNAP_IO_IN(s)   ((uint8_t)((s)->data & 0xFFU))
#define SNAP_IO_OUT(s)  ((uint8_t)(((s)->data >> 8U) & 0xFFU))

/* ═══════════════════════════════════════════════════════════════════════
 * IV. THE ROM (270 bytes total)
 * ═══════════════════════════════════════════════════════════════════════ */

/* --- 4a. Endomorphism Transition Matrix (81 bytes) ---
 *
 * ENDO[i][s] = f(s), where f is the endomorphism at index i.
 * Index: i = f(0) + 3·f(1) + 9·f(2).
 *
 * This table is the CPU of the operating system.
 * Every computation in OICOS ultimately reduces to lookups in this table.
 */
static const uint8_t ENDO[27][3] = {
    {0,0,0}, {1,0,0}, {2,0,0},   /* e0-e2:   f(2)=0, f(1)=0           */
    {0,1,0}, {1,1,0}, {2,1,0},   /* e3-e5:   f(2)=0, f(1)=1  e5=SWP02 */
    {0,2,0}, {1,2,0}, {2,2,0},   /* e6-e8:   f(2)=0, f(1)=2  e7=ROTCW */
    {0,0,1}, {1,0,1}, {2,0,1},   /* e9-e11:  f(2)=1, f(1)=0  e11=RCCW */
    {0,1,1}, {1,1,1}, {2,1,1},   /* e12-e14: f(2)=1, f(1)=1  e13=K(1) */
    {0,2,1}, {1,2,1}, {2,2,1},   /* e15-e17: f(2)=1, f(1)=2  e15=SW12 */
    {0,0,2}, {1,0,2}, {2,0,2},   /* e18-e20: f(2)=2, f(1)=0  e19=SW01 */
    {0,1,2}, {1,1,2}, {2,1,2},   /* e21-e23: f(2)=2, f(1)=1  e21=ID   */
    {0,2,2}, {1,2,2}, {2,2,2}    /* e24-e26: f(2)=2, f(1)=2  e26=K(2) */
};

/* --- 4b. Endomorphism Rank LUT (27 bytes) ---
 *
 * ENDO_RANK[i] = |image(e_i)| ∈ {1, 2, 3}
 *   Rank 1 (constant):    {0, 13, 26}         3 endomorphisms
 *   Rank 3 (surjective):  {5, 7, 11, 15, 19, 21}   6 endomorphisms
 *   Rank 2 (everything else):                  18 endomorphisms
 */
static const uint8_t ENDO_RANK[27] = {
    1,2,2, 2,2,3, 2,3,2,
    2,2,3, 2,1,2, 3,2,2,
    2,3,2, 3,2,2, 2,2,1
};

/* --- 4c. OIC Routing Matrix (42 bytes) ---
 *
 * OIC_ROUTE[rank][domain_class][cell] → new_rank
 *
 * domain_class: 0 = varied (CYC/ALT/REV/BIN), 1 = constant (CON)
 *
 * INVARIANT: OIC_ROUTE[r][d][c] ≤ r   (monotonicity — THE security model)
 *
 * Formally verified via SMT-LIB v2 (Z3). Zero violations across all inputs.
 */
static const uint8_t OIC_ROUTE[3][2][7] = {
    /* RANK_K: absorber — everything stays dead */
    { {0,0,0,0,0,0,0}, {0,0,0,0,0,0,0} },
    /* RANK_S: NULL kills; lossy gates maintain; S₃ gates cap at RANK_S */
    { {0,1,1,1,1,1,1}, {0,0,1,1,1,1,1} },
    /* RANK_P: full lattice; S₃ gates preserve RANK_P */
    { {0,1,1,1,2,2,2}, {0,0,1,1,2,2,2} }
};

static inline uint8_t route_class(uint8_t dom) {
    return (dom == DOM_CON) ? 1 : 0;
}

/* --- 4d. Domain Input Trit Pattern (36 bytes) ---
 *
 * DTRIT[domain][tick % 6] → input trit
 * Period 6 = LCM(1, 2, 3) covers all domain cycles.
 */
static const uint8_t DTRIT[6][6] = {
    {0,1,2,0,1,2},   /* CYC: repeating 0,1,2                             */
    {0,0,0,0,0,0},   /* CON: always 0                                     */
    {0,1,0,1,0,1},   /* ALT: alternating 0,1                              */
    {2,1,0,2,1,0},   /* REV: reversed 2,1,0                               */
    {0,1,0,1,0,1},   /* BIN: default pattern; override at runtime          */
    {0,0,0,0,0,0}    /* IO: overridden at runtime by channel read          */
};

/* --- 4e. S₃ Cayley Composition Table (36 bytes) ---
 *
 * ALU_S3[f][g] = f∘g  (standard function composition)
 *
 * Dense index:  0=ID(e21)  1=SWP02(e5)  2=SWP12(e15)
 *               3=SWP01(e19)  4=ROTCW(e7)  5=ROTCCW(e11)
 *
 * Perfect Latin square: 100% informational efficiency.
 * H(X) = log₂(6) ≈ 2.585 bits. Zero structural waste.
 */
static const uint8_t ALU_S3[6][6] = {
    {0,1,2,3,4,5}, {1,0,5,4,3,2}, {2,4,0,5,1,3},
    {3,5,4,0,2,1}, {4,2,3,1,5,0}, {5,3,1,2,0,4}
};

/* --- 4f. Chirality Composition (9 bytes) ---
 *
 * ALU_CHIRAL[left][right] → composed chirality
 *   0 = ACHIRAL, 1 = CW, 2 = CCW
 */
static const uint8_t ALU_CHIRAL[3][3] = {
    {0,0,0}, {0,2,0}, {0,0,1}
};

/* --- 4g. Quark Validity Matrix (12 bytes) ---
 *
 * VALID[snap_rank][quark_rank] → can this quark appear at this rank?
 */
static const uint8_t VALID[3][4] = {
    /* quark_rank:   0(n/a) 1(K)  2(S)  3(P) */
    /* RANK_K: */  { 0,     1,    1,    1 },   /* Any quark; snap is dead  */
    /* RANK_S: */  { 0,     0,    1,    1 },   /* No constants             */
    /* RANK_P: */  { 0,     0,    0,    1 }    /* Surjective only          */
};

/* --- 4h. S₃ ↔ Endomorphism Index Maps --- */
static const uint8_t S3_TO_ENDO[6] = { 21, 5, 15, 19, 7, 11 };

static const uint8_t ENDO_TO_S3[27] = {
    0xFF,0xFF,0xFF, 0xFF,0xFF,   1, 0xFF,   4, 0xFF,
    0xFF,0xFF,   5, 0xFF,0xFF,0xFF,    2, 0xFF, 0xFF,
    0xFF,   3,0xFF,    0,0xFF,0xFF, 0xFF,0xFF, 0xFF
};

/* --- 4i. Cell → Natural Rank --- */
static const uint8_t CELL_RANK[7] = {
    RANK_K, RANK_S, RANK_S, RANK_S, RANK_P, RANK_P, RANK_P
};

/* --- 4j. Canonical State Quarks (14 bytes) ---
 *
 * Under cyclic tape, g = f₂∘f₁∘f₀ where fiber = {f₀, f₁, f₂}:
 *   NULL:  {0,0,0}    → g=e0  {0,0,0}  rank 1, spin 0
 *   CLPS:  {9,21,21}  → g=e9  {0,0,1}  rank 2, spin 0, 1 basin
 *   SPLT:  {3,21,21}  → g=e3  {0,1,0}  rank 2, spin 0, 2 basins
 *   OSC:   {1,21,21}  → g=e1  {1,0,0}  rank 2, spin 2, 0↔1 cycle
 *   ID:    {21,21,21} → g=e21 {0,1,2}  rank 3, spin 0, 3 fixed pts
 *   SWP:   {5,21,21}  → g=e5  {2,1,0}  rank 3, spin 2, swap 0↔2
 *   ROT:   {7,21,21}  → g=e7  {1,2,0}  rank 3, spin 3, CW rotation
 */
static const uint16_t CANON_SQ[7] = {
    PACK_Q( 0,  0,  0),   /* NULL */
    PACK_Q( 9, 21, 21),   /* CLPS */
    PACK_Q( 3, 21, 21),   /* SPLT */
    PACK_Q( 1, 21, 21),   /* OSC  */
    PACK_Q(21, 21, 21),   /* ID   */
    PACK_Q( 5, 21, 21),   /* SWP  */
    PACK_Q( 7, 21, 21)    /* ROT  */
};

#define CANON_OQ  PACK_Q(21, 21, 21)   /* Default output: identity (faithful) */

/* ═══════════════════════════════════════════════════════════════════════
 * V. THE STEP (Atomic Execution — The Single Operation)
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Every computation in OICOS reduces to this function.
 * Processes, scheduling, IPC, memory management — all are cascades
 * of snap_step calls through a SNAG topology.
 *
 * Cycle: 2 extractions + 2 LUT lookups + 1 index = O(1).
 * Branches: 0.
 */
typedef struct {
    uint8_t  out;          /* emitted output trit ∈ {0,1,2}                */
    uint32_t dst;          /* destination snap index (SNAP_NIL if terminal) */
} snap_result_t;

static inline snap_result_t snap_step(snap_t *s, uint8_t input_trit) {
    /* 1. Extract active endomorphism indices */
    uint8_t s_endo = snap_q(s->sq, input_trit);
    uint8_t o_endo = snap_q(s->oq, input_trit);

    /* 2. Read current FSM state */
    uint8_t cur = snap_st(s);

    /* 3. Compute output FIRST (depends on pre-update state) */
    uint8_t out = ENDO[o_endo][cur];

    /* 4. Compute next state */
    uint8_t nxt = ENDO[s_endo][cur];

    /* 5. Update FSM register */
    snap_set_fsm(s, nxt, out);

    /* 6. Route: output trit selects destination edge */
    snap_result_t r;
    r.out = out;
    r.dst = s->e[out];
    return r;
}

/* Rank degradation — applied at the receiving end during cascade.
 * Source snap's cell type determines how destination's rank degrades. */
static inline void snap_degrade(snap_t *dst, uint8_t source_cell) {
    uint8_t r  = snap_rank(dst);
    uint8_t dc = route_class(snap_dom(dst));
    uint8_t nr = OIC_ROUTE[r][dc][source_cell];
    dst->hdr = PACK_HDR(snap_cell(dst), nr, snap_dom(dst));
}

/* ═══════════════════════════════════════════════════════════════════════
 * V-B. ADAPTIVE STEP (Optional Mutation)
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Wraps snap_step with a lumen/crystallization/decay model.
 * Backward-compatible: mode==0 returns immediately (zero overhead).
 *
 * Data field encoding (48-bit fixnum compatible):
 *   data[43:40] = mode      (4 bits: 0=none, 1=lumen, 2=crystallize, 3=decay)
 *   data[39:36] = target    (4 bits: cell type to become, 0-6)
 *   data[35:20] = threshold (16 bits: step count before mutation)
 *   data[19:4]  = counter   (16 bits: current step count, starts at 0)
 *   data[3:0]   = reserved  (4 bits)
 *
 * Non-adaptive snaps (mode=0) use data[15:0] for I/O channels as before.
 */

#define SNAP_FLAG_ADAPT  0x02U  /* sr bit 1: snap has adaptive data (informational) */

#define ADAPT_NONE        0
#define ADAPT_LUMEN       1
#define ADAPT_CRYSTALLIZE 2
#define ADAPT_DECAY       3

#define ADAPT_MODE(s)    ((uint8_t)(((s)->data >> 40U) & 0x0FU))
#define ADAPT_TARGET(s)  ((uint8_t)(((s)->data >> 36U) & 0x0FU))
#define ADAPT_THRESH(s)  ((uint16_t)(((s)->data >> 20U) & 0xFFFFU))
#define ADAPT_COUNTER(s) ((uint16_t)(((s)->data >> 4U) & 0xFFFFU))

static inline snap_result_t snap_step_adaptive(snap_t *s, uint8_t input_trit) {
    snap_result_t r = snap_step(s, input_trit);

    /* Fast path: no adaptivity (zero overhead for mode==0) */
    uint8_t mode = ADAPT_MODE(s);
    if (mode == 0) return r;

    /* Increment counter */
    uint16_t counter = ADAPT_COUNTER(s) + 1;
    s->data = (s->data & ~(0xFFFFULL << 4)) | ((uint64_t)counter << 4);

    /* Check threshold */
    if (counter >= ADAPT_THRESH(s)) {
        uint8_t target = ADAPT_TARGET(s);

        if (mode == ADAPT_LUMEN || mode == ADAPT_CRYSTALLIZE) {
            /* Mutate: change cell type, preserve edges and domain */
            if (target <= 6) {
                s->sq  = CANON_SQ[target];
                s->oq  = CANON_OQ;
                s->hdr = PACK_HDR(target, CELL_RANK[target], snap_dom(s));
            }
            /* Reset counter */
            s->data &= ~(0xFFFFULL << 4);
        } else if (mode == ADAPT_DECAY) {
            /* Decay to NULL — snap dies */
            s->sq  = CANON_SQ[CELL_NULL];
            s->oq  = CANON_OQ;
            s->hdr = PACK_HDR(CELL_NULL, RANK_K, snap_dom(s));
            snap_set_stat(s, STAT_IDLE);
            /* Clear mode (one-shot) */
            s->data &= ~(0x0FULL << 40);
        }
    }

    return r;
}

/* ═══════════════════════════════════════════════════════════════════════
 * VI. THE CASCADE (The Execution Model)
 * ═══════════════════════════════════════════════════════════════════════
 *
 * One tick = one BFS wave through the SNAG.
 * Every active snap executes exactly one step per tick.
 * The wave propagates level-synchronously: tick t+1 depends only on t.
 *
 * The cascade IS the kernel. There is no scheduler function, no
 * interrupt handler, no main loop. The SNAG topology determines
 * execution order. The wave IS the program counter.
 */
typedef struct {
    snap_t   *snaps;            /* Contiguous snap arena                   */
    uint32_t  snap_count;       /* Total snaps in arena                    */
    uint32_t *active;           /* Current wave (snap indices to execute)  */
    uint32_t *pending;          /* Next wave (populated during tick)       */
    uint32_t  active_count;     /* Snaps in current wave                   */
    uint32_t  pending_count;    /* Snaps in next wave                      */
    uint32_t  tick;             /* Global monotonic tick counter            */
} oicos_t;

/* One tick of the operating system. */
static inline void oicos_tick(oicos_t *os) {
    os->pending_count = 0;
    os->tick++;
    uint16_t cur_gen = (uint16_t)(os->tick & 0xFFFFU);

    for (uint32_t i = 0; i < os->active_count; i++) {
        uint32_t idx = os->active[i];
        snap_t  *s   = &os->snaps[idx];

        /* Per-snap input trit from its clock domain */
        uint8_t trit = DTRIT[snap_dom(s)][os->tick % 6U];

        /* Execute the single operation */
        snap_result_t r = snap_step(s, trit);

        /* Rank degradation + deduplicated routing */
        if (r.dst != SNAP_NIL && r.dst < os->snap_count) {
            snap_t *dst = &os->snaps[r.dst];

            /* Degrade destination rank based on source cell type */
            snap_degrade(dst, snap_cell(s));

            /* Dedup: generation marker prevents duplicate wave entries */
            if (dst->gen != cur_gen && snap_rank(dst) > RANK_K) {
                dst->gen = cur_gen;
                os->pending[os->pending_count++] = r.dst;
            }
        }
    }

    /* Swap wave buffers */
    uint32_t *tmp = os->active;
    os->active  = os->pending;
    os->pending = tmp;
    os->active_count = os->pending_count;
}

/* ═══════════════════════════════════════════════════════════════════════
 * VII. CONSTRUCTORS
 * ═══════════════════════════════════════════════════════════════════════ */

/* Create a canonical snap of the given cell type. */
static inline snap_t snap_make(uint8_t cell, uint8_t dom,
                               uint32_t e0, uint32_t e1, uint32_t e2,
                               uint64_t payload) {
    snap_t s;
    memset(&s, 0, sizeof(s));
    s.sq   = CANON_SQ[cell];
    s.oq   = CANON_OQ;
    s.hdr  = PACK_HDR(cell, CELL_RANK[cell], dom);
    s.sr   = PACK_SR(0, 0, STAT_IDLE);
    s.e[0] = e0;
    s.e[1] = e1;
    s.e[2] = e2;
    s.data = payload;
    return s;
}

/* Validate snap invariants: quarks compatible with declared rank. */
static inline bool snap_valid(const snap_t *s) {
    uint8_t r = snap_rank(s);
    return VALID[r][ENDO_RANK[snap_q(s->sq, 0)]]
        && VALID[r][ENDO_RANK[snap_q(s->sq, 1)]]
        && VALID[r][ENDO_RANK[snap_q(s->sq, 2)]];
}

/* ═══════════════════════════════════════════════════════════════════════
 * VII-B. CHECKSUM (FNV-1a)
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Used in .snap file headers (data field) to validate arena integrity.
 * All tools that read or write .snap files share this implementation.
 */
static inline uint64_t oicos_fnv1a(const void *data, size_t len) {
    const uint8_t *p = (const uint8_t *)data;
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < len; i++)
        h = (h ^ p[i]) * 0x100000001b3ULL;
    return h;
}

/* ═══════════════════════════════════════════════════════════════════════
 * VIII. THE SEVEN PRIMITIVES — OS Topology Patterns
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Every OS service is a wiring pattern of Snaps. No special cases.
 * The cell type determines behavior; edges determine connectivity.
 *
 * ── PROCESS ──────────────────────────────────────────────────────────
 *
 *   A process is a chain of Snaps connected edge-to-edge.
 *   The wave front IS the instruction pointer.
 *
 *     ┌────┐     ┌────┐     ┌────┐     ┌────┐
 *     │ ID │──→──│ ID │──→──│ ID │──→──│NULL│
 *     └────┘     └────┘     └────┘     └────┘
 *     step 0     step 1     step 2     halt
 *
 *   Each Snap's payload carries the process data for that step.
 *   Process termination: wave reaches a NULL snap (absorbed).
 *
 * ── SCHEDULER (Round-Robin) ──────────────────────────────────────────
 *
 *                  ┌──── e[0] ──→ [Process A head]
 *     ┌──────┐    │
 *     │ROTATE│────┼──── e[1] ──→ [Process B head]
 *     │spin=3│    │
 *     └──────┘    └──── e[2] ──→ [Process C head]
 *
 *   ROTATE cycles state 0→1→2→0 each tick.
 *   Output trit selects which process edge fires.
 *   Result: round-robin scheduling with zero overhead.
 *   Connect ROTATE's back-edge to itself for perpetual scheduling.
 *
 * ── FORK (Process Creation) ──────────────────────────────────────────
 *
 *     ┌──────┐    ┌──── e[0] ──→ [Child A]
 *     │ SPLT │────┤
 *     │2basin│    └──── e[1] ──→ [Child B]
 *     └──────┘
 *
 *   SPLIT has 2 attractors: input signals bifurcate into 2 basins.
 *   Each basin maps to a different child process edge.
 *
 * ── MUTEX (Synchronization) ──────────────────────────────────────────
 *
 *     [Writer A]──→──┐
 *                     ├──→──┌──────┐──→──[Critical Section]
 *     [Writer B]──→──┘      │ CLPS │
 *                           │1basin│
 *                           └──────┘
 *
 *   COLLAPSE merges all input signals to one attractor (basin=1).
 *   First signal acquires the resource. Subsequent signals are
 *   absorbed into the same basin — serialized, not lost.
 *   Rank degradation enforces: once collapsed, cannot re-expand.
 *
 * ── TIMER (Periodic Interrupt) ───────────────────────────────────────
 *
 *     ┌──────┐         ┌──────┐
 *     │ OSC  │──→──→───│target│
 *     │spin=2│         └──────┘
 *     └──┬───┘
 *        └── back-edge to self (perpetual oscillation)
 *
 *   OSCILLATOR has a 2-cycle: output alternates between trits.
 *   Connect to a target snap for periodic activation.
 *   Self-loop edge keeps the oscillator in the active wave.
 *
 * ── CHANNEL (IPC) ────────────────────────────────────────────────────
 *
 *   An edge IS a channel. The output trit IS the message.
 *   Ternary signaling: 3 possible messages per edge per tick.
 *   No buffering, no copying, no syscall — signals propagate
 *   through the SNAG at cascade speed.
 *
 * ── MEMORY ───────────────────────────────────────────────────────────
 *
 *   IDENTITY snap = memory cell. Payload = data. Rank = protection.
 *   Free list = chain of NULL snaps linked via `back` field.
 *   Allocation = pop from free list, set cell to desired type.
 *   Deallocation = set cell to NULL, push to free list.
 *   No malloc. No free. No fragmentation. O(1) everything.
 *
 * ═══════════════════════════════════════════════════════════════════════
 * IX. BINARY FORMAT (.snap)
 * ═══════════════════════════════════════════════════════════════════════
 *
 * A .snap file is a sequence of snap_t records. The first record is the
 * file header. The remaining records are the arena contents.
 *
 * The header IS a snap. Self-reference: the file format uses the same
 * structure it describes. Zero extra types.
 *
 * Header snap layout:
 *   sq   = 0               (unused)
 *   oq   = 0               (unused)
 *   hdr  = PACK_HDR(CELL_ID, RANK_P, DOM_CYC)  (metadata snap)
 *   sr   = 0               (unused)
 *   gen  = version number   (spec version)
 *   e[0] = snap_count       (number of data snaps)
 *   e[1] = root_index       (first snap in boot wave)
 *   e[2] = free_head        (first snap in free list)
 *   back = 0x534E4150       (magic: ASCII "SNAP")
 *   data = checksum          (FNV-1a of all data snaps)
 *
 * File layout:
 *   [header snap: 32 bytes]
 *   [snap 0:      32 bytes]
 *   [snap 1:      32 bytes]
 *   ...
 *   [snap N-1:    32 bytes]
 *
 * Total file size: (snap_count + 1) × 32 bytes.
 * Ingestion: mmap or read directly into arena. Zero deserialization.
 * The file IS the memory image.
 */
#define SNAP_MAGIC  0x534E4150u   /* "SNAP" in ASCII (little-endian) */
#define SNAP_VERSION 1u

/* ═══════════════════════════════════════════════════════════════════════
 * X. BOOT SEQUENCE
 * ═══════════════════════════════════════════════════════════════════════
 *
 * The kernel is 4 snaps. 128 bytes.
 *
 *   Snap 0: ROOT (ROTATE, kernel)
 *     ├── e[0] → Snap 1: SCHEDULER (ROTATE, kernel)
 *     ├── e[1] → Snap 2: ALLOCATOR (SPLT, kernel)
 *     └── e[2] → Snap 3: CLOCK     (OSC, kernel)
 *
 *   ROOT cycles through the three kernel services each tick.
 *   SCHEDULER cycles through user processes.
 *   ALLOCATOR manages the free list via SPLIT dynamics.
 *   CLOCK generates periodic timer interrupts via OSCILLATOR dynamics.
 *
 *   Boot:
 *     1. Load .snap file → mmap snap arena.
 *     2. Seed active wave with root snap (from header.e[1]).
 *     3. Loop: oicos_tick(&os) forever.
 *
 *   That's it. No init process. No bootloader. No BIOS.
 *   The SNAG IS the kernel. The cascade IS the execution.
 */

/* ═══════════════════════════════════════════════════════════════════════
 * XI. VERIFICATION
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Self-test of all microcode invariants. Returns true iff all pass.
 */
static inline bool oicos_verify(void) {
    /* 1. Snap size */
    if (sizeof(snap_t) != 32) return false;

    /* 2. Endomorphism closure: all outputs in {0,1,2} */
    for (int i = 0; i < 27; i++)
        for (int s = 0; s < 3; s++)
            if (ENDO[i][s] > 2) return false;

    /* 3. Endomorphism rank consistency */
    for (int i = 0; i < 27; i++) {
        uint8_t seen[3] = {0,0,0};
        for (int s = 0; s < 3; s++) seen[ENDO[i][s]] = 1;
        uint8_t im = (uint8_t)(seen[0] + seen[1] + seen[2]);
        if (im != ENDO_RANK[i]) return false;
    }

    /* 4. OIC routing monotonicity: output rank ≤ input rank */
    for (int r = 0; r < 3; r++)
        for (int d = 0; d < 2; d++)
            for (int c = 0; c < 7; c++)
                if (OIC_ROUTE[r][d][c] > r) return false;

    /* 5. S₃ Cayley table: every row is a permutation of {0..5} */
    for (int f = 0; f < 6; f++) {
        uint8_t seen[6] = {0,0,0,0,0,0};
        for (int g = 0; g < 6; g++) {
            uint8_t v = ALU_S3[f][g];
            if (v > 5) return false;
            seen[v] = 1;
        }
        for (int k = 0; k < 6; k++)
            if (!seen[k]) return false;
    }

    /* 6. Canonical triples produce correct dynamics under cyclic tape.
     *    For each cell, compose g = f₂∘f₁∘f₀ and verify rank + spin.
     */
    for (int cell = 0; cell < 7; cell++) {
        uint16_t sq = CANON_SQ[cell];
        uint8_t f0 = snap_q(sq, 0);
        uint8_t f1 = snap_q(sq, 1);
        uint8_t f2 = snap_q(sq, 2);

        /* Compose: g(s) = f₂(f₁(f₀(s))) */
        uint8_t g[3];
        for (int s = 0; s < 3; s++)
            g[s] = ENDO[f2][ENDO[f1][ENDO[f0][s]]];

        /* Rank = image size of g */
        uint8_t seen[3] = {0,0,0};
        for (int s = 0; s < 3; s++) seen[g[s]] = 1;
        uint8_t rank = (uint8_t)(seen[0] + seen[1] + seen[2]);

        /* Spin = max cycle length in g */
        uint8_t spin = 0;
        uint8_t visited[3] = {0,0,0};
        for (int s = 0; s < 3; s++) {
            if (visited[s]) continue;
            uint8_t len = 0, cur = (uint8_t)s;
            while (!visited[cur]) {
                visited[cur] = 1;
                cur = g[cur];
                len++;
            }
            /* Only count if we returned to start (true cycle) */
            if (cur == s && g[s] != s) {
                if (len > spin) spin = len;
            }
        }

        /* Verify cell→rank mapping */
        if (CELL_RANK[cell] == RANK_K && rank != 1) return false;
        if (CELL_RANK[cell] == RANK_S && rank != 2) return false;
        if (CELL_RANK[cell] == RANK_P && rank != 3) return false;

        /* Verify spin for cells that constrain it */
        if (cell == CELL_ROT && spin != 3) return false;
        if (cell == CELL_OSC && spin != 2) return false;
        if (cell == CELL_SWP && spin != 2) return false;
        if (cell == CELL_ID  && spin != 0) return false;
    }

    /* 7. snap_step output closure: output trit always in {0,1,2} */
    for (int cell = 0; cell < 7; cell++) {
        snap_t s = snap_make((uint8_t)cell, DOM_CYC,
                             SNAP_NIL, SNAP_NIL, SNAP_NIL, 0);
        for (int trit = 0; trit < 3; trit++) {
            for (int st = 0; st < 3; st++) {
                s.sr = PACK_SR((uint8_t)st, 0, STAT_LIVE);
                snap_result_t r = snap_step(&s, (uint8_t)trit);
                if (r.out > 2) return false;
                if (snap_st(&s) > 2) return false;
            }
        }
    }

    /* 8. ROTATE cycles: 3 consecutive steps with same input produce
     *    outputs 0, 1, 2 (Moore machine: output reflects current state).
     *    State advances 0→1→2→0 via CW rotation quark e7={1,2,0}. */
    {
        snap_t rot = snap_make(CELL_ROT, DOM_CYC,
                               SNAP_NIL, SNAP_NIL, SNAP_NIL, 0);
        rot.sr = PACK_SR(0, 0, STAT_LIVE);

        /* Output = identity(current_state), state = rotate(current_state) */
        snap_result_t r0 = snap_step(&rot, 0);  /* out=0, state: 0→1 */
        snap_result_t r1 = snap_step(&rot, 0);  /* out=1, state: 1→2 */
        snap_result_t r2 = snap_step(&rot, 0);  /* out=2, state: 2→0 */

        if (r0.out != 0 || r1.out != 1 || r2.out != 2) return false;
        if (snap_st(&rot) != 0) return false;  /* returned to start */
    }

    /* 9. Rank degradation: RANK_P → apply NULL → RANK_K */
    {
        snap_t dst = snap_make(CELL_ID, DOM_CYC,
                               SNAP_NIL, SNAP_NIL, SNAP_NIL, 0);
        snap_degrade(&dst, CELL_NULL);
        if (snap_rank(&dst) != RANK_K) return false;
    }

    /* 10. Rank degradation: RANK_P → apply ID → RANK_P (preserved) */
    {
        snap_t dst = snap_make(CELL_ID, DOM_CYC,
                               SNAP_NIL, SNAP_NIL, SNAP_NIL, 0);
        snap_degrade(&dst, CELL_ID);
        if (snap_rank(&dst) != RANK_P) return false;
    }

    /* All invariants hold. Microcode ROM verified. */
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Name Tables — shared across all OICOS tools
 * ═══════════════════════════════════════════════════════════════════════ */

static const char * const OICOS_CELL_NAME[7] = {
    "NULL", "CLPS", "SPLT", "OSC", "ID", "SWP", "ROT"
};

static const char * const OICOS_DOM_NAME[6] = {
    "CYC", "CON", "ALT", "REV", "BIN", "IO"
};

static const char * const OICOS_RANK_NAME[3] = { "K", "S", "P" };

static const char * const OICOS_STAT_NAME[4] = { "FREE", "IDLE", "LIVE", "LOCK" };

#endif /* OICOS_H */
