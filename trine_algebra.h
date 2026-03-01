/* =====================================================================
 * TRINE ALGEBRA — Standalone Algebraic Foundation
 * Ternary Resonance Interference Network Embedding
 * =====================================================================
 *
 * This file is the COMPLETE, SELF-CONTAINED algebraic core for TRINE.
 * It has ZERO dependencies on oicos.h or any other OICOS header.
 * Every constant, table, struct, and function needed for ternary
 * algebraic computation is defined here.
 *
 * What this file contains:
 *   1. The snap_t struct      — 32-byte universal data structure
 *   2. ENDO[27][3]            — All 27 endomorphisms of {0,1,2}
 *   3. ENDO_RANK[27]          — Image size of each endomorphism
 *   4. OIC_ROUTE[3][2][7]     — Rank routing matrix (monotonic)
 *   5. DTRIT[6][6]            — Domain clock trit patterns
 *   6. ALU_S3[6][6]           — S3 Cayley composition table
 *   7. ALU_CHIRAL[3][3]       — Chirality composition
 *   8. VALID[3][4]            — Quark validity matrix
 *   9. snap_step()            — O(1) atomic computation (0 branches)
 *  10. snap_step_adaptive()   — Step with lumen/crystallize/decay
 *  11. Cascade engine          — Level-synchronous BFS wave propagation
 *  12. Snap file I/O           — Read/write .snap binary format
 *  13. Self-verification       — Microcode ROM invariant checker
 *
 * ENCODING CONVENTION
 *   Endomorphism index: i = f(0) + 3*f(1) + 9*f(2)
 *   Base-3 little-endian in state-space {0,1,2}.
 *
 *   Key endomorphisms:
 *     e0  = {0,0,0} constant-0      e13 = {1,1,1} constant-1
 *     e26 = {2,2,2} constant-2      e21 = {0,1,2} IDENTITY
 *     e7  = {1,2,0} ROTATE CW       e11 = {2,0,1} ROTATE CCW
 *     e5  = {2,1,0} SWAP(0,2)       e15 = {0,2,1} SWAP(1,2)
 *     e19 = {1,0,2} SWAP(0,1)
 *
 * DERIVATION
 *   All constants derived from formal verification of 387,420,489
 *   ternary endomorphisms collapsing to 7 Myhill-Nerode equivalence
 *   classes (cells). The 270-byte ROM is the complete specification.
 *   2,163 frozen laws, zero exceptions.
 *
 * ===================================================================== */

#ifndef TRINE_ALGEBRA_H
#define TRINE_ALGEBRA_H

#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

/* =====================================================================
 * I. CONSTANTS
 * =====================================================================
 *
 * The Seven Cells — Myhill-Nerode irreducible equivalence classes.
 * 387,420,489 particles collapse to exactly 7 dynamical classes.
 *
 *   Rank K (absorber):    NULL
 *   Rank S (lossy):       COLLAPSE, SPLIT, OSCILLATOR
 *   Rank P (surjective):  IDENTITY, SWAP, ROTATE
 */

#define TRINE_CELL_NULL   0   /* Absorber:   rank K, spin 0 — halt/ground      */
#define TRINE_CELL_CLPS   1   /* Collapse:   rank S, spin 0, 1 basin — join    */
#define TRINE_CELL_SPLT   2   /* Split:      rank S, spin 0, 2 basins — fork   */
#define TRINE_CELL_OSC    3   /* Oscillator: rank S, spin 2 — timer/clock      */
#define TRINE_CELL_ID     4   /* Identity:   rank P, spin 0 — buffer/relay     */
#define TRINE_CELL_SWP    5   /* Swap:       rank P, spin 2 — exchange/switch  */
#define TRINE_CELL_ROT    6   /* Rotate:     rank P, spin 3 — schedule/cycle   */

/* The Rank Lattice — privilege levels. MONOTONICALLY NON-INCREASING.
 * Once rank drops, it never rises. This is the security model.
 *
 *   RANK_P --> RANK_S --> RANK_K
 *   kernel     user       dead
 *   S3 closed  lossy OK   absorber
 */
#define TRINE_RANK_K   0   /* Constant:    dead / information destroyed       */
#define TRINE_RANK_S   1   /* Semi:        user mode / lossy gates permitted  */
#define TRINE_RANK_P   2   /* Permutative: kernel mode / S3 group closure     */

/* Clock Domains — determine the input trit sequence for each snap. */
#define TRINE_DOM_CYC  0   /* Cyclic:      [0,1,2,0,1,2,...]                 */
#define TRINE_DOM_CON  1   /* Constant:    [0,0,0,0,0,0,...]                 */
#define TRINE_DOM_ALT  2   /* Alternating: [0,1,0,1,0,1,...]                 */
#define TRINE_DOM_REV  3   /* Reversed:    [2,1,0,2,1,0,...]                 */
#define TRINE_DOM_BIN  4   /* Binary:      application-defined                */
#define TRINE_DOM_IO   5   /* I/O:         channel-routed input/output        */

/* Snap Status — lifecycle state of each snap. */
#define TRINE_STAT_FREE   0   /* On free list — reclaimable                  */
#define TRINE_STAT_IDLE   1   /* Alive, not scheduled — dormant              */
#define TRINE_STAT_LIVE   2   /* In active or pending wave — executing       */
#define TRINE_STAT_LOCK   3   /* Blocked in synchronization primitive        */

/* Sentinel: null edge destination (no connection). */
#define TRINE_SNAP_NIL    UINT32_MAX

/* Flags in the SR reserved bits [1:0]. */
#define TRINE_FLAG_BCAST  0x01U  /* bit 0: broadcast — fan-out to ALL edges  */
#define TRINE_FLAG_ADAPT  0x02U  /* bit 1: adaptive data present (info only) */

/* .snap binary format magic and version. */
#define TRINE_SNAP_MAGIC    0x534E4150u   /* "SNAP" in ASCII (little-endian) */
#define TRINE_SNAP_VERSION  1u

/* Adaptive mutation modes — encoded in snap data field. */
#define TRINE_ADAPT_NONE        0
#define TRINE_ADAPT_LUMEN       1
#define TRINE_ADAPT_CRYSTALLIZE 2
#define TRINE_ADAPT_DECAY       3

/* =====================================================================
 * II. THE SNAP (32 bytes)
 * =====================================================================
 *
 * The universal and only data structure. Everything is a snap.
 * 2 snaps fit in one 64-byte cache line.
 *
 * Layout:
 *   Offset  Size  Field   Bit Layout
 *   0       2     sq      f0[4:0] | f1[9:5] | f2[14:10] | 0[15]
 *   2       2     oq      o0[4:0] | o1[9:5] | o2[14:10] | 0[15]
 *   4       1     hdr     cell[7:5] | rank[4:3] | dom[2:0]
 *   5       1     sr      st[7:6] | out[5:4] | stat[3:2] | flags[1:0]
 *   6       2     gen     16-bit epoch counter (cascade dedup marker)
 *   8       12    e[3]    trit-routed edge destinations (snap indices)
 *   20      4     back    parent (live) / next free (free list)
 *   24      8     data    64-bit payload
 */
typedef struct {
    uint16_t sq;        /* state quarks:  f0|f1|f2 packed 5:5:5:1           */
    uint16_t oq;        /* output quarks: o0|o1|o2 packed 5:5:5:1           */
    uint8_t  hdr;       /* cell:3 | rank:2 | domain:3                       */
    uint8_t  sr;        /* state:2 | output:2 | status:2 | flags:2          */
    uint16_t gen;       /* generation / epoch / cascade dedup marker         */
    uint32_t e[3];      /* edge[trit] -> destination snap index              */
    uint32_t back;      /* parent snap (LIVE) / next free snap (FREE)        */
    uint64_t data;      /* payload: pointer, value, instruction, hash        */
} __attribute__((packed, aligned(32))) trine_snap_t;

/* =====================================================================
 * III. FIELD ACCESS (all branchless, O(1))
 * ===================================================================== */

/* --- Quark packing: three 5-bit endomorphism indices into uint16 --- */
#define TRINE_PACK_Q(f0,f1,f2)  ((uint16_t)( \
    ((uint16_t)(f0)) | (((uint16_t)(f1)) << 5U) | (((uint16_t)(f2)) << 10U)))

/* Extract endomorphism index for the given input trit (0, 1, or 2). */
static inline uint8_t trine_snap_q(uint16_t packed, uint8_t trit) {
    return (uint8_t)((packed >> (trit * 5U)) & 0x1FU);
}

/* --- Header: cell[7:5] | rank[4:3] | dom[2:0] --- */
#define TRINE_PACK_HDR(cell,rank,dom)  ((uint8_t)( \
    (((unsigned)(cell)) << 5U) | (((unsigned)(rank)) << 3U) | ((unsigned)(dom))))

static inline uint8_t trine_snap_cell(const trine_snap_t *s) {
    return (uint8_t)(s->hdr >> 5U);
}

static inline uint8_t trine_snap_rank(const trine_snap_t *s) {
    return (uint8_t)((s->hdr >> 3U) & 0x03U);
}

static inline uint8_t trine_snap_dom(const trine_snap_t *s) {
    return (uint8_t)(s->hdr & 0x07U);
}

/* --- State register: st[7:6] | out[5:4] | stat[3:2] | flags[1:0] --- */
#define TRINE_PACK_SR(st,out,stat)  ((uint8_t)( \
    (((unsigned)(st)) << 6U) | (((unsigned)(out)) << 4U) | (((unsigned)(stat)) << 2U)))

static inline uint8_t trine_snap_st(const trine_snap_t *s) {
    return (uint8_t)(s->sr >> 6U);
}

static inline uint8_t trine_snap_out(const trine_snap_t *s) {
    return (uint8_t)((s->sr >> 4U) & 0x03U);
}

static inline uint8_t trine_snap_stat(const trine_snap_t *s) {
    return (uint8_t)((s->sr >> 2U) & 0x03U);
}

/* Set FSM state and output trits in the SR register (preserves flags). */
static inline void trine_snap_set_fsm(trine_snap_t *s, uint8_t st, uint8_t out) {
    s->sr = (uint8_t)(((unsigned)st << 6U) | ((unsigned)out << 4U) | (s->sr & 0x0FU));
}

/* Set status bits in the SR register (preserves st, out, flags). */
static inline void trine_snap_set_stat(trine_snap_t *s, uint8_t stat) {
    s->sr = (uint8_t)((s->sr & 0xF3U) | ((unsigned)stat << 2U));
}

/* --- Broadcast flag --- */
static inline bool trine_snap_bcast(const trine_snap_t *s) {
    return (s->sr & TRINE_FLAG_BCAST) != 0U;
}

static inline void trine_snap_set_bcast(trine_snap_t *s, bool on) {
    s->sr = (uint8_t)((s->sr & 0xFCU) | (on ? TRINE_FLAG_BCAST : 0U));
}

/* --- I/O channel convention in data field ---
 *   data[7:0]  = input channel  (0-255)
 *   data[15:8] = output channel (0-255)
 */
#define TRINE_IO_IN(s)   ((uint8_t)((s)->data & 0xFFU))
#define TRINE_IO_OUT(s)  ((uint8_t)(((s)->data >> 8U) & 0xFFU))

/* --- Adaptive mutation field access ---
 *   data[43:40] = mode      (4 bits: 0=none, 1=lumen, 2=crystallize, 3=decay)
 *   data[39:36] = target    (4 bits: cell type to become, 0-6)
 *   data[35:20] = threshold (16 bits: step count before mutation)
 *   data[19:4]  = counter   (16 bits: current step count)
 *   data[3:0]   = reserved
 */
#define TRINE_ADAPT_MODE(s)    ((uint8_t)(((s)->data >> 40U) & 0x0FU))
#define TRINE_ADAPT_TARGET(s)  ((uint8_t)(((s)->data >> 36U) & 0x0FU))
#define TRINE_ADAPT_THRESH(s)  ((uint16_t)(((s)->data >> 20U) & 0xFFFFU))
#define TRINE_ADAPT_COUNTER(s) ((uint16_t)(((s)->data >> 4U) & 0xFFFFU))

/* =====================================================================
 * IV. THE ROM (270 bytes total)
 * =====================================================================
 *
 * These tables are the complete specification of the system.
 * Every computation ultimately reduces to lookups in these tables.
 * All values derived from formal verification — not designed, discovered.
 */

/* --- 4a. Endomorphism Transition Matrix (81 bytes) ---
 *
 * ENDO[i][s] = f(s), where f is the endomorphism at index i.
 * Index encoding: i = f(0) + 3*f(1) + 9*f(2)  (base-3 little-endian).
 *
 * This table IS the CPU. Every computation in the system ultimately
 * reduces to lookups in this 81-byte matrix.
 *
 * The 27 entries cover ALL possible endomorphisms of {0,1,2}:
 *   - 3 constants (rank 1):  e0={0,0,0}  e13={1,1,1}  e26={2,2,2}
 *   - 18 lossy (rank 2):     e.g. e1={1,0,0}  e3={0,1,0}  etc.
 *   - 6 permutations (rank 3, the S3 group):
 *       e21=ID  e5=SWP02  e15=SWP12  e19=SWP01  e7=ROTCW  e11=ROTCCW
 */
static const uint8_t TRINE_ENDO[27][3] = {
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
 * TRINE_ENDO_RANK[i] = |image(e_i)| in {1, 2, 3}
 *   Rank 1 (constant):    indices {0, 13, 26}        — 3 endomorphisms
 *   Rank 3 (surjective):  indices {5, 7, 11, 15, 19, 21} — 6 (the S3 group)
 *   Rank 2 (everything else):                         — 18 endomorphisms
 */
static const uint8_t TRINE_ENDO_RANK[27] = {
    1,2,2, 2,2,3, 2,3,2,
    2,2,3, 2,1,2, 3,2,2,
    2,3,2, 3,2,2, 2,2,1
};

/* --- 4c. OIC Routing Matrix (42 bytes) ---
 *
 * TRINE_OIC_ROUTE[rank][domain_class][cell] -> new_rank
 *
 * domain_class: 0 = varied (CYC/ALT/REV/BIN), 1 = constant (CON)
 *
 * INVARIANT: TRINE_OIC_ROUTE[r][d][c] <= r   (monotonicity)
 *
 * This is THE security model. Rank can only decrease, never increase.
 * Formally verified via SMT solver (Z3). Zero violations.
 */
static const uint8_t TRINE_OIC_ROUTE[3][2][7] = {
    /* RANK_K: absorber — everything stays dead */
    { {0,0,0,0,0,0,0}, {0,0,0,0,0,0,0} },
    /* RANK_S: NULL kills; lossy gates maintain; S3 gates cap at RANK_S */
    { {0,1,1,1,1,1,1}, {0,0,1,1,1,1,1} },
    /* RANK_P: full lattice; S3 gates preserve RANK_P */
    { {0,1,1,1,2,2,2}, {0,0,1,1,2,2,2} }
};

/* Helper: map domain to routing class (constant vs varied). */
static inline uint8_t trine_route_class(uint8_t dom) {
    return (dom == TRINE_DOM_CON) ? 1 : 0;
}

/* --- 4d. Domain Input Trit Pattern (36 bytes) ---
 *
 * TRINE_DTRIT[domain][tick % 6] -> input trit
 *
 * Period 6 = LCM(1, 2, 3) covers all domain cycles.
 * Each domain defines a repeating trit sequence that drives
 * its snaps. The input trit selects which quark fires.
 */
static const uint8_t TRINE_DTRIT[6][6] = {
    {0,1,2,0,1,2},   /* CYC: repeating 0,1,2                               */
    {0,0,0,0,0,0},   /* CON: always 0                                       */
    {0,1,0,1,0,1},   /* ALT: alternating 0,1                                */
    {2,1,0,2,1,0},   /* REV: reversed 2,1,0                                 */
    {0,1,0,1,0,1},   /* BIN: default pattern; override at runtime            */
    {0,0,0,0,0,0}    /* IO: overridden at runtime by channel read            */
};

/* --- 4e. S3 Cayley Composition Table (36 bytes) ---
 *
 * TRINE_ALU_S3[f][g] = f composed with g (standard composition)
 *
 * Dense index:  0=ID(e21)   1=SWP02(e5)   2=SWP12(e15)
 *               3=SWP01(e19)  4=ROTCW(e7)  5=ROTCCW(e11)
 *
 * This is a perfect Latin square: 100% informational efficiency.
 * H(X) = log2(6) ~ 2.585 bits. Zero structural waste.
 */
static const uint8_t TRINE_ALU_S3[6][6] = {
    {0,1,2,3,4,5}, {1,0,5,4,3,2}, {2,4,0,5,1,3},
    {3,5,4,0,2,1}, {4,2,3,1,5,0}, {5,3,1,2,0,4}
};

/* --- 4f. Chirality Composition (9 bytes) ---
 *
 * TRINE_ALU_CHIRAL[left][right] -> composed chirality
 *   0 = ACHIRAL, 1 = CW, 2 = CCW
 */
static const uint8_t TRINE_ALU_CHIRAL[3][3] = {
    {0,0,0}, {0,2,0}, {0,0,1}
};

/* --- 4g. Quark Validity Matrix (12 bytes) ---
 *
 * TRINE_VALID[snap_rank][quark_rank] -> can this quark appear at this rank?
 *
 * Enforces the rank lattice: higher-privilege snaps require
 * higher-rank (more information-preserving) quarks.
 */
static const uint8_t TRINE_VALID[3][4] = {
    /* quark_rank:   0(n/a) 1(K)  2(S)  3(P) */
    /* RANK_K: */  { 0,     1,    1,    1 },   /* Any quark; snap is dead  */
    /* RANK_S: */  { 0,     0,    1,    1 },   /* No constants             */
    /* RANK_P: */  { 0,     0,    0,    1 }    /* Surjective only          */
};

/* --- 4h. S3 <-> Endomorphism Index Maps ---
 *
 * Convert between the dense S3 index (0-5) and the sparse
 * endomorphism index (0-26). Only 6 of 27 endomorphisms are
 * permutations (the S3 group).
 */
static const uint8_t TRINE_S3_TO_ENDO[6] = { 21, 5, 15, 19, 7, 11 };

static const uint8_t TRINE_ENDO_TO_S3[27] = {
    0xFF,0xFF,0xFF, 0xFF,0xFF,   1, 0xFF,   4, 0xFF,
    0xFF,0xFF,   5, 0xFF,0xFF,0xFF,    2, 0xFF, 0xFF,
    0xFF,   3,0xFF,    0,0xFF,0xFF, 0xFF,0xFF, 0xFF
};

/* --- 4i. Cell -> Natural Rank ---
 *
 * Each cell type has an intrinsic rank determined by its
 * endomorphism's image size.
 */
static const uint8_t TRINE_CELL_RANK[7] = {
    TRINE_RANK_K, TRINE_RANK_S, TRINE_RANK_S, TRINE_RANK_S,
    TRINE_RANK_P, TRINE_RANK_P, TRINE_RANK_P
};

/* --- 4j. Canonical State Quarks (14 bytes) ---
 *
 * The canonical quark triple for each cell type. Under cyclic tape,
 * composing g = f2 o f1 o f0 produces the cell's signature dynamics:
 *
 *   NULL:  {e0,  e0,  e0}  -> g=e0  rank 1, spin 0
 *   CLPS:  {e9,  e21, e21} -> g=e9  rank 2, spin 0, 1 basin
 *   SPLT:  {e3,  e21, e21} -> g=e3  rank 2, spin 0, 2 basins
 *   OSC:   {e1,  e21, e21} -> g=e1  rank 2, spin 2, 0<->1 cycle
 *   ID:    {e21, e21, e21} -> g=e21 rank 3, spin 0, 3 fixed pts
 *   SWP:   {e5,  e21, e21} -> g=e5  rank 3, spin 2, swap 0<->2
 *   ROT:   {e7,  e21, e21} -> g=e7  rank 3, spin 3, CW rotation
 */
static const uint16_t TRINE_CANON_SQ[7] = {
    TRINE_PACK_Q( 0,  0,  0),   /* NULL */
    TRINE_PACK_Q( 9, 21, 21),   /* CLPS */
    TRINE_PACK_Q( 3, 21, 21),   /* SPLT */
    TRINE_PACK_Q( 1, 21, 21),   /* OSC  */
    TRINE_PACK_Q(21, 21, 21),   /* ID   */
    TRINE_PACK_Q( 5, 21, 21),   /* SWP  */
    TRINE_PACK_Q( 7, 21, 21)    /* ROT  */
};

/* Default output quark: identity (faithful reproduction of state). */
#define TRINE_CANON_OQ  TRINE_PACK_Q(21, 21, 21)

/* =====================================================================
 * V. THE STEP (Atomic Execution - The Single Operation)
 * =====================================================================
 *
 * Every computation reduces to this function.
 * Processes, scheduling, IPC, memory management — all are cascades
 * of trine_snap_step calls through a topology of snaps.
 *
 * Cost: 2 bitfield extractions + 2 ROM lookups + 1 array index = O(1).
 * Branches: 0.
 * Hot-path footprint: 113 bytes (81 ROM + 32 snap).
 */

/* Result of a single step: output trit and destination snap index. */
typedef struct {
    uint8_t  out;          /* emitted output trit in {0,1,2}                 */
    uint32_t dst;          /* destination snap index (TRINE_SNAP_NIL if none)*/
} trine_result_t;

/* The single operation. Zero branches, O(1), deterministic.
 *
 * Algorithm:
 *   1. Use input_trit to select which state and output quarks fire
 *   2. Read current FSM state (2 bits)
 *   3. Compute output = ENDO[output_quark][current_state]
 *   4. Compute next_state = ENDO[state_quark][current_state]
 *   5. Update FSM register
 *   6. Route: output trit selects destination edge
 */
static inline trine_result_t trine_snap_step(trine_snap_t *s, uint8_t input_trit) {
    /* 1. Extract active endomorphism indices from quark packs */
    uint8_t s_endo = trine_snap_q(s->sq, input_trit);
    uint8_t o_endo = trine_snap_q(s->oq, input_trit);

    /* 2. Read current FSM state */
    uint8_t cur = trine_snap_st(s);

    /* 3. Compute output FIRST (depends on pre-update state) */
    uint8_t out = TRINE_ENDO[o_endo][cur];

    /* 4. Compute next state */
    uint8_t nxt = TRINE_ENDO[s_endo][cur];

    /* 5. Update FSM register */
    trine_snap_set_fsm(s, nxt, out);

    /* 6. Route: output trit selects destination edge */
    trine_result_t r;
    r.out = out;
    r.dst = s->e[out];
    return r;
}

/* Rank degradation — applied at the receiving end during cascade.
 * Source snap's cell type determines how destination's rank degrades.
 * This enforces the monotonicity invariant: rank can only decrease. */
static inline void trine_snap_degrade(trine_snap_t *dst, uint8_t source_cell) {
    uint8_t r  = trine_snap_rank(dst);
    uint8_t dc = trine_route_class(trine_snap_dom(dst));
    uint8_t nr = TRINE_OIC_ROUTE[r][dc][source_cell];
    dst->hdr = TRINE_PACK_HDR(trine_snap_cell(dst), nr, trine_snap_dom(dst));
}

/* =====================================================================
 * V-B. ADAPTIVE STEP (Optional Mutation)
 * =====================================================================
 *
 * Wraps trine_snap_step with a lumen/crystallization/decay model.
 * Backward-compatible: mode==0 returns immediately (zero overhead).
 *
 * Data field encoding (when adaptive):
 *   data[43:40] = mode      (0=none, 1=lumen, 2=crystallize, 3=decay)
 *   data[39:36] = target    (cell type to become, 0-6)
 *   data[35:20] = threshold (step count before mutation)
 *   data[19:4]  = counter   (current step count, starts at 0)
 *   data[3:0]   = reserved
 */
static inline trine_result_t trine_snap_step_adaptive(trine_snap_t *s,
                                                       uint8_t input_trit) {
    trine_result_t r = trine_snap_step(s, input_trit);

    /* Fast path: no adaptivity (zero overhead for mode==0) */
    uint8_t mode = TRINE_ADAPT_MODE(s);
    if (mode == 0) return r;

    /* Increment counter */
    uint16_t counter = TRINE_ADAPT_COUNTER(s) + 1;
    s->data = (s->data & ~(0xFFFFULL << 4)) | ((uint64_t)counter << 4);

    /* Check threshold */
    if (counter >= TRINE_ADAPT_THRESH(s)) {
        uint8_t target = TRINE_ADAPT_TARGET(s);

        if (mode == TRINE_ADAPT_LUMEN || mode == TRINE_ADAPT_CRYSTALLIZE) {
            /* Mutate: change cell type, preserve edges and domain */
            if (target <= 6) {
                s->sq  = TRINE_CANON_SQ[target];
                s->oq  = TRINE_CANON_OQ;
                s->hdr = TRINE_PACK_HDR(target, TRINE_CELL_RANK[target],
                                         trine_snap_dom(s));
            }
            /* Reset counter */
            s->data &= ~(0xFFFFULL << 4);
        } else if (mode == TRINE_ADAPT_DECAY) {
            /* Decay to NULL — snap dies */
            s->sq  = TRINE_CANON_SQ[TRINE_CELL_NULL];
            s->oq  = TRINE_CANON_OQ;
            s->hdr = TRINE_PACK_HDR(TRINE_CELL_NULL, TRINE_RANK_K,
                                     trine_snap_dom(s));
            trine_snap_set_stat(s, TRINE_STAT_IDLE);
            /* Clear mode (one-shot) */
            s->data &= ~(0x0FULL << 40);
        }
    }

    return r;
}

/* =====================================================================
 * VI. CONSTRUCTORS
 * ===================================================================== */

/* Create a canonical snap of the given cell type.
 * All quarks set to canonical values, status = IDLE, FSM state = 0. */
static inline trine_snap_t trine_snap_make(uint8_t cell, uint8_t dom,
                                            uint32_t e0, uint32_t e1,
                                            uint32_t e2, uint64_t payload) {
    trine_snap_t s;
    memset(&s, 0, sizeof(s));
    s.sq   = TRINE_CANON_SQ[cell];
    s.oq   = TRINE_CANON_OQ;
    s.hdr  = TRINE_PACK_HDR(cell, TRINE_CELL_RANK[cell], dom);
    s.sr   = TRINE_PACK_SR(0, 0, TRINE_STAT_IDLE);
    s.e[0] = e0;
    s.e[1] = e1;
    s.e[2] = e2;
    s.data = payload;
    return s;
}

/* Validate snap invariants: quarks compatible with declared rank. */
static inline bool trine_snap_valid(const trine_snap_t *s) {
    uint8_t r = trine_snap_rank(s);
    return TRINE_VALID[r][TRINE_ENDO_RANK[trine_snap_q(s->sq, 0)]]
        && TRINE_VALID[r][TRINE_ENDO_RANK[trine_snap_q(s->sq, 1)]]
        && TRINE_VALID[r][TRINE_ENDO_RANK[trine_snap_q(s->sq, 2)]];
}

/* =====================================================================
 * VII. CHECKSUM (FNV-1a)
 * =====================================================================
 *
 * Used in .snap file headers to validate arena integrity.
 * Standard FNV-1a 64-bit hash.
 */
static inline uint64_t trine_fnv1a(const void *data, size_t len) {
    const uint8_t *p = (const uint8_t *)data;
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < len; i++)
        h = (h ^ p[i]) * 0x100000001b3ULL;
    return h;
}

/* =====================================================================
 * VIII. THE CASCADE (The Execution Model)
 * =====================================================================
 *
 * One tick = one BFS wave through the snap topology (SNAG).
 * Every active snap executes exactly one step per tick.
 * The wave propagates level-synchronously: tick t+1 depends only on t.
 *
 * The cascade IS the kernel. There is no scheduler function, no
 * interrupt handler, no main loop separate from the topology.
 * The SNAG topology determines execution order.
 * The wave IS the program counter.
 */

/* Cascade engine state. */
typedef struct {
    trine_snap_t *snaps;            /* Contiguous snap arena               */
    uint32_t      snap_count;       /* Total snaps in arena                */
    uint32_t     *active;           /* Current wave (snap indices)         */
    uint32_t     *pending;          /* Next wave (populated during tick)   */
    uint32_t      active_count;     /* Snaps in current wave               */
    uint32_t      pending_count;    /* Snaps in next wave                  */
    uint32_t      tick;             /* Global monotonic tick counter        */
} trine_cascade_t;

/* One tick of the cascade engine — the minimal version from oicos.h.
 *
 * Algorithm:
 *   1. For each snap in the active wave:
 *      a. Compute input trit from the snap's clock domain
 *      b. Execute snap_step (the single O(1) operation)
 *      c. If destination exists and is alive:
 *         - Apply rank degradation from source cell type
 *         - If destination not already in pending wave (gen dedup):
 *           add to pending wave
 *   2. Swap active/pending wave buffers
 */
static inline void trine_cascade_tick(trine_cascade_t *cas) {
    cas->pending_count = 0;
    cas->tick++;
    uint16_t cur_gen = (uint16_t)(cas->tick & 0xFFFFU);

    for (uint32_t i = 0; i < cas->active_count; i++) {
        uint32_t idx = cas->active[i];
        trine_snap_t *s = &cas->snaps[idx];

        /* Per-snap input trit from its clock domain */
        uint8_t trit = TRINE_DTRIT[trine_snap_dom(s)][cas->tick % 6U];

        /* Execute the single operation */
        trine_result_t r = trine_snap_step(s, trit);

        /* Rank degradation + deduplicated routing */
        if (r.dst != TRINE_SNAP_NIL && r.dst < cas->snap_count) {
            trine_snap_t *dst = &cas->snaps[r.dst];

            /* Degrade destination rank based on source cell type */
            trine_snap_degrade(dst, trine_snap_cell(s));

            /* Dedup: generation marker prevents duplicate wave entries */
            if (dst->gen != cur_gen && trine_snap_rank(dst) > TRINE_RANK_K) {
                dst->gen = cur_gen;
                cas->pending[cas->pending_count++] = r.dst;
            }
        }
    }

    /* Swap wave buffers */
    uint32_t *tmp = cas->active;
    cas->active  = cas->pending;
    cas->pending = tmp;
    cas->active_count = cas->pending_count;
}

/* Full cascade tick with I/O channel support, broadcast, and adaptive
 * mutations. This is the production-quality version matching oicos_run.c.
 *
 * io_channels: array of 256 uint8_t trit values for I/O domain snaps.
 *              Pass NULL to disable I/O channel handling.
 *
 * Returns the number of terminal emissions (dst == SNAP_NIL) this tick.
 * Terminal output trits are written to emit_buf if non-NULL (caller
 * must ensure emit_buf has room for at least active_count entries).
 */
static inline uint32_t trine_cascade_tick_full(trine_cascade_t *cas,
                                                uint8_t *io_channels,
                                                uint8_t *emit_buf,
                                                uint32_t *emit_count) {
    cas->pending_count = 0;
    cas->tick++;
    uint16_t cur_gen = (uint16_t)(cas->tick & 0xFFFFU);
    uint32_t emitted = 0;

    for (uint32_t i = 0; i < cas->active_count; i++) {
        uint32_t idx = cas->active[i];
        trine_snap_t *s = &cas->snaps[idx];

        /* Per-snap input trit selection (priority order):
         * 1. DOM_IO snaps: read from I/O channel array. The I/O trit
         *    also overrides the output to propagate data downstream.
         * 2. Signal carry: if data[23:16] is non-zero, use carried signal
         *    from parent snap (value is out+1, so subtract 1 to get trit)
         * 3. Default: use DTRIT clock domain pattern */
        uint8_t trit;
        int is_io = (trine_snap_dom(s) == TRINE_DOM_IO && io_channels);
        if (is_io) {
            uint8_t in_ch = TRINE_IO_IN(s);
            trit = io_channels[in_ch] % 3;
        } else {
            uint8_t carry = (uint8_t)((s->data >> 16) & 0xFF);
            if (carry > 0) {
                trit = (carry - 1) % 3;
                /* Clear the carry signal after reading */
                s->data &= ~((uint64_t)0xFF << 16);
            } else {
                trit = TRINE_DTRIT[trine_snap_dom(s)][cas->tick % 6U];
            }
        }

        /* Execute with adaptive mutation support */
        trine_result_t r = trine_snap_step_adaptive(s, trit);

        /* DOM_IO snaps: override output with the I/O channel trit.
         * This ensures the I/O data value propagates downstream through
         * the cascade, rather than being absorbed by the snap's internal
         * state machine. The snap's FSM state IS updated (for the
         * embedding readout), but the SIGNAL that travels downstream
         * is the raw I/O channel value. */
        if (is_io) {
            r.out = trit;
            uint8_t out_ch = TRINE_IO_OUT(s);
            io_channels[out_ch] = r.out;
        }

        if (trine_snap_bcast(s)) {
            /* Broadcast mode: fan out to ALL edges.
             * Signal propagation: write source output trit into
             * destination's data[23:16] as incoming signal. The next
             * tick reads this as a domain override for signal carry. */
            for (int edge = 0; edge < 3; edge++) {
                uint32_t dst_idx = s->e[edge];
                if (dst_idx != TRINE_SNAP_NIL && dst_idx < cas->snap_count) {
                    trine_snap_t *dst = &cas->snaps[dst_idx];
                    trine_snap_degrade(dst, trine_snap_cell(s));
                    if (dst->gen != cur_gen &&
                        trine_snap_rank(dst) > TRINE_RANK_K) {
                        dst->gen = cur_gen;
                        /* Carry signal: store (out+1) so 0 means "no signal" */
                        dst->data = (dst->data & ~((uint64_t)0xFF << 16)) |
                                    ((uint64_t)(r.out + 1) << 16);
                        cas->pending[cas->pending_count++] = dst_idx;
                    }
                }
            }
        } else if (r.dst == TRINE_SNAP_NIL) {
            /* Terminal: no destination — this is an output emission */
            if (emit_buf) {
                emit_buf[emitted] = r.out;
            }
            emitted++;
        } else if (r.dst < cas->snap_count) {
            /* Normal: single edge routing.
             * Signal propagation: carry output trit to destination. */
            trine_snap_t *dst = &cas->snaps[r.dst];
            trine_snap_degrade(dst, trine_snap_cell(s));
            if (dst->gen != cur_gen &&
                trine_snap_rank(dst) > TRINE_RANK_K) {
                dst->gen = cur_gen;
                dst->data = (dst->data & ~((uint64_t)0xFF << 16)) |
                            ((uint64_t)(r.out + 1) << 16);
                cas->pending[cas->pending_count++] = r.dst;
            }
        }
    }

    /* Swap wave buffers */
    uint32_t *tmp = cas->active;
    cas->active  = cas->pending;
    cas->pending = tmp;
    cas->active_count = cas->pending_count;

    if (emit_count) *emit_count = emitted;
    return emitted;
}

/* Initialize a cascade engine from an arena.
 * Allocates wave buffers. Seeds the active wave with all STAT_LIVE snaps.
 * Returns 0 on success, non-zero on allocation failure. */
static inline int trine_cascade_init(trine_cascade_t *cas,
                                      trine_snap_t *arena,
                                      uint32_t snap_count) {
    cas->snaps      = arena;
    cas->snap_count = snap_count;
    cas->tick       = 0;
    cas->active_count  = 0;
    cas->pending_count = 0;

    cas->active  = (uint32_t *)malloc((size_t)snap_count * sizeof(uint32_t));
    cas->pending = (uint32_t *)malloc((size_t)snap_count * sizeof(uint32_t));
    if (!cas->active || !cas->pending) {
        free(cas->active);
        free(cas->pending);
        cas->active = NULL;
        cas->pending = NULL;
        return 1;
    }

    /* Seed wave: all STAT_LIVE snaps (supports multi-root topologies) */
    for (uint32_t i = 0; i < snap_count; i++) {
        if (trine_snap_stat(&arena[i]) == TRINE_STAT_LIVE) {
            cas->active[cas->active_count++] = i;
        }
    }

    return 0;
}

/* Free wave buffers. Does NOT free the snap arena (caller owns it). */
static inline void trine_cascade_free(trine_cascade_t *cas) {
    free(cas->active);
    free(cas->pending);
    cas->active  = NULL;
    cas->pending = NULL;
    cas->active_count  = 0;
    cas->pending_count = 0;
}

/* Run the cascade for up to max_ticks, or until the wave is empty.
 * Returns the final tick count. Wave-empty means cascade completed. */
static inline uint32_t trine_cascade_run(trine_cascade_t *cas,
                                          uint32_t max_ticks) {
    for (uint32_t t = 0; t < max_ticks; t++) {
        if (cas->active_count == 0) break;
        trine_cascade_tick(cas);
    }
    return cas->tick;
}

/* Run the cascade with full I/O and emission support.
 * total_emit receives the total emission count across all ticks.
 * Returns the final tick count. */
static inline uint32_t trine_cascade_run_full(trine_cascade_t *cas,
                                               uint32_t max_ticks,
                                               uint8_t *io_channels,
                                               uint32_t *total_emit) {
    uint32_t total = 0;
    for (uint32_t t = 0; t < max_ticks; t++) {
        if (cas->active_count == 0) break;
        uint32_t e = 0;
        trine_cascade_tick_full(cas, io_channels, NULL, &e);
        total += e;
    }
    if (total_emit) *total_emit = total;
    return cas->tick;
}

/* =====================================================================
 * IX. SNAP FILE I/O
 * =====================================================================
 *
 * .snap binary format:
 *   [header snap: 32 bytes]   (metadata — IS a snap, self-referential)
 *   [snap 0:      32 bytes]
 *   [snap 1:      32 bytes]
 *   ...
 *   [snap N-1:    32 bytes]
 *
 * Header snap layout:
 *   sq   = 0                                    (unused)
 *   oq   = 0                                    (unused)
 *   hdr  = PACK_HDR(CELL_ID, RANK_P, DOM_CYC)  (metadata snap)
 *   sr   = 0                                    (unused)
 *   gen  = version number                       (format version)
 *   e[0] = snap_count                           (number of data snaps)
 *   e[1] = root_index                           (first snap in boot wave)
 *   e[2] = free_head                            (first snap in free list)
 *   back = 0x534E4150                           (magic: ASCII "SNAP")
 *   data = FNV-1a checksum of all data snaps
 *
 * Total file size: (snap_count + 1) * 32 bytes.
 * The file IS the memory image. Zero deserialization needed.
 */

/* Loaded snap image — arena plus header metadata. */
typedef struct {
    trine_snap_t *arena;
    uint32_t      snap_count;
    uint32_t      root_index;
    uint32_t      free_head;
} trine_snap_image_t;

/* Maximum number of snaps to load (safety limit: 32 MB). */
#define TRINE_MAX_SNAPS (1u << 20)

/* Load a .snap file from disk into a snap image.
 * Allocates arena memory (caller must free with trine_snap_image_free).
 * Returns 0 on success, non-zero on error (messages to stderr). */
static inline int trine_snap_file_load(const char *path,
                                        trine_snap_image_t *img) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "trine: cannot open '%s'\n", path);
        return 1;
    }

    /* Read header snap (32 bytes) */
    trine_snap_t hdr;
    if (fread(&hdr, 32, 1, f) != 1) {
        fprintf(stderr, "trine: truncated header in '%s'\n", path);
        fclose(f);
        return 1;
    }

    /* Validate magic */
    if (hdr.back != TRINE_SNAP_MAGIC) {
        fprintf(stderr, "trine: bad magic in '%s' (got 0x%08X, want 0x%08X)\n",
                path, hdr.back, TRINE_SNAP_MAGIC);
        fclose(f);
        return 1;
    }

    /* Validate version */
    if (hdr.gen != TRINE_SNAP_VERSION) {
        fprintf(stderr, "trine: version mismatch in '%s' (got %u, want %u)\n",
                path, hdr.gen, TRINE_SNAP_VERSION);
        fclose(f);
        return 1;
    }

    /* Extract header fields */
    img->snap_count = hdr.e[0];
    img->root_index = hdr.e[1];
    img->free_head  = hdr.e[2];

    /* Safety limit */
    if (img->snap_count > TRINE_MAX_SNAPS) {
        fprintf(stderr, "trine: snap_count %u exceeds max (%u) in '%s'\n",
                img->snap_count, TRINE_MAX_SNAPS, path);
        fclose(f);
        return 1;
    }

    /* Validate root index */
    if (img->root_index >= img->snap_count) {
        fprintf(stderr, "trine: root index %u out of range (count=%u) in '%s'\n",
                img->root_index, img->snap_count, path);
        fclose(f);
        return 1;
    }

    /* Allocate arena (32-byte aligned for cache efficiency) */
    img->arena = (trine_snap_t *)aligned_alloc(32,
                    (size_t)img->snap_count * sizeof(trine_snap_t));
    if (!img->arena) {
        fprintf(stderr, "trine: allocation failed for %u snaps\n",
                img->snap_count);
        fclose(f);
        return 1;
    }

    /* Read arena */
    if (fread(img->arena, 32, img->snap_count, f) != img->snap_count) {
        fprintf(stderr, "trine: truncated arena in '%s'\n", path);
        free(img->arena);
        img->arena = NULL;
        fclose(f);
        return 1;
    }

    /* Verify checksum */
    uint64_t ck = trine_fnv1a(img->arena,
                               (size_t)img->snap_count * sizeof(trine_snap_t));
    if (hdr.data != 0 && ck != hdr.data) {
        fprintf(stderr, "trine: checksum mismatch in '%s': "
                "expected %llu, got %llu\n",
                path, (unsigned long long)hdr.data, (unsigned long long)ck);
        free(img->arena);
        img->arena = NULL;
        fclose(f);
        return 1;
    }

    fclose(f);
    return 0;
}

/* Write a snap image to a .snap file.
 * Returns 0 on success, non-zero on error. */
static inline int trine_snap_file_write(const char *path,
                                         const trine_snap_t *arena,
                                         uint32_t count,
                                         uint32_t root,
                                         uint32_t free_head) {
    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "trine: cannot write '%s'\n", path);
        return 1;
    }

    /* Build header snap */
    trine_snap_t hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.hdr  = TRINE_PACK_HDR(TRINE_CELL_ID, TRINE_RANK_P, TRINE_DOM_CYC);
    hdr.gen  = TRINE_SNAP_VERSION;
    hdr.e[0] = count;
    hdr.e[1] = root;
    hdr.e[2] = free_head;
    hdr.back = TRINE_SNAP_MAGIC;
    hdr.data = trine_fnv1a(arena, (size_t)count * sizeof(trine_snap_t));

    /* Write header + arena */
    if (fwrite(&hdr, 32, 1, f) != 1 ||
        fwrite(arena, 32, count, f) != count) {
        fprintf(stderr, "trine: write error on '%s'\n", path);
        fclose(f);
        return 1;
    }

    fclose(f);
    return 0;
}

/* Free a snap image's arena. */
static inline void trine_snap_image_free(trine_snap_image_t *img) {
    if (img->arena) {
        free(img->arena);
        img->arena = NULL;
    }
    img->snap_count = 0;
    img->root_index = 0;
    img->free_head  = 0;
}

/* =====================================================================
 * X. SELF-VERIFICATION
 * =====================================================================
 *
 * Exhaustive test of all microcode ROM invariants. Returns true iff
 * every invariant holds. Run this at startup to confirm the ROM tables
 * are intact (especially important after cross-compilation or transport).
 */
static inline bool trine_verify(void) {
    /* 1. Snap size must be exactly 32 bytes */
    if (sizeof(trine_snap_t) != 32) return false;

    /* 2. Endomorphism closure: all outputs in {0,1,2} */
    for (int i = 0; i < 27; i++)
        for (int s = 0; s < 3; s++)
            if (TRINE_ENDO[i][s] > 2) return false;

    /* 3. Endomorphism rank consistency: image size matches ENDO_RANK */
    for (int i = 0; i < 27; i++) {
        uint8_t seen[3] = {0,0,0};
        for (int s = 0; s < 3; s++) seen[TRINE_ENDO[i][s]] = 1;
        uint8_t im = (uint8_t)(seen[0] + seen[1] + seen[2]);
        if (im != TRINE_ENDO_RANK[i]) return false;
    }

    /* 4. OIC routing monotonicity: output rank <= input rank */
    for (int r = 0; r < 3; r++)
        for (int d = 0; d < 2; d++)
            for (int c = 0; c < 7; c++)
                if (TRINE_OIC_ROUTE[r][d][c] > r) return false;

    /* 5. S3 Cayley table: every row is a permutation of {0..5} */
    for (int f = 0; f < 6; f++) {
        uint8_t seen[6] = {0,0,0,0,0,0};
        for (int g = 0; g < 6; g++) {
            uint8_t v = TRINE_ALU_S3[f][g];
            if (v > 5) return false;
            seen[v] = 1;
        }
        for (int k = 0; k < 6; k++)
            if (!seen[k]) return false;
    }

    /* 6. Canonical triples produce correct dynamics under cyclic tape.
     *    For each cell, compose g = f2(f1(f0(s))) and verify rank + spin. */
    for (int cell = 0; cell < 7; cell++) {
        uint16_t sq = TRINE_CANON_SQ[cell];
        uint8_t f0 = trine_snap_q(sq, 0);
        uint8_t f1 = trine_snap_q(sq, 1);
        uint8_t f2 = trine_snap_q(sq, 2);

        /* Compose: g(s) = f2(f1(f0(s))) */
        uint8_t g[3];
        for (int s = 0; s < 3; s++)
            g[s] = TRINE_ENDO[f2][TRINE_ENDO[f1][TRINE_ENDO[f0][s]]];

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

        /* Verify cell -> rank mapping */
        if (TRINE_CELL_RANK[cell] == TRINE_RANK_K && rank != 1) return false;
        if (TRINE_CELL_RANK[cell] == TRINE_RANK_S && rank != 2) return false;
        if (TRINE_CELL_RANK[cell] == TRINE_RANK_P && rank != 3) return false;

        /* Verify spin for cells that constrain it */
        if (cell == TRINE_CELL_ROT && spin != 3) return false;
        if (cell == TRINE_CELL_OSC && spin != 2) return false;
        if (cell == TRINE_CELL_SWP && spin != 2) return false;
        if (cell == TRINE_CELL_ID  && spin != 0) return false;
    }

    /* 7. snap_step output closure: output trit always in {0,1,2} */
    for (int cell = 0; cell < 7; cell++) {
        trine_snap_t s = trine_snap_make((uint8_t)cell, TRINE_DOM_CYC,
                                          TRINE_SNAP_NIL, TRINE_SNAP_NIL,
                                          TRINE_SNAP_NIL, 0);
        for (int trit = 0; trit < 3; trit++) {
            for (int st = 0; st < 3; st++) {
                s.sr = TRINE_PACK_SR((uint8_t)st, 0, TRINE_STAT_LIVE);
                trine_result_t r = trine_snap_step(&s, (uint8_t)trit);
                if (r.out > 2) return false;
                if (trine_snap_st(&s) > 2) return false;
            }
        }
    }

    /* 8. ROTATE cycles: 3 consecutive steps produce outputs 0, 1, 2.
     *    State advances 0->1->2->0 via CW rotation quark e7={1,2,0}. */
    {
        trine_snap_t rot = trine_snap_make(TRINE_CELL_ROT, TRINE_DOM_CYC,
                                            TRINE_SNAP_NIL, TRINE_SNAP_NIL,
                                            TRINE_SNAP_NIL, 0);
        rot.sr = TRINE_PACK_SR(0, 0, TRINE_STAT_LIVE);

        trine_result_t r0 = trine_snap_step(&rot, 0);  /* out=0, 0->1 */
        trine_result_t r1 = trine_snap_step(&rot, 0);  /* out=1, 1->2 */
        trine_result_t r2 = trine_snap_step(&rot, 0);  /* out=2, 2->0 */

        if (r0.out != 0 || r1.out != 1 || r2.out != 2) return false;
        if (trine_snap_st(&rot) != 0) return false;  /* returned to start */
    }

    /* 9. Rank degradation: RANK_P -> apply NULL -> RANK_K */
    {
        trine_snap_t dst = trine_snap_make(TRINE_CELL_ID, TRINE_DOM_CYC,
                                            TRINE_SNAP_NIL, TRINE_SNAP_NIL,
                                            TRINE_SNAP_NIL, 0);
        trine_snap_degrade(&dst, TRINE_CELL_NULL);
        if (trine_snap_rank(&dst) != TRINE_RANK_K) return false;
    }

    /* 10. Rank degradation: RANK_P -> apply ID -> RANK_P (preserved) */
    {
        trine_snap_t dst = trine_snap_make(TRINE_CELL_ID, TRINE_DOM_CYC,
                                            TRINE_SNAP_NIL, TRINE_SNAP_NIL,
                                            TRINE_SNAP_NIL, 0);
        trine_snap_degrade(&dst, TRINE_CELL_ID);
        if (trine_snap_rank(&dst) != TRINE_RANK_P) return false;
    }

    /* All invariants hold. ROM verified. */
    return true;
}

/* =====================================================================
 * XI. NAME TABLES
 * =====================================================================
 *
 * Human-readable names for cells, domains, ranks, and statuses.
 * Useful for diagnostics, debugging, and visualization.
 */

static const char * const TRINE_CELL_NAME[7] = {
    "NULL", "CLPS", "SPLT", "OSC", "ID", "SWP", "ROT"
};

static const char * const TRINE_DOM_NAME[6] = {
    "CYC", "CON", "ALT", "REV", "BIN", "IO"
};

static const char * const TRINE_RANK_NAME[3] = { "K", "S", "P" };

static const char * const TRINE_STAT_NAME[4] = { "FREE", "IDLE", "LIVE", "LOCK" };

/* =====================================================================
 * XII. COMPILE-TIME ASSERTIONS
 * ===================================================================== */

_Static_assert(sizeof(trine_snap_t) == 32, "Snap must be exactly 32 bytes");

#endif /* TRINE_ALGEBRA_H */
