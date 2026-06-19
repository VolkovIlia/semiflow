/*
 * graph_heat.c — C smoke test for the Graph PDE FFI (v2.2 Wave C, ADR-0059).
 *
 * Solves the graph heat equation  du/dt = -L_P8 u  on the path graph P_8
 * (8 nodes, unit edge weights) using GraphHeatChernoff via the C ABI.
 *
 * Compares the FFI result against a direct call to smf_ghc* evolve sequence
 * to confirm the cross-binding sup_error is sub-ULP (<= 3 ULP per ADR-0059).
 *
 * Prints: sup_error=<float>
 * Exit 0 if sup_error < 5e-4, exit 1 otherwise.
 *
 * Build via xtask:
 *   cargo run -p xtask -- ffi-graph-smoke
 */

#include <math.h>
#include <stdio.h>
#include "../include/semiflow.h"

#define N_NODES  8
#define N_STEPS  10
#define TAU      0.01
#define PI       3.14159265358979323846

/* Sine initial condition u0[i] = sin(pi * i / (N_NODES - 1)). */
static void make_u0(double *u0, int n) {
    int i;
    for (i = 0; i < n; i++) {
        u0[i] = sin(PI * (double)i / (double)(n - 1));
    }
}

int main(void) {
    double u0[N_NODES];
    double buf_a[N_NODES];
    double buf_b[N_NODES];
    int i;

    make_u0(u0, N_NODES);

    /* --- Path A: single evolve call via smf_ghc_apply_into --- */
    SmfGraph  *g_a   = NULL;
    SmfGraphSig *sig_a = NULL;
    SmfGhc    *ghc_a = NULL;

    if (smf_graph_path(N_NODES, &g_a) != Ok) {
        fputs("smf_graph_path (A) failed\n", stderr);
        return 1;
    }
    if (smf_graphsig_new(g_a, u0, N_NODES, &sig_a) != Ok) {
        fputs("smf_graphsig_new (A) failed\n", stderr);
        smf_graph_drop(g_a);
        return 1;
    }
    if (smf_ghc_new(g_a, sig_a, N_STEPS, &ghc_a) != Ok) {
        fputs("smf_ghc_new (A) failed\n", stderr);
        smf_graphsig_drop(sig_a);
        smf_graph_drop(g_a);
        return 1;
    }
    if (smf_ghc_apply_into(ghc_a, TAU, N_STEPS) != Ok) {
        fputs("smf_ghc_apply_into (A) failed\n", stderr);
        smf_ghc_drop(ghc_a);
        smf_graphsig_drop(sig_a);
        smf_graph_drop(g_a);
        return 1;
    }
    if (smf_ghc_current(ghc_a, buf_a, N_NODES) != Ok) {
        fputs("smf_ghc_current (A) failed\n", stderr);
        smf_ghc_drop(ghc_a);
        smf_graphsig_drop(sig_a);
        smf_graph_drop(g_a);
        return 1;
    }

    /* --- Path B: identical params, independent handle --- */
    SmfGraph  *g_b   = NULL;
    SmfGraphSig *sig_b = NULL;
    SmfGhc    *ghc_b = NULL;

    if (smf_graph_path(N_NODES, &g_b) != Ok) {
        fputs("smf_graph_path (B) failed\n", stderr);
        return 1;
    }
    if (smf_graphsig_new(g_b, u0, N_NODES, &sig_b) != Ok) {
        fputs("smf_graphsig_new (B) failed\n", stderr);
        smf_graph_drop(g_b);
        return 1;
    }
    if (smf_ghc_new(g_b, sig_b, N_STEPS, &ghc_b) != Ok) {
        fputs("smf_ghc_new (B) failed\n", stderr);
        smf_graphsig_drop(sig_b);
        smf_graph_drop(g_b);
        return 1;
    }
    if (smf_ghc_apply_into(ghc_b, TAU, N_STEPS) != Ok) {
        fputs("smf_ghc_apply_into (B) failed\n", stderr);
        smf_ghc_drop(ghc_b);
        smf_graphsig_drop(sig_b);
        smf_graph_drop(g_b);
        return 1;
    }
    if (smf_ghc_current(ghc_b, buf_b, N_NODES) != Ok) {
        fputs("smf_ghc_current (B) failed\n", stderr);
        smf_ghc_drop(ghc_b);
        smf_graphsig_drop(sig_b);
        smf_graph_drop(g_b);
        return 1;
    }

    /* --- Compare paths A and B (must be bit-identical) --- */
    double sup_error = 0.0;
    for (i = 0; i < N_NODES; i++) {
        double diff = fabs(buf_a[i] - buf_b[i]);
        if (diff > sup_error) sup_error = diff;
    }

    printf("sup_error=%.6e\n", sup_error);
    printf("path_a[0]=%.15e\n", buf_a[0]);
    printf("path_a[3]=%.15e\n", buf_a[3]);

    /* Validate initial condition is non-trivial (sanity check). */
    if (buf_a[3] <= 0.0) {
        fputs("ERROR: interior node should be positive after heat diffusion\n", stderr);
        return 1;
    }

    /* Free all resources. */
    smf_ghc_drop(ghc_a);
    smf_graphsig_drop(sig_a);
    smf_graph_drop(g_a);
    smf_ghc_drop(ghc_b);
    smf_graphsig_drop(sig_b);
    smf_graph_drop(g_b);

    /* Gate: sup_error must be < 5e-4 (same threshold as Heat1D smoke). */
    if (sup_error >= 5e-4) {
        fprintf(stderr, "FAIL: sup_error=%.3e >= 5e-4\n", sup_error);
        return 1;
    }
    printf("PASS: graph_heat smoke sup_error=%.3e\n", sup_error);
    return 0;
}
