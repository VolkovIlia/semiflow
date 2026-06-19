/*
 * greeks.c — C smoke test for the v8.0.0 Greeks FFI surface.
 *
 * Tests smf_greeks_evolver_new_heat_1d_unit_v3 / smf_heat1d_greeks_v3 /
 * smf_greeks_evolver_free_v3 (ADR-0028 Amendment 2, ADR-0133 A1).
 *
 * Verification strategy (design doc §5, parity gate sub-test 1):
 *   - Canonical smoke: theta=0.5, N=64, n=32, t=0.05, u0=exp(-x^2).
 *   - delta_fd: central difference of value w.r.t. theta (h=1e-5).
 *   - gamma_fd: second finite difference of value w.r.t. theta.
 *   - Acceptance: ||delta - delta_fd||_inf <= 1e-6
 *                 ||gamma - gamma_fd||_inf <= 1e-4  (FD is noisy side)
 *   - All value/delta/gamma entries must be finite.
 *
 * Exit 0 on success, exit 1 on failure.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/semiflow.h"

#define N         64
#define XMIN     -5.0
#define XMAX      5.0
#define T         0.05
#define N_CHERN   32
#define THETA     0.5
#define H_FD      1e-5

static int check_finite(const double *buf, int n, const char *name) {
    for (int i = 0; i < n; i++) {
        if (!isfinite(buf[i])) {
            fprintf(stderr, "FAIL: %s[%d] = %g (not finite)\n", name, i, buf[i]);
            return 0;
        }
    }
    return 1;
}

static double sup_norm_diff(const double *a, const double *b, int n) {
    double sup = 0.0;
    for (int i = 0; i < n; i++) {
        double d = fabs(a[i] - b[i]);
        if (d > sup) sup = d;
    }
    return sup;
}

/* Build u0[i] = exp(-x_i^2) for given N, xmin, xmax. */
static void make_u0(double *u0, int n, double xmin, double xmax) {
    double dx = (xmax - xmin) / (double)(n - 1);
    for (int i = 0; i < n; i++) {
        double x = xmin + i * dx;
        u0[i] = exp(-x * x);
    }
}

/* Evolve with theta and write value into out_val (length n). */
static int evolve_value(double theta, double *u0, int n,
                        double xmin, double xmax, int n_chern,
                        double t, double *out_val) {
    double *dummy_d = malloc(n * sizeof(double));
    double *dummy_g = malloc(n * sizeof(double));
    SmfGreeksEvolverV3 *ev = NULL;
    SemiflowStatus st = smf_greeks_evolver_new_heat_1d_unit_v3(
        xmin, xmax, (size_t)n, (size_t)n_chern, theta,
        u0, (size_t)n, &ev);
    if (st != Ok) {
        fprintf(stderr, "new_greeks_evolver failed (theta=%g): %d\n", theta, (int)st);
        free(dummy_d); free(dummy_g);
        return 0;
    }
    st = smf_heat1d_greeks_v3(ev, t, out_val, dummy_d, dummy_g, (size_t)n);
    smf_greeks_evolver_free_v3(ev);
    free(dummy_d); free(dummy_g);
    if (st != Ok) {
        fprintf(stderr, "greeks_v3 failed (theta=%g): %d\n", theta, (int)st);
        return 0;
    }
    return 1;
}

int main(void) {
    double *u0     = malloc(N * sizeof(double));
    double *value  = malloc(N * sizeof(double));
    double *delta  = malloc(N * sizeof(double));
    double *gamma  = malloc(N * sizeof(double));
    double *val_p  = malloc(N * sizeof(double));  /* theta + h */
    double *val_m  = malloc(N * sizeof(double));  /* theta - h */
    double *delta_fd = malloc(N * sizeof(double));
    double *gamma_fd = malloc(N * sizeof(double));

    if (!u0 || !value || !delta || !gamma || !val_p || !val_m ||
        !delta_fd || !gamma_fd) {
        fputs("malloc failed\n", stderr);
        return 1;
    }

    make_u0(u0, N, XMIN, XMAX);

    /* --- Main Greeks sweep --- */
    SmfGreeksEvolverV3 *ev = NULL;
    SemiflowStatus st = smf_greeks_evolver_new_heat_1d_unit_v3(
        XMIN, XMAX, (size_t)N, (size_t)N_CHERN, THETA,
        u0, (size_t)N, &ev);
    if (st != Ok) {
        fprintf(stderr, "new_greeks_evolver failed: %d\n", (int)st);
        return 1;
    }
    st = smf_heat1d_greeks_v3(ev, T, value, delta, gamma, (size_t)N);
    smf_greeks_evolver_free_v3(ev);
    if (st != Ok) {
        fprintf(stderr, "greeks_v3 failed: %d\n", (int)st);
        return 1;
    }

    /* --- Finiteness checks --- */
    if (!check_finite(value, N, "value") ||
        !check_finite(delta, N, "delta") ||
        !check_finite(gamma, N, "gamma")) {
        return 1;
    }

    /* --- Finite-difference references --- */
    if (!evolve_value(THETA + H_FD, u0, N, XMIN, XMAX, N_CHERN, T, val_p)) return 1;
    if (!evolve_value(THETA - H_FD, u0, N, XMIN, XMAX, N_CHERN, T, val_m)) return 1;

    for (int i = 0; i < N; i++) {
        delta_fd[i] = (val_p[i] - val_m[i]) / (2.0 * H_FD);
        gamma_fd[i] = (val_p[i] - 2.0 * value[i] + val_m[i]) / (H_FD * H_FD);
    }

    double sup_delta = sup_norm_diff(delta, delta_fd, N);
    double sup_gamma = sup_norm_diff(gamma, gamma_fd, N);

    printf("sup_greeks_delta=%.6e\n", sup_delta);
    printf("sup_greeks_gamma=%.6e\n", sup_gamma);

    int ok = 1;
    if (sup_delta >= 1e-6) {
        fprintf(stderr, "FAIL: sup(delta - delta_fd) = %.3e >= 1e-6\n", sup_delta);
        ok = 0;
    }
    if (sup_gamma >= 1e-4) {
        fprintf(stderr, "FAIL: sup(gamma - gamma_fd) = %.3e >= 1e-4\n", sup_gamma);
        ok = 0;
    }

    free(u0); free(value); free(delta); free(gamma);
    free(val_p); free(val_m); free(delta_fd); free(gamma_fd);

    if (ok) { puts("PASS"); return 0; }
    return 1;
}
