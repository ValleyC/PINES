# Research contract

## Claim

For a frozen finite-horizon classifier, reference implementation, declared
family of target execution semantics, and deployment distribution, the project
seeks an upper confidence bound on absolute accuracy change before full
deployment. The observable is paired prediction disagreement. The method joins
an execution-semantics family, static per-input analysis, physical conformance
composition, and certificate-directed repair. Raw disagreement alone is not the
methodological contribution.

For any label \(y\), classifiers \(f\) and \(g\), and input \(x\),

\[
  |\mathbf{1}[f(x)=y]-\mathbf{1}[g(x)=y]| \leq \mathbf{1}[f(x)\ne g(x)].
\]

Thus a one-sided confidence bound on disagreement bounds absolute accuracy
change without observing audit labels. For emulator \(e\) and hardware \(h\),
the triangle inequality gives

\[
U_{\mathrm{total}}=\min(1,
U_{\mathrm{ref,emu}}(\delta/2)+U_{\mathrm{emu,hw}}(\delta/2)).
\]

The physical estimate uses one independent `(input, hardware run)` observation
per sample. Repeats characterize nondeterminism but are secondary.

## Certificate hierarchy

1. Structural identity or a proved closed-form mapping.
2. Per-input family certification using interval/shared-affine enclosures,
   polygonal guard refinement and local threshold branches, with exhaustive
   finite-state checking on tiny systems. Inputs unresolved at the proof budget
   count as failures, not preservation certificates.
3. Exact Clopper-Pearson disagreement limits on an untouched unlabeled audit
   split, corrected simultaneously across target family members.
4. The physical conformance term from a manifest-bound canary capture.

Exact equivalence is never inferred from a successful pilot. In particular,
pre-integration and post-integration threshold equations have a checked-in
counterexample test and are not claimed equivalent.

## Scope

Included: digital SNN classification over a fixed horizon, forward and
exponential Euler integration, threshold and update order, reset, fixed-point
precision, rounding, overflow and saturation, explicit delays, and deterministic
or stochastic parameters.

Excluded: analog mismatch, continual learning, safety-critical control, and
energy-efficiency claims.
