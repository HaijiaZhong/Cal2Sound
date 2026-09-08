"""Future common wrapper for CREPE, SwiftF0, PESTO, pYIN, and RMVPE.

No backend is imported, installed, or downloaded by this module.
"""


def estimate_f0(audio, sample_rate, *, backend, **backend_options):
    """Estimate an F0 trajectory using an explicitly selected backend (TBD).

    Audio shape, sample-rate requirements, frame times, F0 units, confidence,
    voicing representation, backend names, and options will be standardized
    during migration. No estimator or preprocessing defaults are assumed.
    """
    raise NotImplementedError("F0 estimator integration is pending migration.")
