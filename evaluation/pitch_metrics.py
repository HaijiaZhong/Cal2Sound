"""Pitch metric placeholders; definitions must be confirmed from the paper code."""


def compute_pda(reference_f0, reconstructed_f0, *, tolerance_cents=None):
    """Compute PDA for reference and reconstructed F0 trajectories (TBD).

    PDA definition, F0 units, alignment, voicing masks, aggregation, and
    tolerance must be confirmed during migration. No default is assumed.
    """
    raise NotImplementedError("PDA implementation is pending migration.")


def compute_median_f0_error(reference_f0, reconstructed_f0):
    """Compute median F0 error between two trajectories (TBD).

    Error units, signed versus absolute error, frame alignment, and handling
    of unvoiced or missing frames remain to be confirmed.
    """
    raise NotImplementedError("Median F0 error implementation is pending migration.")


def compute_note_accuracy(reference_f0, reconstructed_f0):
    """Compute note accuracy from reference and reconstructed F0 (TBD).

    Note mapping, tuning reference, frame versus event scoring, alignment,
    and voicing conventions remain to be confirmed.
    """
    raise NotImplementedError("Note accuracy implementation is pending migration.")
