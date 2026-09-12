"""
feedback_cli.py

Everything the participant/audience actually SEES, printed to the
terminal. Kept separate from the computation logic (readiness.py,
state_machine.py, etc.) so the wording can be changed without touching
any of the underlying math.

LANGUAGE RULES FOLLOWED HERE (Section 24):
    - Never "EEG knows/predicts chess outcome."
    - Never state a feature "means" a specific mental state as fact.
    - Contributing features are described as inputs to the estimate,
      not as proof of a cause.
"""

STATE_EMOJI = {"READY": "\U0001F7E2", "HIGH_LOAD": "\U0001F7E1", "PAUSE": "\U0001F534"}


def readiness_bar(score, width=20):
    """Simple text progress bar, e.g. [##############------] 72.3"""
    filled = max(0, min(width, int(round(score / 100 * width))))
    return f"[{'#' * filled}{'-' * (width - filled)}] {score:5.1f}"


def format_calibration_start(duration_sec):
    return (f"Calibrating... please remain still and relaxed for "
            f"{duration_sec:.0f} seconds.")


def format_baseline_established(current_time, baseline_summary):
    return (f"\n[t={current_time:6.1f}s] BASELINE ESTABLISHED\n"
            f"{baseline_summary}\n")


def format_status_line(current_time, state, smoothed_score, quality):
    """One compact line printed on every update, whether or not the state changed."""
    emoji = STATE_EMOJI.get(state, "?")
    bar = readiness_bar(smoothed_score)
    return (f"[t={current_time:6.1f}s] {emoji} {state:9s} "
            f"readiness {bar}  quality={quality:.2f}")


def format_signal_uncertain(current_time, quality):
    return (f"[t={current_time:6.1f}s] \u26AA EEG SIGNAL UNCERTAIN "
            f"(usable channels: {quality:.0%})\n"
            f"             Check headset/electrode contact. "
            f"Readiness estimate frozen at last known value.")


def _contributor_lines(top_contributors):
    if not top_contributors:
        return ["    (no baseline deviation data available for this window)"]
    lines = []
    for name, z in top_contributors:
        direction = "above" if z > 0 else "below"
        lines.append(f"    - {name}: {abs(z):.2f} SD {direction} your personal baseline")
    return lines


def format_transition_block(current_time, transition, readiness_result, smoothed_score):
    """
    The highlighted message printed whenever the state actually
    changes - the terminal equivalent of Section 15's feedback states
    and Section 22's explainability panel.
    """
    state = transition["state"]
    lines = ["", "=" * 64]

    if transition["recovered_from_pause"]:
        lines.append(f"[t={current_time:6.1f}s] \U0001F7E2 STATE STABILIZED")
        lines.append("  Ready to continue.")
    elif state == "READY":
        lines.append(f"[t={current_time:6.1f}s] \U0001F7E2 READY")
        lines.append("  Cognitive state appears stable.")
        lines.append("  Continue evaluating the position.")
    elif state == "HIGH_LOAD":
        lines.append(f"[t={current_time:6.1f}s] \U0001F7E1 HIGH COGNITIVE LOAD")
        lines.append("  Take a moment.")
        lines.append("  Re-evaluate the position.")
    elif state == "PAUSE":
        lines.append(f"[t={current_time:6.1f}s] \U0001F534 PAUSE")
        lines.append("  Your current EEG-derived state is unstable relative to your baseline.")
        lines.append("  Take a short pause and reassess the position.")

    lines.append(f"  Readiness: {smoothed_score:.1f}/100  "
                 f"(transitioned from {transition['previous_state']})")
    lines.append("  Contributing EEG features (model inputs, not proof of cause):")
    lines.extend(_contributor_lines(readiness_result["top_contributors"]))
    lines.append("=" * 64)
    return "\n".join(lines)


def format_final_summary(current_time, final_state):
    return (f"\nReplay finished at t={current_time:.1f}s. "
            f"Final state: {final_state}")