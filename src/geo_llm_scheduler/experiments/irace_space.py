"""Finite irace encoding of the validated v4 component catalogue."""

from dataclasses import asdict

from geo_llm_scheduler.experiments.component_search_v4 import NUMERIC, SLOTS, Components


def parameter_text(mode: str = "joint") -> str:
    """Export categorical grids without allowing interpolation outside declared values."""
    if mode not in ("joint", "numeric"):
        raise ValueError("Expected joint or numeric search")
    domains = {**SLOTS, **NUMERIC} if mode == "joint" else NUMERIC
    lines = []
    for name, values in domains.items():
        encoded = [f"w{i}" for i in range(len(values))] if name == "mutation_weights" else values
        lines.append(f'{name} "--{name} " c ({", ".join(map(str, encoded))})')
    return "\n".join(lines) + "\n"


def decode_configuration(values: dict) -> Components:
    """Decode irace's string-valued configuration through the strict v4 validator."""
    fields = dict(values)
    unknown = set(fields) - set(SLOTS) - set(NUMERIC)
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    for name in set(fields) & set(NUMERIC):
        value = str(fields[name])
        if name == "mutation_weights":
            choices = {f"w{i}": w for i, w in enumerate(NUMERIC[name])}
            if value not in choices:
                raise ValueError("Unknown weight group")
            fields[name] = choices[value]
        elif name in ("neighborhood", "replacement_cap", "ls_width", "ls_depth"):
            fields[name] = int(value)
        else:
            fields[name] = float(value)
    return Components(**fields)


def default_table(mode: str = "joint") -> str:
    """Provide the adopted default as an initial configuration for either search arm."""
    parameter_text(mode)
    names = list({**SLOTS, **NUMERIC} if mode == "joint" else NUMERIC)
    defaults = asdict(Components())
    defaults["mutation_weights"] = "w0"
    return " ".join(names) + "\n" + " ".join(str(defaults[n]) for n in names) + "\n"
