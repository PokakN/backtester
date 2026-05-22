import importlib.util
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

STRATEGIES_DIR = Path(__file__).parent

_REQUIRED_INFO_KEYS = {"name", "description", "parameters", "parameter_ranges", "warmup_bars", "tags"}


def _validate_parameter_ranges(parameter_ranges: dict) -> str | None:
    """Return error string if invalid, None if valid."""
    for key, spec in parameter_ranges.items():
        if not isinstance(spec, dict) or not {"min", "max", "step"}.issubset(spec.keys()):
            return f"'{key}' missing min/max/step"
        if not all(isinstance(spec[k], (int, float)) for k in ("min", "max", "step")):
            return f"'{key}' min/max/step must be numeric"
        if spec["min"] > spec["max"]:
            return f"'{key}' min > max"
        if spec["step"] <= 0:
            return f"'{key}' step must be > 0"
    return None


def discover_strategies(strategies_dir: Path) -> dict:
    """
    Load all strategy modules from strategies_dir.

    Each valid module must expose:
      - STRATEGY_INFO: dict with keys name, description, parameters,
        parameter_ranges (each entry has min/max/step), warmup_bars (positive int), tags (list)
      - generate_signals: callable

    Returns:
        {strategy_name: {"info": STRATEGY_INFO, "fn": generate_signals, "module": module}}
    """
    strategies = {}

    for path in sorted(strategies_dir.glob("*.py")):
        if path.name == "__init__.py":
            continue

        module_name = f"strategies.{path.stem}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as exc:
            logger.error("Failed to load strategy file %s: %s", path.name, exc)
            continue

        # Validate STRATEGY_INFO presence and type
        info = getattr(module, "STRATEGY_INFO", None)
        if not isinstance(info, dict):
            logger.error(
                "Skipping %s: STRATEGY_INFO is missing or not a dict", path.name
            )
            continue

        # Validate required keys
        missing_keys = _REQUIRED_INFO_KEYS - info.keys()
        if missing_keys:
            logger.error(
                "Skipping %s: STRATEGY_INFO missing keys %s", path.name, missing_keys
            )
            continue

        # Validate name
        if not isinstance(info.get("name"), str):
            logger.error("Skipping %s: STRATEGY_INFO['name'] must be a str", path.name)
            continue

        # Validate description
        if not isinstance(info.get("description"), str):
            logger.error(
                "Skipping %s: STRATEGY_INFO['description'] must be a str", path.name
            )
            continue

        # Validate parameters
        if not isinstance(info.get("parameters"), dict):
            logger.error(
                "Skipping %s: STRATEGY_INFO['parameters'] must be a dict", path.name
            )
            continue

        # Validate parameter_ranges
        if not isinstance(info.get("parameter_ranges"), dict):
            logger.error(
                "Skipping %s: STRATEGY_INFO['parameter_ranges'] must be a dict", path.name
            )
            continue
        range_err = _validate_parameter_ranges(info["parameter_ranges"])
        if range_err:
            logger.error(
                "Skipping %s: STRATEGY_INFO['parameter_ranges'] invalid: %s",
                path.name, range_err,
            )
            continue

        # Validate that every parameter key has a corresponding range entry
        missing_ranges = set(info["parameters"].keys()) - set(info["parameter_ranges"].keys())
        if missing_ranges:
            logger.error(
                "Skipping %s: STRATEGY_INFO['parameter_ranges'] missing entries for: %s",
                path.name, missing_ranges,
            )
            continue

        # Validate warmup_bars
        warmup = info.get("warmup_bars")
        if not isinstance(warmup, int) or isinstance(warmup, bool) or warmup <= 0:
            logger.error(
                "Skipping %s: STRATEGY_INFO['warmup_bars'] must be a positive int", path.name
            )
            continue

        # Validate tags
        if not isinstance(info.get("tags"), list):
            logger.error(
                "Skipping %s: STRATEGY_INFO['tags'] must be a list", path.name
            )
            continue

        # Validate generate_signals
        fn = getattr(module, "generate_signals", None)
        if not callable(fn):
            logger.error(
                "Skipping %s: generate_signals is missing or not callable", path.name
            )
            continue

        strategy_name = info["name"]
        if strategy_name in strategies:
            logger.error(
                "Skipping %s: duplicate strategy name '%s' already registered",
                path.name,
                strategy_name,
            )
            continue

        strategies[strategy_name] = {"info": info, "fn": fn, "module": module}
        logger.debug("Loaded strategy '%s' from %s", strategy_name, path.name)

    return strategies


STRATEGIES = discover_strategies(STRATEGIES_DIR)
