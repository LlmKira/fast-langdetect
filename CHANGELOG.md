# Changelog

All notable changes to this project will be documented in this file.

## [0.4.0] - 2025-09-15

- Behavior: Always replace newline characters in input to prevent FastText errors. This adjustment is logged at DEBUG level only.
- Default input truncation: Truncate inputs to 80 characters by default for stable predictions. Configurable via `LangDetectConfig(max_input_length=...)`; set `None` to disable.
- Simplified config: Removed previously proposed `verbose` and `replace_newlines` options; newline replacement is unconditional and logging of adjustments is controlled by global logger level.
- Logging: Deprecated-parameter messages lowered from WARNING to INFO to reduce noise.
- Documentation: README now includes language code → name mapping guidance and an explicit model license note (CC BY-SA 3.0) alongside MIT for code.
