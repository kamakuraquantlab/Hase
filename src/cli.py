"""Run the CLI from a source checkout: `python src/cli.py ...`.

The real module is `hase.cli`, which is where the installed `hase` command
points. This file holds no logic: imported, it *becomes* that module, so
`import cli` and `import hase.cli` are the same object and patching an
attribute through either name patches both.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hase import cli as _real  # noqa: E402

if __name__ == "__main__":
    sys.exit(_real.main())

sys.modules[__name__] = _real
