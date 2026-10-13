"""
Make pyneogp use a local checkout of NeoGP.jl instead of the package from
GitHub, for developing NeoGP.jl and the bindings together.

    python use_local_neogp.py path/to/NeoGP.jl

rewrites ``pyneogp/juliapkg.json`` to develop NeoGP from the checkout; juliacall
installs it on the next import (with an editable install, ``pip install -e .``).
Revert the file with ``git checkout pyneogp/juliapkg.json`` before committing.
"""

import json
import pathlib
import sys

if len(sys.argv) != 2:
    sys.exit(__doc__)
checkout = pathlib.Path(sys.argv[1]).expanduser().resolve()
if not (checkout / "src" / "NeoGP.jl").is_file():
    sys.exit(f"{checkout} is not a checkout of NeoGP.jl")

path = pathlib.Path(__file__).parent / "pyneogp" / "juliapkg.json"
spec = json.loads(path.read_text())
neogp = spec["packages"]["NeoGP"]
spec["packages"]["NeoGP"] = {"uuid": neogp["uuid"], "path": str(checkout), "dev": True}
path.write_text(json.dumps(spec, indent=4) + "\n")
print(f"{path}: NeoGP is developed from {checkout}")
