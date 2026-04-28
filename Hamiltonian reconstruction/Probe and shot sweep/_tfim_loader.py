"""Load the TFIM n=10 reconstruction module despite the space in the parent
folder name (``Hamiltonian reconstruction``)."""
from importlib import util as _importlib_util
from pathlib import Path as _Path

_HERE = _Path(__file__).resolve().parent
_TFIM_PY = _HERE.parent / "TFIM" / "hamiltonian_reconstruction_tfim_n10.py"

_spec = _importlib_util.spec_from_file_location("tfim_n10", _TFIM_PY)
tfim = _importlib_util.module_from_spec(_spec)
_spec.loader.exec_module(tfim)
