from pathlib import Path

def guarantee_existence(path: str) -> Path:
    """Checks if string is an existing path, else creates it
	
	Parameter
	---------
	path : str
	
	Returns
	-------
	Path
		pathlib.Path object of path
	"""
	path_obj = Path(path)
    if not path_obj.exists():
        path_obj.mkdir(parents=True)
    return path_obj.resolve()