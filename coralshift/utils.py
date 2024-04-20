def lat_lon_string_from_tuples(
    lats: tuple[float, float], lons: tuple[float, float], dp: int = 0
):
    round_lats = iterative_to_string_list(lats, dp)
    round_lons = iterative_to_string_list(lons, dp)

    return (
        f"n{max(round_lats)}_s{min(round_lats)}_w{min(round_lons)}_e{max(round_lons)}"
    )


def gen_seafloor_indices(xa_da: xa.Dataset, var: str, dim: str = "lev"):
    """Generate indices of seafloor values for a given variable in an xarray dataset.

    Args:
        xa_da (xa.Dataset): xarray dataset containing variable of interest
        var (str): name of variable of interest
        dim (str, optional): dimension along which to search for seafloor values. Defaults to "lev".

    Returns:
        indices_array (np.ndarray): array of indices of seafloor values for given variable
    """
    nans = np.isnan(xa_da[var]).sum(dim=dim)  # separate out
    indices_array = -(nans.values) - 1
    indices_array[indices_array == -(len(xa_da[dim].values) + 1)] = -1
    return indices_array


def extract_seafloor_vals(xa_da, indices_array):
    vals_array = xa_da.values
    t, j, i = indices_array.shape
    # create open grid for indices along each dimension
    t_grid, j_grid, i_grid = np.ogrid[:t, :j, :i]
    # select values from vals_array using indices_array
    return vals_array[t_grid, indices_array, j_grid, i_grid]


def generate_remapping_file(
    eg_xa: xa.Dataset | xa.DataArray,
    remap_template_fp: str | Path,
    resolution: float = 0.25,
    out_grid: str = "lonlat",
):
    xsize, ysize, xfirst, yfirst, x_inc, y_inc = generate_remap_info(
        eg_xa, resolution, out_grid
    )

    print(f"Saving regridding info to {remap_template_fp}...")
    with open(remap_template_fp, "w") as file:
        file.write(
            f"gridtype = {out_grid}\n"
            f"xsize = {xsize}\n"
            f"ysize = {ysize}\n"
            f"xfirst = {xfirst}\n"
            f"yfirst = {yfirst}\n"
            f"xinc = {x_inc}\n"
            f"yinc = {y_inc}\n"
        )


class FileName:
    def __init__(
        self,
        # source_id: str,
        # member_id: str,
        variable_id: str | list,
        grid_type: str,
        fname_type: str,
        date_range: str = None,
        lats: list[float, float] = None,
        lons: list[float, float] = None,
        levs: list[int, int] = None,
        plevels: list[float, float] = None,
    ):
        """
        Args:
            source_id (str): name of climate model
            member_id (str): model run
            variable_id (str): variable name
            grid_type (str): type of grid (tripolar or latlon)
            fname_type (str): type of file (individual, time-concatted, var_concatted) TODO
            lats (list[float, float], optional): latitude range. Defaults to None.
            lons (list[float, float], optional): longitude range. Defaults to None.
            plevels (list[float, float], optional): pressure level range. Defaults to None.
            fname (str, optional): filename. Defaults to None to allow construction.
        """
        self.variable_id = variable_id
        self.grid_type = grid_type
        self.fname_type = fname_type
        self.date_range = date_range
        self.lats = lats
        self.lons = lons
        self.levs = levs
        self.plevels = plevels

    def get_spatial(self):
        if self.lats and self.lons:  # if spatial range specified (i.e. cropping)
            # cast self.lats and self.lons lists to integers. A little crude, but avoids decimals in filenames
            lats = [int(lat) for lat in self.lats]
            lons = [int(lon) for lon in self.lons]
            return lat_lon_string_from_tuples(lats, lons).upper()
        else:
            return "uncropped"

    def get_plevels(self):
        if self.plevels == [-1] or self.plevels == -1:  # seafloor
            return f"sfl-{max(self.levs)}"
        elif not self.plevels:
            return "sfc"
            if self.plevels[0] is None:
                return "sfc"
        elif isinstance(self.plevels, float):  # specified pressure level
            return "{:.0f}".format(self.plevels / 100)
        elif isinstance(self.plevels, list):  # pressure level range
            if self.plevels[0] is None:  # if plevels is list of None, surface
                return "sfc"
            return f"levs_{min(self.plevels)}-{max(self.plevels)}"
        else:
            raise ValueError(
                f"plevel must be one of [-1, float, list]. Instead received '{self.plevels}'"
            )

    def get_var_str(self):
        if isinstance(self.variable_id, list):
            return "_".join(self.variable_id)
        else:
            return self.variable_id

    def get_grid_type(self):
        if self.grid_type == "tripolar":
            return "tp"
        elif self.grid_type == "latlon":
            return "ll"
        else:
            raise ValueError(
                f"grid_type must be 'tripolar' or 'latlon'. Instead received '{self.grid_type}'"
            )

    def get_date_range(self):
        if not self.date_range:
            return None
        if self.fname_type == "time_concatted" or self.fname_type == "var_concatted":
            # Can't figure out how it's being done currently
            return str("-".join((str(self.date_range[0]), str(self.date_range[1]))))
        else:
            return self.date_range

    def join_as_necessary(self):
        var_str = self.get_var_str()
        spatial = self.get_spatial()
        plevels = self.get_plevels()
        grid_type = self.get_grid_type()
        date_range = self.get_date_range()

        # join these variables separated by '_', so long as the variable is not None
        return "_".join(
            [i for i in [var_str, spatial, plevels, grid_type, date_range] if i]
        )

    def construct_fname(self):

        if self.fname_type == "var_concatted":
            if not isinstance(self.variable_id, list):
                raise TypeError(
                    f"Concatted variable requires multiple variable_ids. Instead received '{self.variable_id}'"
                )

        self.fname = self.join_as_necessary()

        return f"{self.fname}.nc"
