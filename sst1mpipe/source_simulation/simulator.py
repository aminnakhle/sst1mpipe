# Main file for the source simulator
from astropy.coordinates import SkyCoord
from astropy import units as u
from gammapy.modeling.models import (
    PowerLawSpectralModel,
    PointSpatialModel,
    SkyModel,
)

class SourceSimulator:
    def create_point_source_model(
        self,
        name: str,
        ra: float,
        dec: float,
        amplitude: float,
        index: float,
        reference_energy: float = 1.0, # TeV
    ):
        """
        Creates a gammapy SkyModel for a point source with a power-law spectrum.

        Parameters
        ----------
        name : str
            Name of the source.
        ra : float
            Right Ascension of the source (degrees).
        dec : float
            Declination of the source (degrees).
        amplitude : float
            Spectral amplitude at the reference energy (e.g., in cm-2 s-1 TeV-1).
        index : float
            Spectral index of the power law.
        reference_energy : float, optional
            Reference energy for the spectral model (TeV). Default is 1.0 TeV.

        Returns
        -------
        gammapy.modeling.models.SkyModel
            The SkyModel object representing the source.
        """
        spatial_model = PointSpatialModel(
            lon_0=ra * u.deg, lat_0=dec * u.deg, frame="icrs"
        )
        spectral_model = PowerLawSpectralModel(
            index=index,
            amplitude=amplitude * u.Unit("cm-2 s-1 TeV-1"), # Example unit
            reference=reference_energy * u.TeV,
        )
        sky_model = SkyModel(
            spatial_model=spatial_model,
            spectral_model=spectral_model,
            name=name,
        )
        return sky_model

    pass
