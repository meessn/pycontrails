"""Wave-vortex downwash functions.

This module includes equations from the original CoCiP model
:cite:`schumannContrailCirrusPrediction2012`. An alternative set of equations based on
:cite:`unterstrasserPropertiesYoungContrails2016` is available in
:py:mod:`unterstrasser_wake_vortex`.

Unterstrasser Notes
-------------------

Improved estimation of the survival fraction of the contrail ice crystal number ``f_surv``
during the  wake-vortex phase. This is a parameterised model that is developed based on
outputs provided by large eddy simulations.

For comparison, CoCiP assumes that ``f_surv`` is equal to the change in the contrail ice water
content (by mass) before and after the wake vortex phase. However, for larger (smaller) ice
particles, their survival fraction by number could be smaller (larger) than their survival fraction
by mass. This is particularly important in the "soot-poor" scenario, for example, in cleaner
lean-burn engines where their soot emissions can be 3-4 orders of magnitude lower than conventional
RQL engines.

"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pycontrails.models.cocip import wind_shear
from pycontrails.physics import constants, thermo


def max_downward_displacement(
    wingspan: npt.NDArray[np.floating] | float,
    true_airspeed: npt.NDArray[np.floating],
    aircraft_mass: npt.NDArray[np.floating] | float,
    air_temperature: npt.NDArray[np.floating],
    dT_dz: npt.NDArray[np.floating],
    ds_dz: npt.NDArray[np.floating],
    air_pressure: npt.NDArray[np.floating],
    effective_vertical_resolution: float,
    wind_shear_enhancement_exponent: float | npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """
    Calculate the maximum contrail downward displacement after the wake vortex phase.

    Parameters
    ----------
    wingspan : npt.NDArray[np.floating] | float
        aircraft wingspan, [:math:`m`]
    true_airspeed : npt.NDArray[np.floating]
        true airspeed for each waypoint, [:math:`m s^{-1}`]
    aircraft_mass : npt.NDArray[np.floating] | float
        aircraft mass for each waypoint, [:math:`kg`]
    air_temperature : npt.NDArray[np.floating]
        ambient temperature for each waypoint, [:math:`K`]
    dT_dz : npt.NDArray[np.floating]
        potential temperature gradient, [:math:`K m^{-1}`]
    ds_dz : npt.NDArray[np.floating]
        Difference in wind speed over dz in the atmosphere, [:math:`m s^{-1} / m`]
    air_pressure : npt.NDArray[np.floating]
        pressure altitude at each waypoint, [:math:`Pa`]
    effective_vertical_resolution: float
        Passed through to :func:`wind_shear.wind_shear_enhancement_factor`, [:math:`m`]
    wind_shear_enhancement_exponent: npt.NDArray[np.floating] | float
        Passed through to :func:`wind_shear.wind_shear_enhancement_factor`

    Returns
    -------
    npt.NDArray[np.floating]
        Max contrail downward displacement after the wake vortex phase, [:math:`m`]

    References
    ----------
    - :cite:`holzapfelProbabilisticTwoPhaseWake2003`
    - :cite:`schumannContrailCirrusPrediction2012`
    """
    rho_air = thermo.rho_d(air_temperature, air_pressure)
    n_bv = thermo.brunt_vaisala_frequency(air_pressure, air_temperature, dT_dz)
    t_0 = effective_time_scale(wingspan, true_airspeed, aircraft_mass, rho_air)

    dz_max_strong = downward_displacement_strongly_stratified(
        wingspan, true_airspeed, aircraft_mass, rho_air, n_bv
    )

    is_weakly_stratified = n_bv * t_0 < 0.8
    if isinstance(wingspan, np.ndarray):
        wingspan = wingspan[is_weakly_stratified]
    if isinstance(aircraft_mass, np.ndarray):
        aircraft_mass = aircraft_mass[is_weakly_stratified]

    dz_max_weak = downward_displacement_weakly_stratified(
        wingspan=wingspan,
        true_airspeed=true_airspeed[is_weakly_stratified],
        aircraft_mass=aircraft_mass,
        rho_air=rho_air[is_weakly_stratified],
        n_bv=n_bv[is_weakly_stratified],
        dz_max_strong=dz_max_strong[is_weakly_stratified],
        ds_dz=ds_dz[is_weakly_stratified],
        t_0=t_0[is_weakly_stratified],
        effective_vertical_resolution=effective_vertical_resolution,
        wind_shear_enhancement_exponent=wind_shear_enhancement_exponent,
    )

    dz_max_strong[is_weakly_stratified] = dz_max_weak
    return dz_max_strong


def effective_time_scale(
    wingspan: npt.NDArray[np.floating] | float,
    true_airspeed: npt.NDArray[np.floating],
    aircraft_mass: npt.NDArray[np.floating] | float,
    rho_air: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    r"""
    Calculate the effective time scale of the wake vortex.

    Parameters
    ----------
    wingspan : npt.NDArray[np.floating]
        aircraft wingspan, [:math:`m`]
    true_airspeed : npt.NDArray[np.floating]
        true airspeed for each waypoint, [:math:`m \ s^{-1}`]
    aircraft_mass : npt.NDArray[np.floating]
        aircraft mass for each waypoint, [:math:`kg`]
    rho_air : npt.NDArray[np.floating]
        density of air for each waypoint, [:math:`kg \ m^{-3}`]

    Returns
    -------
    npt.NDArray[np.floating]
        Wake vortex effective time scale, [:math:`s`]

    Notes
    -----
    See section 2.5 (pg 547) of :cite:`schumannContrailCirrusPrediction2012`.

    References
    ----------
    - :cite:`schumannContrailCirrusPrediction2012`
    """
    c = np.pi**4 / 32
    return c * wingspan**3 * rho_air * true_airspeed / (aircraft_mass * constants.g)


def downward_displacement_strongly_stratified(
    wingspan: npt.NDArray[np.floating] | float,
    true_airspeed: npt.NDArray[np.floating],
    aircraft_mass: npt.NDArray[np.floating] | float,
    rho_air: npt.NDArray[np.floating],
    n_bv: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """
    Calculate the maximum contrail downward displacement under strongly stratified conditions.

    Parameters
    ----------
    wingspan : npt.NDArray[np.floating] | float
        aircraft wingspan, [:math:`m`]
    true_airspeed : npt.NDArray[np.floating]
        true airspeed for each waypoint, [:math:`m s^{-1}`]
    aircraft_mass : npt.NDArray[np.floating] | float
        aircraft mass for each waypoint, [:math:`kg`]
    rho_air : npt.NDArray[np.floating]
        density of air for each waypoint, [:math:`kg m^{-3}`]
    n_bv : npt.NDArray[np.floating]
        Brunt-Vaisaila frequency, [:math:`s^{-1}`]

    Returns
    -------
    npt.NDArray[np.floating]
        Maximum contrail downward displacement, strongly stratified conditions, [:math:`m`]

    Notes
    -----
    See section 2.5 (pg 547 - 548) of :cite:`schumannContrailCirrusPrediction2012`.

    References
    ----------
    - :cite:`schumannContrailCirrusPrediction2012`
    """
    c = (1.49 * 16) / (2 * np.pi**3)  # This is W2 in Schumann's Fortran code
    return (c * aircraft_mass * constants.g) / (wingspan**2 * rho_air * true_airspeed * n_bv)


def downward_displacement_weakly_stratified(
    wingspan: npt.NDArray[np.floating] | float,
    true_airspeed: npt.NDArray[np.floating],
    aircraft_mass: npt.NDArray[np.floating] | float,
    rho_air: npt.NDArray[np.floating],
    n_bv: npt.NDArray[np.floating],
    dz_max_strong: npt.NDArray[np.floating],
    ds_dz: npt.NDArray[np.floating],
    t_0: npt.NDArray[np.floating],
    effective_vertical_resolution: float,
    wind_shear_enhancement_exponent: npt.NDArray[np.floating] | float,
) -> npt.NDArray[np.floating]:
    """
    Calculate the maximum contrail downward displacement under weakly/stably stratified conditions.

    Parameters
    ----------
    wingspan : npt.NDArray[np.floating] | float
        aircraft wingspan, [:math:`m`]
    true_airspeed : npt.NDArray[np.floating]
        true airspeed for each waypoint, [:math:`m s^{-1}`]
    aircraft_mass : npt.NDArray[np.floating] | float
        aircraft mass for each waypoint, [:math:`kg`]
    rho_air : npt.NDArray[np.floating]
        density of air for each waypoint, [:math:`kg m^{-3}`]
    n_bv : npt.NDArray[np.floating]
        Brunt-Vaisaila frequency, [:math:`s^{-1}`]
    dz_max_strong : npt.NDArray[np.floating]
        Max contrail downward displacement under strongly stratified conditions, [:math:`m`]
    ds_dz : npt.NDArray[np.floating]
        Difference in wind speed over dz in the atmosphere, [:math:`m s^{-1} / m`]
    t_0 : npt.NDArray[np.floating]
        Wake vortex effective time scale, [:math:`s`]
    effective_vertical_resolution: float
        Passed through to :func:`wind_shear.wind_shear_enhancement_factor`, [:math:`m`]
    wind_shear_enhancement_exponent: npt.NDArray[np.floating] | float
        Passed through to :func:`wind_shear.wind_shear_enhancement_factor`

    Returns
    -------
    npt.NDArray[np.floating]
        Maximum contrail downward displacement, weakly/stably stratified conditions, [:math:`m`]

    Notes
    -----
    See section 2.5 (pg 548) of :cite:`schumannContrailCirrusPrediction2012`.

    References
    ----------
    - :cite:`schumannContrailCirrusPrediction2012`
    """
    b_0 = wake_vortex_separation(wingspan)
    dz_max = np.maximum(dz_max_strong, 10.0)
    shear_enhancement_factor = wind_shear.wind_shear_enhancement_factor(
        dz_max, effective_vertical_resolution, wind_shear_enhancement_exponent
    )

    # Calculate epsilon and epsilon star
    # In Schumann's Fortran code, epsn = EDR and epsn_st = EPSN
    epsn = turbulent_kinetic_energy_dissipation_rate(ds_dz, shear_enhancement_factor)
    epsn_st = normalized_dissipation_rate(epsn, wingspan, true_airspeed, aircraft_mass, rho_air)
    return b_0 * (7.68 * (1 - 4.07 * epsn_st + 5.67 * epsn_st**2) * (0.79 - n_bv * t_0) + 1.88)


def wake_vortex_separation(
    wingspan: npt.NDArray[np.floating] | float,
) -> npt.NDArray[np.floating] | float:
    """
    Calculate the wake vortex separation.

    Parameters
    ----------
    wingspan : npt.NDArray[np.floating] | float
        aircraft wingspan, [:math:`m`]

    Returns
    -------
    npt.NDArray[np.floating]
        wake vortex separation, [:math:`m`]
    """
    return (np.pi * wingspan) / 4.0


def turbulent_kinetic_energy_dissipation_rate(
    ds_dz: npt.NDArray[np.floating],
    shear_enhancement_factor: npt.NDArray[np.floating] | float = 1.0,
) -> npt.NDArray[np.floating]:
    """
    Calculate the turbulent kinetic energy dissipation rate (epsilon).

    The shear enhancement factor is used to account for any sub-grid scale turbulence.

    Parameters
    ----------
    ds_dz : npt.NDArray[np.floating]
        Difference in wind speed over dz in the atmosphere, [:math:`m s^{-1} / m`]
    shear_enhancement_factor : npt.NDArray[np.floating] | float
        Multiplication factor to enhance the wind shear

    Returns
    -------
    npt.NDArray[np.floating]
        turbulent kinetic energy dissipation rate, [:math:`m^{2} s^{-3}`]

    Notes
    -----
    - See eq. (37) in :cite:`schumannContrailCirrusPrediction2012`.
    - In a personal correspondence, Dr. Schumann identified a print error in Eq. (37)
      of the 2012 paper where the shear term should not be squared.
      The correct equation is listed in Eq. (13) :cite:`schumannTurbulentMixingStably1995`.

    References
    ----------
    - :cite:`schumannContrailCirrusPrediction2012`
    - :cite:`schumannTurbulentMixingStably1995`
    """
    return 0.5 * 0.1**2 * (ds_dz * shear_enhancement_factor**2)


def normalized_dissipation_rate(
    epsilon: npt.NDArray[np.floating],
    wingspan: npt.NDArray[np.floating] | float,
    true_airspeed: npt.NDArray[np.floating],
    aircraft_mass: npt.NDArray[np.floating] | float,
    rho_air: npt.NDArray[np.floating] | float,
) -> npt.NDArray[np.floating]:
    """
    Calculate the normalized dissipation rate of the sinking wake vortex.

    Parameters
    ----------
    epsilon: npt.NDArray[np.floating]
        turbulent kinetic energy dissipation rate, [:math:`m^{2} s^{-3}`]
    wingspan : npt.NDArray[np.floating] | float
        aircraft wingspan, [:math:`m`]
    true_airspeed : npt.NDArray[np.floating]
        true airspeed for each waypoint, [:math:`m s^{-1}`]
    aircraft_mass : npt.NDArray[np.floating]
        aircraft mass for each waypoint, [:math:`kg`]
    rho_air : npt.NDArray[np.floating]
        density of air for each waypoint, [:math:`kg m^{-3}`]

    Returns
    -------
    npt.NDArray[np.floating]
        Normalized dissipation rate of the sinking wake vortex

    Notes
    -----
    See page 548 of :cite:`schumannContrailCirrusPrediction2012`.

    References
    ----------
    - :cite:`schumannContrailCirrusPrediction2012`
    """
    c = (np.pi / 4) ** (1 / 3) * np.pi**3 / 8  # This is W6 in Schumann's Fortran code
    numer = c * (epsilon * wingspan) ** (1 / 3) * wingspan**2 * rho_air * true_airspeed

    # epsn_st = epsilon star
    epsn_st = numer / (constants.g * aircraft_mass)

    # In a personal correspondence, Schumann gives the precise value
    # of 0.358906526 here
    # In the 2012 paper, Schumann gives 0.36
    # The precise value is likely insignificant because we don't expect epsn_st
    # to be larger than 0.36
    return np.minimum(epsn_st, 0.36)


def initial_contrail_width(
    wingspan: npt.NDArray[np.floating] | float, dz_max: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    """
    Calculate the initial contrail width.

    Parameters
    ----------
    wingspan : npt.NDArray[np.floating] | float
        aircraft wingspan, [:math:`m`]
    dz_max : npt.NDArray[np.floating]
        Max contrail downward displacement after the wake vortex phase, [:math:`m`]
        Only the size of this array is used; the values are ignored.

    Returns
    -------
    npt.NDArray[np.floating]
        Initial contrail width, [:math:`m`]
    """
    return np.full_like(dz_max, np.pi / 4) * wingspan


def initial_contrail_depth(
    dz_max: npt.NDArray[np.floating], initial_wake_vortex_depth: float | npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    """
    Calculate the initial contrail depth.

    Parameters
    ----------
    dz_max : npt.NDArray[np.floating]
        Max contrail downward displacement after the wake vortex phase, [:math:`m`]
    initial_wake_vortex_depth : float | npt.NDArray[np.floating]
        Initial wake vortex depth scaling factor.
        Denoted `C_D0` in eq (14) in :cite:`schumannContrailCirrusPrediction2012`.

    Returns
    -------
    npt.NDArray[np.floating]
        Initial contrail depth, [:math:`m`]
    """
    return dz_max * initial_wake_vortex_depth
