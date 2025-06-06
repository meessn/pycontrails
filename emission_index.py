import numpy as np
import constants

def p3t3_nox(PT3_inflight, TT3_inflight, interp_func_far, interp_func_pt3, specific_humidity, WAR, engine_model):
    """
    p3t3 method to predict ei_nox for the state of the art and 2035 PW1127G engine
    can be used for both saf and kerosene, make sure to implement the correct interp_func

    Args:
        PT3_inflight (float): Inflight PT3 value.
        TT3_inflight (float): Inflight TT3 value.
        FAR_inflight (float): Inflight FAR value.
        interp_func_far (function): Interpolation function far sls graph.
        interp_func_pt3 (function): Interpolation function pt3 sls graph.

    Returns:
        float: EI_NOx at this point in flight
    """
    WAR = WAR / 100 # percentage to factor
    tolerance = 0.01  # 1% tolerance
    far_sls = interp_func_far(TT3_inflight)
    pt3_sls = interp_func_pt3(TT3_inflight)
    if WAR == 0 or abs(WAR - specific_humidity) < tolerance * specific_humidity: #ensure that regular flight without WI is not performed with WI correlation
        # print('no wi correction, just humidity')
        if engine_model == 'GTF'  or engine_model == 'GTF2035' or engine_model == 'GTF2035_wi':
            ei_nox_sls = 1.4094*pt3_sls**0.1703*np.exp(0.0011*TT3_inflight)*12.2308**(16.4302*far_sls)
        elif engine_model == 'GTF1990' or engine_model == 'GTF2000':
            ei_nox_sls = 0.1921*pt3_sls**-0.7686*np.exp(0.0084*TT3_inflight)*2.01**(60*far_sls)

        result = ei_nox_sls * (PT3_inflight / pt3_sls) ** 0.3 * np.exp(19 * (0.006344 - specific_humidity))
    elif WAR != 0:
        if engine_model == 'GTF' or engine_model == 'GTF2035' or engine_model == 'GTF2035_wi':
            ei_nox_sls = 1.4094 * pt3_sls ** 0.1703 * np.exp(0.0011 * TT3_inflight) * 12.2308 ** (16.4302 * far_sls)
        elif engine_model == 'GTF1990' or engine_model == 'GTF2000':
            ei_nox_sls = 0.1921 * pt3_sls ** -0.7686 * np.exp(0.0084 * TT3_inflight) * 2.01 ** (60 * far_sls)

        result = ei_nox_sls * (PT3_inflight / pt3_sls) ** 0.3 * np.exp(19*(0.006344-specific_humidity)) * np.exp(
            (-2.465 * WAR ** 2 - 0.915 * WAR) / (WAR ** 2 + 0.0516))

    return result

def p3t3_nox_xue(PT3_inflight, TT3_inflight, interp_func_far, interp_func_pt3, specific_humidity, WAR, engine_model):
    """
    p3t3 method to predict ei_nox for the state of the art and 2035 PW1127G engine
    can be used for both saf and kerosene, make sure to implement the correct interp_func

    Args:
        PT3_inflight (float): Inflight PT3 value.
        TT3_inflight (float): Inflight TT3 value.
        FAR_inflight (float): Inflight FAR value.
        interp_func_far (function): Interpolation function far sls graph.
        interp_func_pt3 (function): Interpolation function pt3 sls graph.

    Returns:
        float: EI_NOx at this point in flight
    """
    WAR = WAR / 100 # percentage to factor
    tolerance = 0.01  # 1% tolerance
    far_sls = interp_func_far(TT3_inflight)
    pt3_sls = interp_func_pt3(TT3_inflight)
    if WAR == 0 or abs(WAR - specific_humidity) < tolerance * specific_humidity: #ensure that regular flight without WI is not performed with WI correlation
        # print('no wi correction, just humidity')
        if engine_model == 'GTF'  or engine_model == 'GTF2035' or engine_model == 'GTF2035_wi':
            ei_nox_sls = 1.4094*pt3_sls**0.1703*np.exp(0.0011*TT3_inflight)*12.2308**(16.4302*far_sls)
        elif engine_model == 'GTF1990' or engine_model == 'GTF2000':
            ei_nox_sls = 0.1921*pt3_sls**-0.7686*np.exp(0.0084*TT3_inflight)*2.01**(60*far_sls)

        result = ei_nox_sls * (PT3_inflight / pt3_sls) ** 0.3 * np.exp(19 * (0.006344 - specific_humidity))
    elif WAR != 0:
        if engine_model == 'GTF' or engine_model == 'GTF2035' or engine_model == 'GTF2035_wi':
            ei_nox_sls = 1.4094 * pt3_sls ** 0.1703 * np.exp(0.0011 * TT3_inflight) * 12.2308 ** (16.4302 * far_sls)
        elif engine_model == 'GTF1990' or engine_model == 'GTF2000':
            ei_nox_sls = 0.1921 * pt3_sls ** -0.7686 * np.exp(0.0084 * TT3_inflight) * 2.01 ** (60 * far_sls)

        war = WAR*100
        relative_nox = 1e-6 * war ** 6 - 6e-5 * war ** 5 + 1.2e-3 * war ** 4 - 1.26e-2 * war ** 3 + 7.7e-2 * war ** 2 - 0.3337 * war + 1
        result = ei_nox_sls * (PT3_inflight / pt3_sls) ** 0.3 * np.exp(19*(0.006344-specific_humidity)) * relative_nox

    return result



def p3t3_nvpm(PT3_inflight, TT3_inflight, FAR_inflight, interp_func_far, interp_func_pt3, saf, thrust_setting):
    """
    p3t3 method to predict ei_nox for the state of the art and 2035 PW1127G engine
    can be used for both saf and kerosene, make sure to implement the correct interp_func

    Args:
        PT3_inflight (float): Inflight PT3 value.
        TT3_inflight (float): Inflight TT3 value.
        FAR_inflight (float): Inflight FAR value.
        interp_func_far (function): Interpolation function far sls graph.
        interp_func_pt3 (function): Interpolation function pt3 sls graph.

    Returns:
        float: EI_nvpm at this point in flight
    """

    far_sls = interp_func_far(TT3_inflight)
    pt3_sls = interp_func_pt3(TT3_inflight)
    t = (TT3_inflight-694.3775)/151.5468


    ei_nvpm_mass_sls = (-0.9319*t**6) + (-4.9607*t**5) - (5.0610*t**4) + (8.7014*t**3) + (24.5177*t**2) + (14.2445*t) + 2.9497

    ei_nvpm_mass = ei_nvpm_mass_sls * (PT3_inflight/pt3_sls)**1.35*(FAR_inflight/far_sls)**2.5

    v = (-4.0240*np.exp(1) + 12.2274) * np.exp(-1.2210*t) *10**14
    ei_nvpm_number = v*ei_nvpm_mass
    if saf != 0:
        del_saf = saf_correction_number(saf, thrust_setting)
        ei_nvpm_number *= 1.0 + del_saf / 100.0

    return ei_nvpm_number



def p3t3_nvpm_mass(PT3_inflight, TT3_inflight, FAR_inflight, interp_func_far, interp_func_pt3, saf, thrust_setting):
    """
    p3t3 method to predict ei_nox for the state of the art and 2035 PW1127G engine
    can be used for both saf and kerosene, make sure to implement the correct interp_func

    Args:
        PT3_inflight (float): Inflight PT3 value.
        TT3_inflight (float): Inflight TT3 value.
        FAR_inflight (float): Inflight FAR value.
        interp_func_far (function): Interpolation function far sls graph.
        interp_func_pt3 (function): Interpolation function pt3 sls graph.

    Returns:
        float: EI_nvpm at this point in flight
    """

    far_sls = interp_func_far(TT3_inflight)
    pt3_sls = interp_func_pt3(TT3_inflight)
    t = (TT3_inflight-694.3775)/151.5468

    ei_nvpm_mass_sls = (-0.9319 * t ** 6) + (-4.9607 * t ** 5) - (5.0610 * t ** 4) + (8.7014 * t ** 3) + (
                24.5177 * t ** 2) + (14.2445 * t) + 2.9497

    ei_nvpm_mass = ei_nvpm_mass_sls * (PT3_inflight / pt3_sls) ** 1.35 * (FAR_inflight / far_sls) ** 2.5

    if saf != 0:
        del_saf = saf_correction_mass(saf, thrust_setting)
        ei_nvpm_mass *= 1.0 + del_saf / 100.0

    return ei_nvpm_mass


def piecewise_exp(t, A, B, m, c):
    """
    A, B: Parameters for the exponential decay (for t > t_threshold)
    m, c: Parameters for the linear function (for t <= t_threshold)
    """
    t_threshold = (636 - 694.4) / 151.5  # Convert TT3 = 629 to corresponding t value
    c = (A + 12.2274) * np.exp(B * t_threshold) - m * t_threshold

    return np.where(
        t <= t_threshold,
        m * t + c,  # Linear part for t <= t_threshold
        (A + 12.2274) * np.exp(B * t)  # Exponential part for t > t_threshold
    )

def p3t3_nvpm_piecewise(PT3_inflight, TT3_inflight, FAR_inflight, interp_func_far, interp_func_pt3, saf, thrust_setting):
    """
    p3t3 method to predict ei_nox for the state of the art and 2035 PW1127G engine
    can be used for both saf and kerosene, make sure to implement the correct interp_func

    Args:
        PT3_inflight (float): Inflight PT3 value.
        TT3_inflight (float): Inflight TT3 value.
        FAR_inflight (float): Inflight FAR value.
        interp_func_far (function): Interpolation function far sls graph.
        interp_func_pt3 (function): Interpolation function pt3 sls graph.

    Returns:
        float: EI_nvpm at this point in flight
    """

    far_sls = interp_func_far(TT3_inflight)
    pt3_sls = interp_func_pt3(TT3_inflight)
    t = (TT3_inflight-694.3775)/151.5468


    ei_nvpm_mass_sls = (-0.9319*t**6) + (-4.9607*t**5) - (5.0610*t**4) + (8.7014*t**3) + (24.5177*t**2) + (14.2445*t) + 2.9497

    ei_nvpm_mass = ei_nvpm_mass_sls * (PT3_inflight/pt3_sls)**1.35*(FAR_inflight/far_sls)**2.5

    # Updated piecewise function for v with new coefficients
    A, B, m, c = -9.02348899, -2.17750758, -0.02119595, 7.51666667
    v = piecewise_exp(t, A, B, m, c) * 10 ** 14
    ei_nvpm_number = v*ei_nvpm_mass
    if saf != 0:
        del_saf = saf_correction_number(saf, thrust_setting)
        ei_nvpm_number *= 1.0 + del_saf / 100.0

    return ei_nvpm_number




def NOx_correlation_de_boer(PT3_inflight, TT3_inflight, TT4_inflight, specific_humidity, WAR_inflight):
    """
    NOx correlation for GTF 2035 and Water Injection

    Args:
        PT3_inflight (float): PT3 retrieved from GSP at this point of flight.
        TT3_inflight (float): TT3 retrieved from GSP at this point of flight.
        TT4_inflight (float): TT4 retrieved from GSP at this point of flight.
        WAR_inflight (float): WAR retrieved from GSP at this point of flight.



    Returns:
        float: EI_NOx at this point in flight
    """
    WAR_inflight = WAR_inflight / 100
    if WAR_inflight < 0.01:
        h = specific_humidity
    else:
        h = WAR_inflight
    result = (8.4+0.0209*np.exp(0.0082*TT3_inflight))*((PT3_inflight/300)**0.4)*np.exp(19*(0.006344-h))*((TT4_inflight-TT3_inflight)/300)**0.71


    return result

def NOx_correlation_kyprianidis(PT3_inflight, TT3_inflight, TT4_inflight, specific_humidity, WAR_inflight):
    """
    NOx correlation for GTF 2035 and Water Injection

    Args:
        PT3_inflight (float): PT3 retrieved from GSP at this point of flight.
        TT3_inflight (float): TT3 retrieved from GSP at this point of flight.
        TT4_inflight (float): TT4 retrieved from GSP at this point of flight.
        WAR_inflight (float): WAR retrieved from GSP at this point of flight.



    Returns:
        float: EI_NOx at this point in flight
    """
    WAR_inflight = WAR_inflight / 100
    if WAR_inflight < 0.01:
        h = specific_humidity
    else:
        h = WAR_inflight

    result = (8.4+0.0209*np.exp(0.0082*TT3_inflight))*((PT3_inflight/30)**0.4)*np.exp(19*(0.006344-h))*((TT4_inflight-TT3_inflight)/300)**0.0


    return result

def NOx_correlation_kypriandis_optimized_tf(PT3_inflight, TT3_inflight, TT4_inflight, specific_humidity, WAR_inflight):
    """
    NOx correlation for GTF 2035 and Water Injection

    Args:
        PT3_inflight (float): PT3 retrieved from GSP at this point of flight.
        TT3_inflight (float): TT3 retrieved from GSP at this point of flight.
        TT4_inflight (float): TT4 retrieved from GSP at this point of flight.
        WAR_inflight (float): WAR retrieved from GSP at this point of flight.



    Returns:
        float: EI_NOx at this point in flight
    """
    WAR_inflight = WAR_inflight / 100
    if WAR_inflight < 0.01:
        h = specific_humidity
    else:
        h = WAR_inflight
    result = (8.4+0.0209*np.exp(0.0082*TT3_inflight))*((PT3_inflight/30)**0.4)*np.exp(19*(0.006344-h))*((TT4_inflight-TT3_inflight)/300)**-0.3747


    return result

def NOx_correlation_kaiser_optimized_tf(PT3_inflight, TT3_inflight, specific_humidity, WAR_inflight):
    """
    NOx correlation for GTF 2035 and Water Injection

    Args:
        PT3_inflight (float): PT3 retrieved from GSP at this point of flight.
        TT3_inflight (float): TT3 retrieved from GSP at this point of flight.

        WAR_inflight (float): WAR retrieved from GSP at this point of flight.



    Returns:
        float: EI_NOx at this point in flight
    """
    WAR_inflight = WAR_inflight / 100
    if WAR_inflight < 0.01:
        h = specific_humidity
    else:
        h = WAR_inflight
    result = (32 *
          np.exp((TT3_inflight - 826) / 194) *
          (PT3_inflight / 29.65) ** 0.4 *
          0.6349 *
          np.exp((-2.465 * h ** 2 - 0.915 *h) / (h ** 2 + 0.0516))
         )

    return result

def NOx_correlation_kaiser(PT3_inflight, TT3_inflight, specific_humidity, WAR_inflight):
    """
    NOx correlation for GTF 2035 and Water Injection

    Args:
        PT3_inflight (float): PT3 retrieved from GSP at this point of flight.
        TT3_inflight (float): TT3 retrieved from GSP at this point of flight.

        WAR_inflight (float): WAR retrieved from GSP at this point of flight.



    Returns:
        float: EI_NOx at this point in flight
    """
    WAR_inflight = WAR_inflight / 100
    if WAR_inflight < 0.01:
        h = specific_humidity
    else:
        h = WAR_inflight
    result = (32 *
          np.exp((TT3_inflight - 826) / 194) *
          (PT3_inflight / 29.65) ** 0.4 *
          0.72 *
          np.exp((-2.465 * h ** 2 - 0.915 *h) / (h ** 2 + 0.0516))
         )

    return result


def meem_nvpm(altitude, mach, altitude_cruise, flight_phase, saf):
    """
        MEEM method for nvPM mass and number, from the paper:

        A New Approach to Estimate Particulate Matter Emissions From Ground Certification Data:
        The nvPM Mission Emissions Estimation Methodology

        by Ahrens et al. (2022)

        Valid above 3000 ft (914 m)
        and poor results for approach?

        Args:




        Returns:

        """


    # first convert altitude from m to ft, as this is used in the paper too
    altitude_ft = altitude * constants.m_to_ft
    altitude_cruise_ft = altitude_cruise * constants.m_to_ft
    T_sls = 288.15 #K
    P_sls = 101325 #Pa
    operating_pr_icao = 31.7
    rated_thrust_icao = 120.4

    if altitude_ft > 3000:
        """STEP 1"""
        if altitude_ft < 36089:
            T_alt = T_sls*(1-(altitude_ft/145442))
            P_alt = P_sls*(1-(altitude_ft/145442))**5.255876
        else:
            T_alt = T_sls*0.751865
            P_alt = P_sls*0.223361*np.exp(-1*((altitude_ft-36089)/20806))

        #total properties
        Tt_alt = T_alt*(1 + ((constants.kappa-1)/2)*(mach**2))
        Pt_alt = P_alt*(1 + ((constants.kappa-1)/2)*(mach**2))**(constants.kappa/(constants.kappa-1))

        # CLIMB
        if flight_phase == 'climb':
            eta_comp = 0.88
            pressure_coef = 0.85 + (1.15-0.85)*((altitude_ft-3000)/(altitude_cruise_ft-3000))
        elif flight_phase == 'cruise':
            eta_comp = 0.88
            pressure_coef = 0.95
        elif flight_phase == 'descent':
            eta_comp = 0.70
            pressure_coef = 0.12
        else:
            return None, None

        P3_alt = (1 + pressure_coef*(operating_pr_icao - 1))*Pt_alt
        T3_alt = (1 + (1/eta_comp)*(((P3_alt/Pt_alt)**((constants.kappa-1)/constants.kappa))-1))*Tt_alt

        """STEP 2"""
        T3_gr = T3_alt
        P3_gr = ((1 + eta_comp*((T3_gr/T_sls)-1))**(constants.kappa/(constants.kappa-1)))*P_sls

        F_gr_F_rated = ((P3_gr/P_sls)-1)/(operating_pr_icao-1)

        """STEP 3"""
        #4 point interpolation method because for EI_mass the
        # max is at T/O condition and EI_number is also very close to T/O. so 57.5 or 92.5 will be too far off

        EI_mass_icao_sl = [7.8, 0.6, 26.3, 36.3]
        EI_number_icao_sl = [5.78e15, 3.85e14, 1.60e15, 1.45e15]


        thrust_setting_icao = [0.07, 0.3, 0.85, 1]

        # Perform interpolation directly and handles the values <0.07 and >1 correctly
        EI_mass_interpolated_gr = np.interp(F_gr_F_rated, thrust_setting_icao, EI_mass_icao_sl)
        EI_number_interpolated_gr = np.interp(F_gr_F_rated, thrust_setting_icao, EI_number_icao_sl)


        """STEP 4"""
        EI_mass_alt = EI_mass_interpolated_gr*((P3_alt/P3_gr)**1.35)*1.1**2.5
        EI_number_alt = EI_mass_alt*(EI_number_interpolated_gr/EI_mass_interpolated_gr)

        if saf != 0:
            del_saf_mass = saf_correction_mass(saf, F_gr_F_rated)
            EI_mass_alt *= 1.0 + del_saf_mass / 100.0
            del_saf_number = saf_correction_number(saf, F_gr_F_rated)
            EI_number_alt *= 1.0 + del_saf_number / 100.0

        return EI_mass_alt, EI_number_alt

    else:
        return None, None
    return None, None


def p3t3_nvpm_meem(PT3_inflight, TT3_inflight, FAR_inflight, interp_func_far, interp_func_pt3, saf, thrust_setting, engine_model):
    """
    adjusted meem method

    Args:
        PT3_inflight (float): Inflight PT3 value.
        TT3_inflight (float): Inflight TT3 value.
        FAR_inflight (float): Inflight FAR value.
        interp_func_far (function): Interpolation function far sls graph.
        interp_func_pt3 (function): Interpolation function pt3 sls graph.

    Returns:
        float: EI_nvpm at this point in flight
    """

    far_sls = interp_func_far(TT3_inflight)
    pt3_sls = interp_func_pt3(TT3_inflight)

    F_gr_F_rated = thrust_setting

    if engine_model == 'GTF' or engine_model == 'GTF2035'or engine_model == 'GTF2035_wi':
        EI_mass_icao_sl = [7.8, 0.6, 26.3, 36.3]
        EI_number_icao_sl = [5.78e15, 3.85e14, 1.60e15, 1.45e15]
    elif engine_model == 'GTF1990': #see excel engine model cfm56!
        EI_mass_icao_sl = [30.6, 58.2, 92.3, 102]
        EI_number_icao_sl = [4.43e15, 9.03e15, 2.53e15, 1.62e15]
    elif engine_model == 'GTF2000':
        EI_mass_icao_sl = [5.5, 3.13, 50.8, 64]
        EI_number_icao_sl = [7.98e14, 4.85e14, 1.39e15, 1.02e15]
    else:
        raise ValueError(f"Unsupported engine_model: {engine_model}.")


    thrust_setting_icao = [0.07, 0.3, 0.85, 1]
    ei_nvpm_mass_sls = np.interp(F_gr_F_rated, thrust_setting_icao, EI_mass_icao_sl)
    # print(ei_nvpm_mass_sls)
    # print(F_gr_F_rated)
    ei_nvpm_number_sls = np.interp(F_gr_F_rated, thrust_setting_icao, EI_number_icao_sl)


    # print('meemp3t3 mass', ei_nvpm_mass_sls)
    ei_nvpm_mass = ei_nvpm_mass_sls * (PT3_inflight/pt3_sls)**1.35*(FAR_inflight/far_sls)**2.5
    # print('far', (FAR_inflight/far_sls)**2.5)
    # print('pt3', (PT3_inflight/pt3_sls)**1.35)
    # print('mass', ei_nvpm_mass)
    # print('ratio', (ei_nvpm_number_sls / ei_nvpm_mass_sls))
    # print((PT3_inflight/pt3_sls)**1.35*(FAR_inflight/far_sls)**2.5)
    ei_nvpm_number = ei_nvpm_mass * (ei_nvpm_number_sls / ei_nvpm_mass_sls)
    if saf != 0:
        del_saf = saf_correction_number(saf, thrust_setting)
        ei_nvpm_number *= 1.0 + del_saf / 100.0
    # print(result)

    return ei_nvpm_number


def p3t3_nvpm_meem_mass(PT3_inflight, TT3_inflight, FAR_inflight, interp_func_far, interp_func_pt3, saf, thrust_setting, engine_model):
    """
    adjusted meem mass

    Args:
        PT3_inflight (float): Inflight PT3 value.
        TT3_inflight (float): Inflight TT3 value.
        FAR_inflight (float): Inflight FAR value.
        interp_func_far (function): Interpolation function far sls graph.
        interp_func_pt3 (function): Interpolation function pt3 sls graph.

    Returns:
        float: EI_nvpm at this point in flight
    """

    far_sls = interp_func_far(TT3_inflight)
    pt3_sls = interp_func_pt3(TT3_inflight)

    F_gr_F_rated = thrust_setting

    if engine_model == 'GTF' or engine_model == 'GTF2035' or engine_model == 'GTF2035_wi':
        EI_mass_icao_sl = [7.8, 0.6, 26.3, 36.3]
    elif engine_model == 'GTF1990':
        EI_mass_icao_sl = [30.6, 58.2, 92.3, 102] # see engine model cfm56 excel for calculation smoke number to nvPM number and mass
    elif engine_model == 'GTF2000':
        EI_mass_icao_sl = [5.5, 3.13, 50.8, 64]
    else:
        raise ValueError(f"Unsupported engine_model: {engine_model}.")

    thrust_setting_icao = [0.07, 0.3, 0.85, 1]
    ei_nvpm_mass_sls = np.interp(F_gr_F_rated, thrust_setting_icao, EI_mass_icao_sl)

    ei_nvpm_mass = ei_nvpm_mass_sls * (PT3_inflight / pt3_sls) ** 1.35 * (FAR_inflight / far_sls) ** 2.5

    if saf != 0:
        del_saf = saf_correction_mass(saf, thrust_setting)
        ei_nvpm_mass *= 1.0 + del_saf / 100.0

    return ei_nvpm_mass


def thrust_setting(engine_model, tt3, interp_func_pt3, interp_func_fgr):


    # p_amb = 1.01325
    if engine_model == 'GTF':
        rated_thrust_icao = 120.4
    elif engine_model == 'GTF2035' or engine_model == 'GTF2035_wi':
        rated_thrust_icao = 120.4
    elif engine_model == 'GTF1990' or engine_model == 'GTF2000':
        rated_thrust_icao = 120.1
    else:
        raise ValueError(f"Unsupported engine_model: {engine_model}. ")

    # pt3_sls = interp_func_pt3(tt3)
    f_gr = interp_func_fgr(tt3)
    thrust_set = f_gr/rated_thrust_icao
    thrust_clipped = max(0.07, min(1.0, thrust_set))
    return thrust_clipped

def saf_correction_mass(saf, thrust_setting):
    delta_h = 0.015*saf
    a0 = -124.05
    a1 = 1.02
    a2 = 0.6
    d_nvpm_ein_pct = (a0 + a1 * (thrust_setting * 100.0)) * delta_h

    # Adjust when delta_h is large
    if isinstance(delta_h, np.ndarray):
        filt = delta_h > 0.5
        d_nvpm_ein_pct[filt] *= np.exp(0.5 * (a2 - delta_h[filt]))
    elif delta_h > 0.5:
        d_nvpm_ein_pct *= np.exp(0.5 * (a2 - delta_h))

    d_nvpm_ein_pct = max(-90.0, min(d_nvpm_ein_pct, 0.0))
    return d_nvpm_ein_pct

def saf_correction_number(saf, thrust_setting):
    delta_h = 0.015*saf
    a0 = -114.21
    a1 = 1.06
    a2 = 0.5
    d_nvpm_ein_pct = (a0 + a1 * (thrust_setting * 100.0)) * delta_h

    # Adjust when delta_h is large
    if isinstance(delta_h, np.ndarray):
        filt = delta_h > 0.5
        d_nvpm_ein_pct[filt] *= np.exp(0.5 * (a2 - delta_h[filt]))
    elif delta_h > 0.5:
        d_nvpm_ein_pct *= np.exp(0.5 * (a2 - delta_h))

    d_nvpm_ein_pct = max(-90.0, min(d_nvpm_ein_pct, 0.0))
    return d_nvpm_ein_pct