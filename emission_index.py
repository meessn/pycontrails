import numpy as np
import constants

def p3t3_nox(PT3_inflight, TT3_inflight, interp_func_far, interp_func_pt3, specific_humidity):
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

    far_sls = interp_func_far(TT3_inflight)
    pt3_sls = interp_func_pt3(TT3_inflight)
    # print(far_sls)
    # print(pt3_sls)
    # V2
    ei_nox_sls = 0.8699*pt3_sls**0.0765*np.exp(0.0024*TT3_inflight)*2.01**(60*far_sls)
    # print(ei_nox_sls)
    result = ei_nox_sls*(PT3_inflight/pt3_sls)**0.3*np.exp(19*(0.006344-specific_humidity))

    return result

def p3t3_nox_wi(PT3_inflight, TT3_inflight, interp_func_far, interp_func_pt3, specific_humidity, war):
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

    far_sls = interp_func_far(TT3_inflight)
    pt3_sls = interp_func_pt3(TT3_inflight)
    # print(far_sls)
    # print(pt3_sls)
    ei_nox_sls = 0.8699*pt3_sls**0.0765*np.exp(0.0024*TT3_inflight)*2.01**(60*far_sls)
    # print(ei_nox_sls)
    result = ei_nox_sls*(PT3_inflight/pt3_sls)**0.3*np.exp(19*(0.006344-specific_humidity)) * np.exp((-2.465*war**2-0.915*war)/(war**2+0.0516))

    return result

def p3t3_nvpm(PT3_inflight, TT3_inflight, FAR_inflight, interp_func_far, interp_func_pt3, saf):
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
    t = (TT3_inflight-696.4)/154.5

    if saf == 0:
        ei_nvpm_mass_sls = (-1.4110*t**6) + (-5.3007*t**5) - (3.5961*t**4) + (9.2888*t**3) + (23.6098*t**2) + (13.9142*t) + 2.9213
        print(ei_nvpm_mass_sls)
        ei_nvpm_mass = ei_nvpm_mass_sls * (PT3_inflight/pt3_sls)**1.35*(FAR_inflight/far_sls)**2.5
        # ei_nvpm_mass = ei_nvpm_mass_sls * (PT3_inflight / pt3_sls) ** 1.35 * (1.1) ** 2.5
        print(ei_nvpm_mass)
        v = (-4.0106*np.exp(1) + 12.2323) * np.exp(-1.2529*t) *10**14
        result = v*ei_nvpm_mass
        print(result)
    elif saf == 20:
        ei_nvpm_mass_sls = (-0.5444 * t ** 6) + (-4.4315 * t ** 5) - (5.3065 * t ** 4) + (8.8020 * t ** 3) + (
                    23.0131 * t ** 2) + (12.3004 * t) + 2.2554
        # print(ei_nvpm_mass_sls)
        ei_nvpm_mass = ei_nvpm_mass_sls * (PT3_inflight / pt3_sls) ** 1.35 * (FAR_inflight / far_sls) ** 2.5
        # print(ei_nvpm_mass)
        v = (-3.9951 * np.exp(1) + 12.2380) * np.exp(-1.2642 * t) * 10 ** 14
        result = v * ei_nvpm_mass
    elif saf == 100:
        ei_nvpm_mass_sls = (0.3255 * t ** 6) + (-3.6254 * t ** 5) - (6.9934 * t ** 4) + (8.2737 * t ** 3) + (
                22.5514 * t ** 2) + (10.9926 * t) + 1.7097
        # print(ei_nvpm_mass_sls)
        ei_nvpm_mass = ei_nvpm_mass_sls * (PT3_inflight / pt3_sls) ** 1.35 * (FAR_inflight / far_sls) ** 2.5
        # print(ei_nvpm_mass)
        v = (-3.9663 * np.exp(1) + 12.2486) * np.exp(-1.2962 * t) * 10 ** 14
        result = v * ei_nvpm_mass
    return result

# def p3t3_nvpm_meem(PT3_inflight, TT3_inflight, FAR_inflight, interp_func_far, interp_func_pt3, saf, flight_phase):
#     """
#     p3t3 method to predict ei_nox for the state of the art and 2035 PW1127G engine
#     can be used for both saf and kerosene, make sure to implement the correct interp_func
#
#     Args:
#         PT3_inflight (float): Inflight PT3 value.
#         TT3_inflight (float): Inflight TT3 value.
#         FAR_inflight (float): Inflight FAR value.
#         interp_func_far (function): Interpolation function far sls graph.
#         interp_func_pt3 (function): Interpolation function pt3 sls graph.
#
#     Returns:
#         float: EI_nvpm at this point in flight
#     """
#     # average EI_num_gr / EI_mass_gr of engine variants: 3.69222
#     average_num_mass_gr = 6.1537
#
#     far_sls = interp_func_far(TT3_inflight)
#     pt3_sls = interp_func_pt3(TT3_inflight)
#     t = (TT3_inflight-696.4)/154.5
#
#     if saf == 0:
#         ei_nvpm_mass_sls = (-1.4110*t**6) + (-5.3007*t**5) - (3.5961*t**4) + (9.2888*t**3) + (23.6098*t**2) + (13.9142*t) + 2.9213
#         print(ei_nvpm_mass_sls)
#         ei_nvpm_mass = ei_nvpm_mass_sls * (PT3_inflight/pt3_sls)**1.35*(FAR_inflight/far_sls)**2.5
#         # ei_nvpm_mass = ei_nvpm_mass_sls * (PT3_inflight / pt3_sls) ** 1.35 * (1.1) ** 2.5
#         if flight_phase == 'descent':
#             result = ei_nvpm_mass*average_num_mass_gr * 10 ** 14
#         else:
#             v = (-4.0106*np.exp(1) + 12.2323) * np.exp(-1.2529*t) *10**14
#             result = v*ei_nvpm_mass
#             print(result)
#     elif saf == 20:
#         ei_nvpm_mass_sls = (-0.5444 * t ** 6) + (-4.4315 * t ** 5) - (5.3065 * t ** 4) + (8.8020 * t ** 3) + (
#                     23.0131 * t ** 2) + (12.3004 * t) + 2.2554
#         # print(ei_nvpm_mass_sls)
#         ei_nvpm_mass = ei_nvpm_mass_sls * (PT3_inflight / pt3_sls) ** 1.35 * (FAR_inflight / far_sls) ** 2.5
#         # print(ei_nvpm_mass)
#         if flight_phase == 'descent':
#             result = ei_nvpm_mass*average_num_mass_gr * 10 ** 14
#
#         else:
#             v = (-3.9951 * np.exp(1) + 12.2380) * np.exp(-1.2642 * t) * 10 ** 14
#             result = v * ei_nvpm_mass
#     elif saf == 100:
#         ei_nvpm_mass_sls = (0.3255 * t ** 6) + (-3.6254 * t ** 5) - (6.9934 * t ** 4) + (8.2737 * t ** 3) + (
#                 22.5514 * t ** 2) + (10.9926 * t) + 1.7097
#         # print(ei_nvpm_mass_sls)
#         ei_nvpm_mass = ei_nvpm_mass_sls * (PT3_inflight / pt3_sls) ** 1.35 * (FAR_inflight / far_sls) ** 2.5
#         # print(ei_nvpm_mass)
#         if flight_phase == 'descent':
#             result = ei_nvpm_mass*average_num_mass_gr * 10 ** 14
#         else:
#             v = (-3.9663 * np.exp(1) + 12.2486) * np.exp(-1.2962 * t) * 10 ** 14
#             result = v * ei_nvpm_mass
#
#     return result

def p3t3_nvpm_mass(PT3_inflight, TT3_inflight, FAR_inflight, interp_func_far, interp_func_pt3, saf):
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
    t = (TT3_inflight-696.4)/154.5

    if saf == 0:
        ei_nvpm_mass_sls = (-1.4110 * t ** 6) + (-5.3007 * t ** 5) - (3.5961 * t ** 4) + (9.2888 * t ** 3) + (
                    23.6098 * t ** 2) + (13.9142 * t) + 2.9213
        print(ei_nvpm_mass_sls)
        result = ei_nvpm_mass_sls * (PT3_inflight / pt3_sls) ** 1.35 * (FAR_inflight / far_sls) ** 2.5
        # ei_nvpm_mass = ei_nvpm_mass_sls * (PT3_inflight / pt3_sls) ** 1.35 * (1.1) ** 2.5

    elif saf == 20:
        ei_nvpm_mass_sls = (-0.5444 * t ** 6) + (-4.4315 * t ** 5) - (5.3065 * t ** 4) + (8.8020 * t ** 3) + (
                23.0131 * t ** 2) + (12.3004 * t) + 2.2554
        # print(ei_nvpm_mass_sls)
        result = ei_nvpm_mass_sls * (PT3_inflight / pt3_sls) ** 1.35 * (FAR_inflight / far_sls) ** 2.5

    elif saf == 100:
        ei_nvpm_mass_sls = (0.3255 * t ** 6) + (-3.6254 * t ** 5) - (6.9934 * t ** 4) + (8.2737 * t ** 3) + (
                22.5514 * t ** 2) + (10.9926 * t) + 1.7097
        # print(ei_nvpm_mass_sls)
        result = ei_nvpm_mass_sls * (PT3_inflight / pt3_sls) ** 1.35 * (FAR_inflight / far_sls) ** 2.5


    return result

def NOx_correlation_de_boer(PT3_inflight, TT3_inflight, TT4_inflight, WAR_inflight):
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
    result = (8.4+0.0209*np.exp(0.0082*TT3_inflight))*((PT3_inflight/300)**0.4)*np.exp(19*(0.006344-WAR_inflight))*((TT4_inflight-TT3_inflight)/300)**0.71


    return result

def NOx_correlation_kyprianidis(PT3_inflight, TT3_inflight, TT4_inflight, WAR_inflight):
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
    result = (8.4+0.0209*np.exp(0.0082*TT3_inflight))*((PT3_inflight/30)**0.4)*np.exp(19*(0.006344-WAR_inflight))*((TT4_inflight-TT3_inflight)/300)**0.0


    return result

def NOx_correlation_kypriandis_optimized_tf(PT3_inflight, TT3_inflight, TT4_inflight, WAR_inflight):
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
    result = (8.4+0.0209*np.exp(0.0082*TT3_inflight))*((PT3_inflight/30)**0.4)*np.exp(19*(0.006344-WAR_inflight))*((TT4_inflight-TT3_inflight)/300)**-0.4690


    return result

def NOx_correlation_kaiser_optimized_tf(PT3_inflight, TT3_inflight, WAR_inflight):
    """
    NOx correlation for GTF 2035 and Water Injection

    Args:
        PT3_inflight (float): PT3 retrieved from GSP at this point of flight.
        TT3_inflight (float): TT3 retrieved from GSP at this point of flight.

        WAR_inflight (float): WAR retrieved from GSP at this point of flight.



    Returns:
        float: EI_NOx at this point in flight
    """
    result = (32 *
          np.exp((TT3_inflight - 826) / 194) *
          (PT3_inflight / 29.65) ** 0.4 *
          0.6138 *
          np.exp((-2.465 * WAR_inflight ** 2 - 0.915 * WAR_inflight) / (WAR_inflight ** 2 + 0.0516))
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
        # print('T_alt', T_alt)
        # print('P_alt', P_alt)
        #total properties
        Tt_alt = T_alt*(1 + ((constants.kappa-1)/2)*(mach**2))
        Pt_alt = P_alt*(1 + ((constants.kappa-1)/2)*(mach**2))**(constants.kappa/(constants.kappa-1))
        # print('Tt_alt', Tt_alt)
        # print('Pt_alt', Pt_alt)
        # Determine eta_comp and pressure_coef dependent on flight stage:
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
        # print('eta_comp', eta_comp)
        # print('pressure coef', pressure_coef)
        P3_alt = (1 + pressure_coef*(operating_pr_icao - 1))*Pt_alt
        T3_alt = (1 + (1/eta_comp)*(((P3_alt/Pt_alt)**((constants.kappa-1)/constants.kappa))-1))*Tt_alt
        # print('T3_alt', T3_alt)
        # print('P3_alt', P3_alt)
        """STEP 2"""
        T3_gr = T3_alt
        P3_gr = ((1 + eta_comp*((T3_gr/T_sls)-1))**(constants.kappa/(constants.kappa-1)))*P_sls
        # print('T3_gr', T3_gr)
        # print('P3_gr', P3_gr)
        F_gr_F_rated = ((P3_gr/P_sls)-1)/(operating_pr_icao-1)
        # print('F_gr_F_rated', F_gr_F_rated)
        """STEP 3"""
        #4 point interpolation method because for EI_mass the
        # max is at T/O condition and EI_number is also very close to T/O. so 57.5 or 92.5 will be too far off
        if saf == False:
            EI_mass_icao_sl = [7.8, 0.6, 26.3, 36.3]
            EI_number_icao_sl = [5.78e15, 3.85e14, 1.60e15, 1.45e15]

        thrust_setting_icao = [0.07, 0.3, 0.85, 1]

        # Perform interpolation directly and handles the values <0.07 and >1 correctly
        EI_mass_interpolated_gr = np.interp(F_gr_F_rated, thrust_setting_icao, EI_mass_icao_sl)
        EI_number_interpolated_gr = np.interp(F_gr_F_rated, thrust_setting_icao, EI_number_icao_sl)
        # print('EI_mass_interpolated_gr',EI_mass_interpolated_gr )
        # print('EI_number_interpolated_gr', EI_number_interpolated_gr)

        """STEP 4"""
        EI_mass_alt = EI_mass_interpolated_gr*((P3_alt/P3_gr)**1.35)*1.1**2.5
        EI_number_alt = EI_mass_alt*(EI_number_interpolated_gr/EI_mass_interpolated_gr)
        # print('meem results')
        # print(EI_mass_alt)
        # print(EI_number_alt)

        return EI_mass_alt, EI_number_alt

    else:
        return None, None
    return None, None

def meem_nvpm_5_point(altitude, mach, altitude_cruise, flight_phase, saf):
    """
        MEEM method for nvPM mass and number, from the paper:

        A New Approach to Estimate Particulate Matter Emissions From Ground Certification Data:
        The nvPM Mission Emissions Estimation Methodology

        by Ahrens et al. (2022)

        Valid above 3000 ft (914 m)
        and poor results for approach?

        5 point interpolation for nvpm_number
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
        # print('T_alt', T_alt)
        # print('P_alt', P_alt)
        #total properties
        Tt_alt = T_alt*(1 + ((constants.kappa-1)/2)*(mach**2))
        Pt_alt = P_alt*(1 + ((constants.kappa-1)/2)*(mach**2))**(constants.kappa/(constants.kappa-1))
        # print('Tt_alt', Tt_alt)
        # print('Pt_alt', Pt_alt)
        # Determine eta_comp and pressure_coef dependent on flight stage:
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
        # print('eta_comp', eta_comp)
        # print('pressure coef', pressure_coef)
        P3_alt = (1 + pressure_coef*(operating_pr_icao - 1))*Pt_alt
        T3_alt = (1 + (1/eta_comp)*(((P3_alt/Pt_alt)**((constants.kappa-1)/constants.kappa))-1))*Tt_alt
        # print('T3_alt', T3_alt)
        # print('P3_alt', P3_alt)
        """STEP 2"""
        T3_gr = T3_alt
        P3_gr = ((1 + eta_comp*((T3_gr/T_sls)-1))**(constants.kappa/(constants.kappa-1)))*P_sls
        # print('T3_gr', T3_gr)
        # print('P3_gr', P3_gr)
        F_gr_F_rated = ((P3_gr/P_sls)-1)/(operating_pr_icao-1)
        print('F_gr_F_rated', F_gr_F_rated)
        """STEP 3"""
        #4 point interpolation method because for EI_mass the
        # max is at T/O condition and EI_number is also very close to T/O. so 57.5 or 92.5 will be too far off
        if saf == False:
            EI_mass_icao_sl = [7.8, 0.6, 26.3, 36.3]
            """5 point for EI_number!!"""
            EI_number_icao_sl = [5.78e15, 3.85e14, 1.60e15, 1.60e15, 1.45e15]

        thrust_setting_icao = [0.07, 0.3, 0.85, 1]
        thrust_setting_icao_number = [0.07, 0.3, 0.85, 0.925, 1]

        # Perform interpolation directly and handles the values <0.07 and >1 correctly
        EI_mass_interpolated_gr = np.interp(F_gr_F_rated, thrust_setting_icao, EI_mass_icao_sl)
        EI_number_interpolated_gr = np.interp(F_gr_F_rated, thrust_setting_icao_number, EI_number_icao_sl)
        # print('EI_mass_interpolated_gr',EI_mass_interpolated_gr )
        # print('EI_number_interpolated_gr', EI_number_interpolated_gr)

        """STEP 4"""
        EI_mass_alt = EI_mass_interpolated_gr*((P3_alt/P3_gr)**1.35)*1.1**2.5
        EI_number_alt = EI_mass_alt*(EI_number_interpolated_gr/EI_mass_interpolated_gr)
        # print('meem results')
        # print(EI_mass_alt)
        # print(EI_number_alt)

        return EI_mass_alt, EI_number_alt

    else:
        return None, None
    return None, None

def p3t3_meem_nvpm(altitude, air_temperature, air_pressure , mach, PT3_inflight, TT3_inflight, P3_inflight, T3_inflight, eta_comp, saf):
    """
        meem with actual values from GSP12

        Args:




        Returns:

        """


    # first convert altitude from m to ft, as this is used in the paper too
    altitude_ft = altitude * constants.m_to_ft
    # altitude_cruise_ft = altitude_cruise * constants.m_to_ft
    T_sls = 288.15 #K
    P_sls = 101325 #Pa
    operating_pr_icao = 31.7
    rated_thrust_icao = 120.4

    if altitude_ft > 3000:
        # """STEP 1"""
        # if altitude_ft < 36089:
        #     T_alt = T_sls*(1-(altitude_ft/145442))
        #     P_alt = P_sls*(1-(altitude_ft/145442))**5.255876
        # else:
        #     T_alt = T_sls*0.751865
        #     P_alt = P_sls*0.223361*np.exp(-1*((altitude_ft-36089)/20806))

        T_alt = air_temperature
        P_alt = air_pressure
        # print('T_alt', T_alt)
        # print('P_alt', P_alt)
        #total properties
        Tt_alt = T_alt*(1 + ((constants.kappa-1)/2)*(mach**2))
        Pt_alt = P_alt*(1 + ((constants.kappa-1)/2)*(mach**2))**(constants.kappa/(constants.kappa-1))
        # print('Tt_alt', Tt_alt)
        # print('Pt_alt', Pt_alt)
        # Determine eta_comp and pressure_coef dependent on flight stage:
        # CLIMB
        # if flight_phase == 'climb':
        #     eta_comp = 0.88
        #     pressure_coef = 0.85 + (1.15-0.85)*((altitude_ft-3000)/(altitude_cruise_ft-3000))
        # elif flight_phase == 'cruise':
        #     eta_comp = 0.88
        #     pressure_coef = 0.95
        # elif flight_phase == 'descent':
        #     eta_comp = 0.70
        #     pressure_coef = 0.12
        # else:
        #     return None, None
        # print('eta_comp', eta_comp)
        # print('pressure coef', pressure_coef)
        P3_alt = P3_inflight
        T3_alt = T3_inflight
        # print('T3_alt', T3_alt)
        # print('P3_alt', P3_alt)
        """STEP 2"""
        T3_gr = T3_alt
        P3_gr = ((1 + eta_comp*((T3_gr/T_sls)-1))**(constants.kappa/(constants.kappa-1)))*P_sls
        # print('T3_gr', T3_gr)
        # print('P3_gr', P3_gr)
        F_gr_F_rated = ((P3_gr/P_sls)-1)/(operating_pr_icao-1)
        # print('F_gr_F_rated', F_gr_F_rated)
        """STEP 3"""
        #4 point interpolation method because for EI_mass the
        # max is at T/O condition and EI_number is also very close to T/O. so 57.5 or 92.5 will be too far off
        if saf == False:
            EI_mass_icao_sl = [7.8, 0.6, 26.3, 36.3]
            EI_number_icao_sl = [5.78e15, 3.85e14, 1.60e15, 1.45e15]

        thrust_setting_icao = [0.07, 0.3, 0.85, 1]

        # Perform interpolation directly and handles the values <0.07 and >1 correctly
        EI_mass_interpolated_gr = np.interp(F_gr_F_rated, thrust_setting_icao, EI_mass_icao_sl)
        EI_number_interpolated_gr = np.interp(F_gr_F_rated, thrust_setting_icao, EI_number_icao_sl)
        # print('EI_mass_interpolated_gr',EI_mass_interpolated_gr )
        # print('EI_number_interpolated_gr', EI_number_interpolated_gr)

        """STEP 4"""
        EI_mass_alt = EI_mass_interpolated_gr*((P3_alt/P3_gr)**1.35)*1.1**2.5
        EI_number_alt = EI_mass_alt*(EI_number_interpolated_gr/EI_mass_interpolated_gr)
        # print('meem results')
        # print(EI_mass_alt)
        # print(EI_number_alt)

        return EI_mass_alt, EI_number_alt

    else:
        return None, None
    return None, None

def p3t3_nvpm_meem(PT3_inflight, TT3_inflight, FAR_inflight, interp_func_far, interp_func_pt3, saf):
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
    # t = (TT3_inflight-696.4)/154.5
    p_amb = 1.01325
    operating_pr_icao = 31.7
    F_gr_F_rated = ((pt3_sls/p_amb) - 1) / (operating_pr_icao - 1)
    if saf == 0:
        EI_mass_icao_sl = [7.8, 0.6, 26.3, 36.3]
        EI_number_icao_sl = [5.78e15, 3.85e14, 1.60e15, 1.45e15]

        thrust_setting_icao = [0.07, 0.3, 0.85, 1]
        ei_nvpm_mass_sls = np.interp(F_gr_F_rated, thrust_setting_icao, EI_mass_icao_sl)
        ei_nvpm_number_sls = np.interp(F_gr_F_rated, thrust_setting_icao, EI_number_icao_sl)

        # ei_nvpm_mass_sls = (-1.4110*t**6) + (-5.3007*t**5) - (3.5961*t**4) + (9.2888*t**3) + (23.6098*t**2) + (13.9142*t) + 2.9213
        # print(ei_nvpm_mass_sls)
        ei_nvpm_mass = ei_nvpm_mass_sls * (PT3_inflight/pt3_sls)**1.35*(FAR_inflight/far_sls)**2.5
        # ei_nvpm_mass = ei_nvpm_mass_sls * (PT3_inflight / pt3_sls) ** 1.35 * (1.1) ** 2.5

        result = ei_nvpm_mass * ei_nvpm_number_sls
        print(result)
    # elif saf == 20:
    #     ei_nvpm_mass_sls = (-0.5444 * t ** 6) + (-4.4315 * t ** 5) - (5.3065 * t ** 4) + (8.8020 * t ** 3) + (
    #                 23.0131 * t ** 2) + (12.3004 * t) + 2.2554
    #     # print(ei_nvpm_mass_sls)
    #     ei_nvpm_mass = ei_nvpm_mass_sls * (PT3_inflight / pt3_sls) ** 1.35 * (FAR_inflight / far_sls) ** 2.5
    #     # print(ei_nvpm_mass)
    #     v = (-3.9951 * np.exp(1) + 12.2380) * np.exp(-1.2642 * t) * 10 ** 14
    #     result = v * ei_nvpm_mass
    # elif saf == 100:
    #     ei_nvpm_mass_sls = (0.3255 * t ** 6) + (-3.6254 * t ** 5) - (6.9934 * t ** 4) + (8.2737 * t ** 3) + (
    #             22.5514 * t ** 2) + (10.9926 * t) + 1.7097
    #     # print(ei_nvpm_mass_sls)
    #     ei_nvpm_mass = ei_nvpm_mass_sls * (PT3_inflight / pt3_sls) ** 1.35 * (FAR_inflight / far_sls) ** 2.5
    #     # print(ei_nvpm_mass)
    #     v = (-3.9663 * np.exp(1) + 12.2486) * np.exp(-1.2962 * t) * 10 ** 14
    #     result = v * ei_nvpm_mass
    return result