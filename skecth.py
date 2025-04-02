from emission_index import p3t3_nox, p3t3_nvpm_meem, p3t3_nvpm_meem_mass, thrust_setting
import pickle
# df_gsp['ei_nox_p3t3'] = df_gsp.apply(
#     lambda row: p3t3_nox(
#         row['PT3'],
#         row['TT3'],
#         interp_func_far,
#         interp_func_pt3,
#         row['specific_humidity'],
#         row['WAR_gsp'],
#         engine_model
#     ),
#     axis=1
# )
engine_model = 'GTF2035'

if engine_model in ('GTF'):
    with open('p3t3_graphs_sls_gtf_final.pkl', 'rb') as f:
        loaded_functions = pickle.load(f)
elif engine_model in ('GTF2035', 'GTF2035_wi'):
    with open('p3t3_graphs_sls_gtf2035_final.pkl', 'rb') as f:
        loaded_functions = pickle.load(f)
elif engine_model in ('GTF1990', 'GTF2000'):
    with open('p3t3_graphs_sls_1990_2000_final.pkl', 'rb') as f:
        loaded_functions = pickle.load(f)
else:
    raise ValueError(f"Unsupported engine_model: {engine_model}.")

interp_func_far = loaded_functions['interp_func_far']
interp_func_pt3 = loaded_functions['interp_func_pt3']
interp_func_fgr = loaded_functions['interp_func_fgr']


pt3 = 4.23447
tt3 = 498.62
far = 0.016153

WAR_gsp = 0
specific_humidity = 0

ei_nox_toc = p3t3_nox(pt3,tt3, interp_func_far, interp_func_pt3, specific_humidity, WAR_gsp, engine_model)
print(ei_nox_toc)

# df_gsp['ei_nvpm_number_p3t3_meem'] = df_gsp.apply(
#         lambda row: p3t3_nvpm_meem(
#             row['PT3'],
#             row['TT3'],
#             row['FAR'],
#             interp_func_far,
#             interp_func_pt3,
#             row['SAF'],
#             row['thrust_setting_meem'],
#             engine_model
#         ),
#         axis=1
#     )
#
#     df_gsp['ei_nvpm_mass_p3t3_meem'] = df_gsp.apply(
#         lambda row: p3t3_nvpm_meem_mass(
#             row['PT3'],
#             row['TT3'],
#             row['FAR'],
#             interp_func_far,
#             interp_func_pt3,
#             row['SAF'],
#             row['thrust_setting_meem'],
#             engine_model
#         ),
#         axis=1
#     )
# df_gsp['thrust_setting_meem'] = df_gsp.apply(
#         lambda row: thrust_setting(
#             engine_model,
#             row['TT3'],
#             interp_func_pt3,
#             interp_func_fgr
#         ),
#         axis=1
#     )
SAF = 0
thrust_setting_toc = thrust_setting(engine_model, tt3, interp_func_pt3, interp_func_fgr)
ei_mass_toc = p3t3_nvpm_meem_mass(pt3, tt3, far, interp_func_far, interp_func_pt3, SAF, thrust_setting_toc, engine_model)
ei_number_toc = p3t3_nvpm_meem(pt3, tt3, far, interp_func_far, interp_func_pt3, SAF, thrust_setting_toc, engine_model)

print(ei_mass_toc)
print(ei_number_toc)

#check t/o
#835,38	32,17344	2,74E-02

# pt3 = 32.17344
# tt3 = 835.38
# far = 0.027407
# thrust_setting_to = thrust_setting(engine_model, tt3, interp_func_pt3, interp_func_fgr)
# ei_mass_to = p3t3_nvpm_meem_mass(pt3, tt3, far, interp_func_far, interp_func_pt3, SAF, thrust_setting_to, engine_model)
# ei_number_to = p3t3_nvpm_meem(pt3, tt3, far, interp_func_far, interp_func_pt3, SAF, thrust_setting_to, engine_model)
# print(thrust_setting_to)
# print(ei_mass_to)
# print(ei_number_to)
