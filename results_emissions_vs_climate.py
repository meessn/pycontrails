import pandas as pd
import matplotlib.pyplot as plt

# Load data
emissions_df = pd.read_csv("results_report/emissions/all_emissions_changes_vs_GTF1990.csv")
climate_with_df = pd.read_csv("results_report/climate/contrail_yes_rad_vs_gtf1990.csv")
climate_without_df = pd.read_csv("results_report/climate/contrail_no_rad_vs_gtf1990.csv")


# Map engine names
def map_engine_names(row):
    if row['engine'] == 'GTF1990':
        return 'CFM1990'
    elif row['engine'] == 'GTF2000':
        return 'CFM2008'
    elif row['engine'] == 'GTF2035':
        suffix = '' if row['saf_level'] == 0 else f' - {row["saf_level"]}'
        return f'GTF2035{suffix}'
    elif row['engine'] == 'GTF2035_wi':
        suffix = '' if row['saf_level'] == 0 else f' - {row["saf_level"]}'
        return f'GTF2035WI{suffix}'
    else:
        return row['engine']

emissions_df['mapped_engine'] = emissions_df.apply(map_engine_names, axis=1)
emissions_df.set_index('mapped_engine', inplace=True)
climate_with_df.set_index('engine_display', inplace=True)
climate_without_df.set_index('engine_display', inplace=True)

# Define desired engine order
engine_order = ['CFM1990', 'CFM2008', 'GTF', 'GTF2035', 'GTF2035 - 20', 'GTF2035 - 100',
                'GTF2035WI', 'GTF2035WI - 20', 'GTF2035WI - 100']

# Filter and reorder
common_engines = [e for e in engine_order if e in emissions_df.index and e in climate_with_df.index and e in climate_without_df.index]

# Normalize by CFM1990 (i.e., add 100 to all % changes)
def normalize(df):
    return df.loc[common_engines] + 100

# CO2 Plot Data
co2_df = pd.DataFrame({
    'CO$_2$ Emission': normalize(emissions_df['co2_conservative_sum_change']),
    'CO$_2$ Climate Contrails': normalize(climate_with_df['CO2 Conservative']),
    'CO$_2$ Climate No Contrails': normalize(climate_without_df['CO2 Conservative']),
})

# Plot CO2
co2_df.plot(kind='bar', figsize=(10, 6), alpha=0.7)
plt.title("Emissions vs Climate Impact Reduction: CO$_2$")
plt.ylabel("Relative Index (%)")
# plt.xticks(rotation=45, ha='right')
plt.xticks(
    ticks=range(len(common_engines)),
    labels=[
        "CFM1990", "CFM2008", "GTF",
        "GTF2035", "GTF2035\n20", "GTF2035\n100",
        "GTF2035WI", "GTF2035WI\n20", "GTF2035WI\n100"
    ],
    rotation=0,
    ha='center'
)
plt.tight_layout()
plt.savefig('results_report/emissions_vs_climate/emissions_climate_co2.png', format='png')
plt.show()

# NOx Plot Data
nox_df = pd.DataFrame({
    'NOx Emission': normalize(emissions_df['nox_sum_change']),
    'NOx Climate Contrails': normalize(climate_with_df['NOx']),
    'NOx Climate No Contrails': normalize(climate_without_df['NOx']),
})

# Plot NOx
nox_df.plot(kind='bar', figsize=(10, 6), alpha=0.7)
plt.title("Emissions vs Climate Impact Reduction: NOx")
plt.ylabel("Relative Index (%)")
# plt.xticks(rotation=45, ha='right')
plt.xticks(
    ticks=range(len(common_engines)),
    labels=[
        "CFM1990", "CFM2008", "GTF",
        "GTF2035", "GTF2035\n20", "GTF2035\n100",
        "GTF2035WI", "GTF2035WI\n20", "GTF2035WI\n100"
    ],
    rotation=0,
    ha='center'
)
plt.tight_layout()
plt.savefig('results_report/emissions_vs_climate/emissions_climate_nox.png', format='png')
plt.show()

# Third Plot Data: Fuel, nvPM, Contrail
third_df = pd.DataFrame({
    'Fuel Flow': normalize(emissions_df['fuel_kg_sum_change']),
    'nvPM Number': normalize(emissions_df['nvpm_num_sum_change']),
    'Contrail Climate': normalize(
        climate_with_df['Contrail']
    ),
})

# Plot Third
third_df.plot(kind='bar', figsize=(10, 6), alpha=0.7)
plt.title("Emissions vs Climate Impact Reduction: Contrails")
plt.ylabel("Relative Index (%)")
# plt.xticks(rotation=45, ha='right')
plt.xticks(
    ticks=range(len(common_engines)),
    labels=[
        "CFM1990", "CFM2008", "GTF",
        "GTF2035", "GTF2035\n20", "GTF2035\n100",
        "GTF2035WI", "GTF2035WI\n20", "GTF2035WI\n100"
    ],
    rotation=0,
    ha='center'
)
plt.tight_layout()
plt.savefig('results_report/emissions_vs_climate/emissions_climate_contrails.png', format='png')
plt.show()

non_co2_df = pd.DataFrame({
    'Contrails': normalize(climate_with_df['Non-CO2']),
    'No Contrails': normalize(climate_without_df['Non-CO2']),
}).loc[common_engines]
non_co2_df.index.name = None
# Plot
non_co2_df.plot(kind='bar', figsize=(10, 6), alpha=0.7)
plt.title("Non-CO$_2$ Climate Impact")
plt.ylabel("Relative Index (%)")
plt.xticks(
    ticks=range(len(common_engines)),
    labels=[
        "CFM1990", "CFM2008", "GTF",
        "GTF2035", "GTF2035\n20", "GTF2035\n100",
        "GTF2035WI", "GTF2035WI\n20", "GTF2035WI\n100"
    ],
    rotation=0,
    ha='center'
)
plt.tight_layout()
plt.savefig('results_report/emissions_vs_climate/emissions_climate_non_co2.png', format='png')
plt.show()


total_df = pd.DataFrame({
    'Contrails': normalize(climate_with_df['Total Climate Impact Conservative']),
    'No Contrails': normalize(climate_without_df['Total Climate Impact Conservative']),
}).loc[common_engines]
total_df.index.name = None
# Plot
total_df.plot(kind='bar', figsize=(10, 6), alpha=0.7)
plt.title("Total Climate Impact (Conservative)")
plt.ylabel("Relative Index (%)")
plt.xticks(
    ticks=range(len(common_engines)),
    labels=[
        "CFM1990", "CFM2008", "GTF",
        "GTF2035", "GTF2035\n20", "GTF2035\n100",
        "GTF2035WI", "GTF2035WI\n20", "GTF2035WI\n100"
    ],
    rotation=0,
    ha='center'
)
plt.tight_layout()
plt.savefig('results_report/emissions_vs_climate/emissions_climate_total_cons.png', format='png')
plt.show()


