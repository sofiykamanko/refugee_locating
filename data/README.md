# Data

## raw/

| File | Content | Source |
|---|---|---|
| `destatis_12521-0021_foreigners_by_state_nationality_2025.csv` | Foreigners by state and nationality (209 nationalities), 31 Dec 2025 | Destatis GENESIS-Online, table 12521-0021 (free registration required to download) |
| `destatis_counts_selected_2025.csv` | The eight origins used in the generator, all foreigners, and the average count per other nationality | Derived from the file above |
| `ba_employment_asylum_countries_by_state.xlsx` | Employment rate of nationals of the eight main non-European asylum countries, by state, monthly | Bundesagentur für Arbeit, Statistik, "Beschäftigte aus den nichteuropäischen Asylherkunftsländern" |

Other inputs are hard-coded in `src/refsim/generator.py` with their source next to each value:

- attribute distributions: Strasser Ceballos & Kern (2025), Table 5 (IAB-BAMF-SOEP sample)
- state quotas: BAMF, "Erstverteilung" (Königsteiner Schlüssel)
- employment effects of education and German language: IAB-Forum (2025), "10 Jahre Fluchtmigration 2015"
- women/men employment ratio: IAB (2015 arrival cohort, 2022)

## synthetic/

`synthetic_refugees.csv`: 5,000 people, seed 42, no community effect. Recreate with
`python scripts/generate_data.py --n 5000 --seed 42`. Other generated files are not committed.
