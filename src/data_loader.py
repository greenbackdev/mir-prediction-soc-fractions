import pandas as pd
import sqlite3


class DataLoader:

    def __init__(self):
        pass

    @staticmethod
    def _str2dict(spectrum):
        return dict(subString.split(":") for subString in spectrum[1:-1].replace('Unnamed:', 'Unnamed').replace('"', '').replace(' ', '').split(","))

    def run(
        self,
        data_source_name,
        filter_re_yield=True,
        filter_layer_upper_limit=True
    ):
        conn = sqlite3.connect('db.sqlite3')
        cur = conn.cursor()

        columns = [
            'sample_id',
            'layer_upper_limit',
            'layer_lower_limit',
            'id_site',
            'lat',
            'lon',
            'land_cover',
            'toc',
            'clay',
            'sand',
            'silt',
            'ph',
            'caco3',
            'tocre6',
            'stablecarbon_perc',
            'yield_rock_eval',
            'mir_spectrum',
        ]

        query = f"""SELECT s.id sample_id, s.layer_upper_limit, s.layer_lower_limit, site.id, site.lat, site.lon, land_cover, toc, clay, sand, silt, ph, caco3, re.tocre6, stable_carbon_partysoc, yield_rock_eval, mir_spectrum
            FROM measure m
            INNER JOIN sample s ON m.sample_id = s.id
            INNER JOIN site ON s.site_id = site.id
            INNER JOIN data_source ds ON s.data_source_id = ds.id
            INNER JOIN rock_eval_analysis re ON re.sample_id = s.id
            INNER JOIN ir_absorbance ir ON ir.sample_id = s.id
            WHERE ds.name = '{data_source_name}' """

        cur.execute(query)
        sample_data = pd.DataFrame(cur.fetchall(), columns=columns)

        sample_data['stablecarbon_frac'] = sample_data.stablecarbon_perc / 100
        sample_data['activecarbon_frac'] = 1 - sample_data.stablecarbon_frac
        sample_data['stablecarbon_qty'] = sample_data.stablecarbon_frac * \
            sample_data.toc
        sample_data['activecarbon_qty'] = (
            1 - sample_data.stablecarbon_frac) * sample_data.toc

        sample_data.id_site = sample_data.id_site.astype(int)

        # Remove points without MIR spectra
        sample_data = sample_data[(~sample_data.mir_spectrum.isna())]

        print(f'Loaded {len(sample_data)} spectra.')

        # select mineral soils
        sample_data = sample_data[sample_data.toc <= 120]
        print(f'{len(sample_data)} spectra after removing organic soils.')

        if filter_re_yield:
            # select samples with good RE yield
            sample_data = sample_data[
                (sample_data.yield_rock_eval.between(0.7, 1.3))
            ]
            print(
                f'{len(sample_data)} spectra after removing samples with RE yield < 0.7 or > 1.3.')

        if filter_layer_upper_limit:
            # select topsoil samples
            sample_data = sample_data[
                (sample_data.layer_upper_limit <= 30)
            ]
            print(
                f'{len(sample_data)} spectra after removing samples with layer_upper_limit > 30 cm.')

        # Treat spectra
        mir_spectra = pd.concat(
            [
                pd.DataFrame.from_dict(self._str2dict(
                    spectrum), orient='index', columns=[i]).T
                for i, spectrum in sample_data.mir_spectrum.items()
                if spectrum is not None
            ]
        )
        mir_spectra.drop(['Unnamed0', 'ID'], axis=1,
                         errors='ignore', inplace=True)
        mir_spectra = mir_spectra.astype(float)

        wavelenghts_MIR = [float(w) for w in mir_spectra.columns]

        return sample_data, wavelenghts_MIR, mir_spectra
