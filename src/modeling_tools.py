import json
import pandas as pd
import numpy as np
import os
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cross_decomposition import PLSRegression
from statsmodels.othermod.betareg import BetaModel
from statsmodels.tools import add_constant
from sklearn.model_selection import cross_validate, RepeatedKFold, ParameterGrid, train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error, r2_score, make_scorer
from scipy import stats


class AbsorbanceSpectraPreProcesser:
    """"
    Preprocess infrared absorbance spectra.

    Parameters:
    - df: dataframe of absorbance spectra.
    - shift position: the wavelength at which the spectrometer changed (can be None).

    The absorbance spectra should be provided as a Pandas DataFrame of
    absorbance intensity values with:
        - index: soil identifier, named `id`.
        - columns: one column per band; the column name is the wavelength expressed in [nm] and with precision of one digit.

    The pre-prossing procedure is inspired from:
        Stevens A, Nocita M, Tóth G, Montanarella L, van Wesemael B (2013)
        Prediction of Soil Organic Carbon at the European Scale by Visible
        and Near InfraRed Reflectance Spectroscopy.
        PLoS ONE 8(6): e66409. doi:10.1371/journal.pone.0066409
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def _correct_shift(df, shift_position_right):

        shift_position_right = float(shift_position_right)
        shift_position_index = list(df.columns).index(shift_position_right)
        shift_position_left = df.columns[shift_position_index - 1]

        # oberved gap between points at the shift position
        gap = df[shift_position_right] - df[shift_position_left]

        # slope of the spectrum before the shift
        slope_left = (df[shift_position_left] - df[df.columns[shift_position_index - 2]]
                      ) / (shift_position_left - df.columns[shift_position_index - 2])
        # slope of the spectrum after the shift
        slope_right = (df[df.columns[shift_position_index + 1]] - df[shift_position_right]
                       ) / (df.columns[shift_position_index + 1] - shift_position_right)
        # average slope around the shift
        slope_avg = (slope_left + slope_right) / 2

        # actual shift: gap minus the expected slope
        shift = gap - \
            (slope_avg * (shift_position_right - shift_position_left))

        # shift all points after the shift position
        columns_to_shift = [c for c in df.columns if float(
            c) >= shift_position_right]
        df[columns_to_shift] = df[columns_to_shift].apply(
            lambda x: x - shift.loc[x.index], axis=0)

        return df

    @staticmethod
    def _standard_normal_variate(x):
        mean = x.mean()
        std = x.std()
        return (x - mean) / std

    @staticmethod
    def _savgol_nonuniform(x, y, window_length, polyorder, deriv=0):
        """
        Savitzky-Golay smoothing 1D filter

        Implements the method described in
        https://dsp.stackexchange.com/questions/1676/savitzky-golay-smoothing-filter-for-not-equally-spaced-data
        free to use at the user's risk

        :param x:
        :param y:
        :param window_length: the smoothing sample, e.g. window_length=2 for smoothing over 5 points
        :param polyorder: the degree of the local polynomial fit, e.g. polyorder=2 for a parabolic fit
        :param deriv: The order of the derivative to compute. This must be a nonnegative integer.
                The default is 0, which means to filter the data without differentiating.
        :return:
        """
        if type(x) is not np.array:
            x = np.array(x)
        if type(y) is not np.array:
            y = np.array(y)

        n = int((window_length - 1) / 2)

        if x.shape != y.shape:
            raise RuntimeError("x and y arrays are of different shape")
        if x.shape[0] < window_length:
            raise RuntimeError(
                "not enough data to start the smoothing process")
        if 2 * n + 1 <= polyorder + 1:
            raise RuntimeError(
                "need at least deg+1 points to make the polynomial")

        # smooth start and end data
        ysm = np.zeros(y.shape)
        for i in range(n):
            j = y.shape[0] - i - 1
            if deriv == 0:
                ysm[i] = y[i]
                ysm[j] = y[j]
            if deriv == 1:
                ysm[i] = (y[i] - y[i - 1]) / (x[i] - x[i - 1])
                ysm[j] = (y[j] - y[j - 1]) / (x[j] - x[j - 1])
            if deriv == 2:
                ysm[i] = (((y[i] - y[i - 1]) / (x[i] - x[i - 1])) - ((y[i - 1] - y[i - 2]) / (x[i - 1] - x[i - 2]))) / \
                    (x[i] - x[i - 1])
                ysm[j] = (((y[j] - y[j - 1]) / (x[j] - x[j - 1])) - ((y[j - 1] - y[j - 2]) / (x[j - 1] - x[j - 2]))) / \
                    (x[j] - x[j - 1])
            if deriv >= 3:
                raise NotImplementedError("derivatives >= 3 not implemented")

        m = 2 * n + 1  # the size of the filter window
        o = polyorder + 1  # the smoothing order
        A = np.zeros((m, o))  # A matrix
        t = np.zeros(m)
        # start smoothing
        for i in range(n, x.shape[0] - n):
            for j in range(m):
                t[j] = x[i + j - n] - x[i]
            for j in range(m):
                r = 1.0
                for k in range(o):
                    A[j, k] = r
                    r *= t[j]
            tA = A.transpose()  # A transposed
            tAA = np.matmul(tA, A)  # make tA.A
            tAA = np.linalg.inv(tAA)  # make (tA.A)-¹ in place
            tAAtA = np.matmul(tAA, tA)  # make (tA.A)-¹.tA

            # compute the polynomial's value at the center of the sample
            ysm[i] = 0.0
            for j in range(m):
                ysm[i] += tAAtA[deriv, j] * y[i + j - n]

        return ysm

    def run(self,
            df: pd.DataFrame,
            shift_position=None,
            min_wavelength=500,
            max_wavelength=10000,
            window_length=51,
            polyorder=3,
            deriv=1,
            resampling_step=10
            ) -> dict:
        """
        - Correct for the shift in absorbance at the splice
           of the two detectors.
        - Remove beginning and end according to `min_wavelength` and `max_wavelength`.
        - Savitzky-Golay (SG) smoothing and derivative (parameters:
        `window_length`, `polyorder`, `deriv`)
        - Standard Normal Variate (SNV) transformation on SG-filtered
           spectral data.
        - Linear interpolation at 1nm resolution.
        - Subsampling: keep only one band every `resampling_step` nm.
        """

        df.columns = [float(c) for c in df.columns]

        # Correct for the shift in absorbance at the splice
        #    of the two detectors.
        if shift_position is not None:
            df = self._correct_shift(df, shift_position)

        # Remove beginning and end.
        if min_wavelength is None:
            min_wavelength = int(np.floor(min([float(c) for c in df.columns])))
        if max_wavelength is None:
            max_wavelength = int(np.ceil(max([float(c) for c in df.columns])))

        df = df[
            [
                float(c)
                for c in df.columns
                if float(c) >= min_wavelength and float(c) <= max_wavelength
            ]
        ]

        # SG nonuniform filter

        df = pd.DataFrame([self._savgol_nonuniform(
            y=list(df.iloc[i, :]),
            x=df.columns.values.astype(float),
            window_length=window_length,
            polyorder=polyorder,
            deriv=deriv)
            for i in range(len(df))],
            columns=df.columns,
            index=df.index)

        # Standard Normal Variate
        df = df.apply(self._standard_normal_variate, axis=1)

        # Linear interpolation at 1nm resolution
        columns = range(int(min(df.columns)), int(max(df.columns)) + 1)

        columns_to_add = [float(c)
                          for c in columns if float(c) not in df.columns]

        df_empty = pd.DataFrame(
            [
                [np.nan for i in range(len(columns_to_add))]
                for j in df.index
            ],
            columns=columns_to_add,
            index=df.index
        )
        df = pd.concat([df, df_empty], axis=1)

        df = df[[c for c in sorted(df.columns)]]
        df = df.interpolate(method='linear', axis=1, limit_direction='both')

        # Subsampling
        wavelengths = [int(w) for w in range(
            min_wavelength,
            max_wavelength + resampling_step,
            resampling_step
        )
            if float(w) in df.columns]
        df = df[wavelengths]

        return df


def rmspe(y_true, y_pred):
    return np.sqrt(np.mean(np.square(((y_true - y_pred) / y_true)), axis=0))


def rpd(y_true, y_pred):
    return np.std(y_true) / root_mean_squared_error(y_true, y_pred)


def rpiq(y_true, y_pred):
    return stats.iqr(y_true) / root_mean_squared_error(y_true, y_pred)


def bias_function(y_true, y_pred):
    return y_pred.mean() - y_true.mean()


default_params = {
    'shift_position': None,
    'min_wavelength': 440,    # checked visually where the signal became less noisy
    'max_wavelength': 10000,
    'window_length': 5,
    'polyorder': 2,
    'deriv': 1,
    'resampling_step': 2,
    'n_components': 10,
}


def get_predictions_config(X, df):

    predictions_config = {
        'toc_pls': {
            'feature': 'toc',
            'name': 'TOC',
            'unit': 'gC/kg',
            'X': X,
            'data': df,
            'model_name': 'pls'
        },
        'stablecarbon_frac_pls': {
            'feature': 'stablecarbon_frac',
            'name': 'Stable carbon fraction',
            'unit': 'unitless',
            'X': X,
            'data': df,
            'model_name': 'pls'
        },
        'stablecarbon_frac_beta_reg': {
            'feature': 'stablecarbon_frac',
            'name': 'Stable carbon fraction',
            'unit': 'unitless',
            'X': X,
            'data': df,
            'model_name': 'beta-regression'
        },
        'stablecarbon_qty_pls': {
            'feature': 'stablecarbon_qty',
            'name': 'Stable carbon quantity',
            'unit': 'gC/kg',
            'X': X,
            'data': df,
            'model_name': 'pls'
        },
        'stablecarbon_qty_beta_reg': {
            'feature': 'stablecarbon_qty',
            'name': 'Stable carbon quantity',
            'unit': 'gC/kg',
            'X': X,
            'data': df,
            'model_name': 'beta-regression'
        },
        'activecarbon_qty_pls': {
            'feature': 'activecarbon_qty',
            'name': 'Active carbon quantity',
            'unit': 'gC/kg',
            'X': X,
            'data': df,
            'model_name': 'pls'
        },
        'activecarbon_qty_beta_reg': {
            'feature': 'activecarbon_qty',
            'name': 'Active carbon quantity',
            'unit': 'gC/kg',
            'X': X,
            'data': df,
            'model_name': 'beta-regression'
        },
        'activecarbon_frac_beta_reg': {
            'feature': 'activecarbon_frac',
            'name': 'Active carbon fraction',
            'unit': 'unitless',
            'X': X,
            'data': df,
            'model_name': 'beta-regression'
        },
    }

    return predictions_config


class SMWrapper(BaseEstimator, RegressorMixin):
    """
    A universal sklearn-style wrapper for statsmodels regressors
    https://stackoverflow.com/questions/41045752/using-statsmodel-estimations-with-scikit-learn-cross-validation-is-it-possible
    """

    def __init__(self, model_class, fit_intercept=True):
        self.model_class = model_class
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        if self.fit_intercept:
            X = add_constant(X)
        self.model_ = self.model_class(y, X)
        self.results_ = self.model_.fit()
        return self

    def predict(self, X):
        if self.fit_intercept:
            X = add_constant(X)
        return self.results_.predict(X)


class Prediction:

    def __init__(
        self,
        name,
        feature,
        unit,
        X,
        data,
        model_name,
        train_and_test=True,
        n_components=None,
        save=False,
        output_folder=None,
        wavelengths=None
    ):
        self.name = name
        self.feature = feature
        self.unit = unit
        self.wavelengths = wavelengths
        self.X = X.values
        if len(data[feature]) == 0:
            raise (Exception, f"No data for ${feature}.")
        self.y = data[feature].values
        if train_and_test:
            (
                X_train,
                X_test,
                y_train,
                y_test,
                indices_train,
                indices_test
            ) = self._train_test_split(self.X, self.y)
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test
            self.indices_train = indices_train
            self.indices_test = indices_test
        if model_name not in ['pls', 'beta-regression']:
            raise (
                Exception, "Possible model_name values are 'pls' and 'beta-regression'")
        self.model_name = model_name
        self.n_components = n_components
        self.save = save
        self.output_folder = output_folder
        if save and output_folder is None:
            raise (
                Exception, "Must provide a value for `output_folder` when `save=True`.")
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)

    def set_n_components(self, n):
        self.n_components = n

    def set_pretrained_pls(self, model):
        self.pls = model

    def set_pretrained_beta_regression(self, model):
        self.beta_regression = model

    @staticmethod
    def _train_test_split(
        X,
        y,
        return_indices=True,
        test_size=0.2,
        algorithm='random',
        random_state=1,
        how='X',
        distance_metric='euclidean'
    ):
        """
        Split train and test set.
        - X: input array
        - y: target array
        - return indices: if True, return X_train, X_test, y_train, y_test, indices_train, indices_test
        - test_size: fraction
        - algorithm: 'random' or 'kennard_stone'
        - random_state: ingnored if method is not 'random'
        - how: 'X' or 'y'. Ignored if method is not 'kennard_stone'
        - distance_metric: Ignored if method is not 'kennard_stone'.

        """

        if algorithm not in ['random', 'kennard_stone']:
            raise (
                ValueError, "The only admitted values for `algorithm` are 'random' and `kennard_stone`.")

        if how not in ['X', 'y']:
            raise (ValueError, "The only admitted values for `how` are 'X' and `y`.")

        y_with_indices = np.array([[i, v] for i, v in enumerate(y)])

        if algorithm == 'kennard_stone':
            if how == 'X':
                # Kennard-Stone algorithm, selecting dividing data based on spectra.
                # --> Provides same test set for all targets but we should check that
                # the target variable distribution is similar in the test and training sets.

                (
                    X_train,
                    X_test,
                    y_train_with_indices,
                    y_test_with_indices,
                ) = astartes.train_test_split(
                    X,
                    y_with_indices,
                    test_size=test_size,
                    sampler='kennard_stone',
                    hopts={'metric': distance_metric}
                )

            else:
                # Kennard-Stone algorithm, selecting dividing data based on target variable.
                (
                    y_train_with_indices,
                    y_test_with_indices,
                    X_train,
                    X_test
                ) = astartes.train_test_split(
                    y_with_indices,
                    X,
                    test_size=test_size,
                    sampler='kennard_stone',
                    hopts={'metric': distance_metric}
                )
        else:
            (
                X_train,
                X_test,
                y_train_with_indices,
                y_test_with_indices,
            ) = train_test_split(
                X,
                y_with_indices,
                test_size=test_size,
                random_state=random_state
            )

        if return_indices:
            indices_train = [
                int(v) for v in y_train_with_indices[:, 0].reshape(1, -1).flatten()]
            y_train = np.array(y_train_with_indices)[:, 1]
            indices_test = [
                int(v) for v in y_test_with_indices[:, 0].reshape(1, -1).flatten()]
            y_test = np.array(y_test_with_indices)[:, 1]
            return X_train, X_test, y_train, y_test, indices_train, indices_test

        return X_train, X_test, y_train, y_test

    def _run_dimensionality_reduction(self, n_components):
        pls = PLSRegression(n_components=n_components).fit(
            self.X_train, self.y_train)
        X_transformed = pls.transform(self.X)
        X_train_transformed = pls.transform(self.X_train)
        X_test_transformed = pls.transform(self.X_test)

        self.X_tranformed = X_transformed
        self.X_train_transformed = X_train_transformed
        self.X_test_transformed = X_test_transformed

        return (pls, X_transformed, X_train_transformed, X_test_transformed)

    @staticmethod
    def _get_prediction_metrics(y, y_pred):

        scores = {
            'R2': r2_score(y, y_pred),
            'RMSE': root_mean_squared_error(y, y_pred),
            'MAPE': mean_absolute_percentage_error(y, y_pred),
            'RPD': rpd(y, y_pred),
            'RPIQ': rpiq(y, y_pred),
            'Bias': bias_function(y, y_pred)
        }

        return scores

    @staticmethod
    def _run_beta_regression_prediction(beta_regression, X):
        prediction_frame = beta_regression.get_prediction(
            add_constant(X)).summary_frame()
        y_pred = prediction_frame['predicted'].values
        y_pred_ci_lower = prediction_frame['ci_lower'].values
        y_pred_ci_upper = prediction_frame['ci_upper'].values
        return y_pred, y_pred_ci_lower, y_pred_ci_upper

    def cross_validate(self, n_splits=5, n_repeats=5, random_state=1, full_scores=False):

        cv = RepeatedKFold(n_splits=n_splits,
                           n_repeats=n_repeats, random_state=random_state)

        if full_scores:
            scoring = {
                'r2': 'r2',
                'neg_mean_squared_error': 'neg_mean_squared_error',
                'mape': make_scorer(mean_absolute_percentage_error, greater_is_better=False),
                'rpd': make_scorer(rpd),
                'rpiq': make_scorer(rpiq),
                'bias': make_scorer(bias_function, greater_is_better=False),
            }
        else:
            scoring = ('r2', 'neg_mean_squared_error')

        if self.model_name == 'pls':
            pls = PLSRegression(n_components=self.n_components)
            scores = cross_validate(
                pls,
                self.X,
                self.y,
                cv=cv,
                scoring=scoring,
                return_train_score=True,
                n_jobs=-1
            )

        else:
            pls, X_transformed, _, _ = self._run_dimensionality_reduction(
                n_components=self.n_components
            )

            beta_regression = SMWrapper(BetaModel)

            scores = cross_validate(
                beta_regression,
                X_transformed,
                self.y,
                cv=cv,
                scoring=scoring,
                return_train_score=True,
                n_jobs=-1,
                error_score='raise'
            )

        return scores

    def fit(self, X, y):

        pls = PLSRegression(n_components=self.n_components).fit(X, y)
        self.pls = pls

        if self.model_name == 'pls':
            return pls

        X_transformed = pls.transform(X)

        beta_regression = BetaModel(y, add_constant(X_transformed)).fit()
        self.beta_regression = beta_regression
        return beta_regression

    def predict(self, X):

        if self.model_name == 'pls':
            y_pred = self.pls.predict(X)
            return y_pred, None, None

        X_transformed = self.pls.transform(X)
        (
            y_pred,
            y_pred_ci_lower,
            y_pred_ci_upper
        ) = self._run_beta_regression_prediction(
            self.beta_regression,
            X_transformed
        )
        return y_pred, y_pred_ci_lower, y_pred_ci_upper

    def run_cross_validation(self, n_splits=5, n_repeats=20):

        scores = self.cross_validate(
            n_splits=n_splits, n_repeats=n_repeats, full_scores=True)

        train_r2 = scores['train_r2']
        train_rmse = np.sqrt(-scores['train_neg_mean_squared_error'])
        train_mape = scores['train_mape']
        train_rpd = scores['train_rpd']
        train_rpiq = scores['train_rpiq']
        train_bias = scores['train_bias']

        test_r2 = scores['test_r2']
        test_rmse = np.sqrt(-scores['test_neg_mean_squared_error'])
        test_mape = scores['test_mape']
        test_rpd = scores['test_rpd']
        test_rpiq = scores['test_rpiq']
        test_bias = scores['test_bias']

        cross_validation_results = {
            'N': len(self.y),
            'CV-cal_R2': f'{train_r2.mean():.2f} ± {train_r2.std():.2f}',
            'CV-cal_RMSE': f'{train_rmse.mean():.2f} ± {train_rmse.std():.2f}',
            'CV-cal_MAPE': f'{train_mape.mean():.2f} ± {train_mape.std():.2f}',
            'CV-cal_RPD': f'{train_rpd.mean():.2f} ± {train_rpd.std():.2f}',
            'CV-cal_RPIQ': f'{train_rpiq.mean():.2f} ± {train_rpiq.std():.2f}',
            'CV-cal_bias': f'{train_bias.mean():.2f} ± {train_bias.std():.2f}',
            'CV-val_R2': f'{test_r2.mean():.2f} ± {test_r2.std():.2f}',
            'CV-val_RMSE': f'{test_rmse.mean():.2f} ± {test_rmse.std():.2f}',
            'CV-val_MAPE': f'{test_mape.mean():.2f} ± {test_mape.std():.2f}',
            'CV-val_RPD': f'{test_rpd.mean():.2f} ± {test_rpd.std():.2f}',
            'CV-val_RPIQ': f'{test_rpiq.mean():.2f} ± {test_rpiq.std():.2f}',
            'CV-val_bias': f'{test_bias.mean():.2f} ± {test_bias.std():.2f}',
        }

        if self.save:
            with open(os.path.join(self.output_folder, 'cross_validation_results.json'), "w") as f:
                json.dump(cross_validation_results, f, indent=4)

        print('Cross validation results (on whole dataset):')
        print(
            f"CV-cal: R2 = {cross_validation_results['CV-cal_R2']}, RMSE = {cross_validation_results['CV-cal_RMSE']}")
        print(
            f"CV-val: R2 = {cross_validation_results['CV-val_R2']}, RMSE = {cross_validation_results['CV-val_RMSE']}")

    def run_train(self):

        _ = self.fit(self.X_train, self.y_train)

    def run_train_test_prediction(self):

        _ = self.fit(self.X_train, self.y_train)
        y_train_pred, y_train_pred_ci_lower, y_train_pred_ci_upper = self.predict(
            self.X_train)
        y_test_pred, y_test_pred_ci_lower, y_test_pred_ci_upper = self.predict(
            self.X_test)
        y_pred, y_pred_ci_lower, y_pred_ci_upper = self.predict(self.X)

        self.y_train_pred = y_train_pred
        self.y_train_pred_ci_lower = y_train_pred_ci_lower
        self.y_train_pred_ci_upper = y_train_pred_ci_upper
        self.y_test_pred = y_test_pred
        self.y_test_pred_ci_lower = y_test_pred_ci_lower
        self.y_test_pred_ci_upper = y_test_pred_ci_upper
        self.y_pred = y_pred
        self.y_pred_ci_lower = y_pred_ci_lower
        self.y_pred_ci_upper = y_pred_ci_upper

    def run_prediction(self):

        y_pred, y_pred_ci_lower, y_pred_ci_upper = self.predict(self.X)
        self.y_pred = y_pred
        self.y_pred_ci_lower = y_pred_ci_lower
        self.y_pred_ci_upper = y_pred_ci_upper


class GridSearch:

    def __init__(self):
        pass

    @staticmethod
    def get_best_parameters(
        results,
        metric='val_r2s_mean',
        lower_metric_better=False,
        params=[
            'n_components',
            'polyorder',
            'deriv',
            'window_length',
        ],
        lower_parameter_better={
            'n_components': True,
            'polyorder': True,
            'deriv': True,
            'window_length': False,
        }
    ):
        """
        One standard error method of Breiman et al. (1984):
        select the simplest model whose mean falls within
        one standard deviation of the minimum.

        Here the simplest model is chosen based on (in order):
        - lowest n. components
        - lowest polyorder
        - lowest derivative
        - largest window_length
        """

        params = [p for p in params if p in results.columns]
        lower_parameter_better = [
            lower_parameter_better[p]
            for p in params
        ]

        error = results[metric]
        if not lower_metric_better:
            error = -error

        argmin_err = error.argmin()
        std_at_min = results.iloc[argmin_err]['val_r2s_std']
        metric_at_min = results.iloc[argmin_err][metric]
        best_results = results[
            results[metric].between(
                metric_at_min - std_at_min,
                metric_at_min + std_at_min
            )
        ]

        best_result = best_results.sort_values(
            by=params,
            ascending=lower_parameter_better
        ).iloc[0]

        return best_result[params].astype(float).to_dict()

    def run(
        self,
        prediction_name,
        parameters_dict,
        spectra,
        sample_data,
        output_folder,
        default_params=default_params,
        clear_display=False
    ):

        os.makedirs(output_folder, exist_ok=True)

        parameters_dict = {
            k: parameters_dict[k]
            for k in parameters_dict
            if k != 'n_committees'
        }

        parameters_grid = ParameterGrid(parameters_dict)

        train_r2s = []
        train_rmses = []
        test_r2s = []
        test_rmses = []
        count = 0

        for params in parameters_grid:
            if clear_display:
                clear_output(wait=True)
            print(params, ', iteration {} of {}'.format(
                count+1, len(list(parameters_grid))))
            count += 1

            preprocesser = AbsorbanceSpectraPreProcesser()

            params = {**default_params, **params}
            preprocessed_spectra = preprocesser.run(
                spectra, **{
                    k: params[k]
                    for k in params
                    if k != 'n_components'
                }
            )

            wavelengths = preprocessed_spectra.columns
            merged_data = preprocessed_spectra.merge(
                sample_data,
                left_index=True,
                right_index=True
            )

            X = merged_data[wavelengths]

            prediction_config = get_predictions_config(X, merged_data)[
                prediction_name]

            prediction = Prediction(**prediction_config)
            prediction.set_n_components(params['n_components'])
            X = prediction.X_train

            scores = prediction.cross_validate(n_splits=3, n_repeats=5)

            train_r2 = scores['train_r2']
            train_rmse = np.sqrt(-scores['train_neg_mean_squared_error'])

            test_r2 = scores['test_r2']
            test_rmse = np.sqrt(-scores['test_neg_mean_squared_error'])

            train_r2s.append(train_r2)
            train_rmses.append(train_rmse)
            test_r2s.append(test_r2)
            test_rmses.append(test_rmse)

        train_r2s = np.array(train_r2s)
        train_rmses = np.array(train_rmses)
        test_r2s = np.array(test_r2s)
        test_rmses = np.array(test_rmses)

        self.train_r2s = train_r2s
        self.train_rmses = train_rmses
        self.test_r2s = test_r2s
        self.test_rmses = test_rmses

        results = pd.DataFrame(parameters_grid)
        results['val_r2s_mean'] = test_r2s.mean(axis=1)
        results['val_r2s_std'] = test_r2s.std(axis=1)
        results['val_rmses_mean'] = test_rmses.mean(axis=1)
        results['val_rmses_std'] = test_rmses.std(axis=1)
        results['cal_r2s_mean'] = train_r2s.mean(axis=1)
        results['cal_r2s_std'] = train_r2s.std(axis=1)
        results['cal_rmses_mean'] = train_rmses.mean(axis=1)
        results['cal_rmses_std'] = train_rmses.std(axis=1)
        results.to_csv(os.path.join(output_folder, 'results.csv'), index=False)

        self.results = results

        best_parameters = self.get_best_parameters(results)
        self.best_parameters = best_parameters
        with open(os.path.join(output_folder, 'best_parameters.json'), "w") as outfile:
            json.dump(best_parameters, outfile)
        print("Best parameters:\n", best_parameters)

        return results, best_parameters


def run_preprocessing_training_prediction(
    spectra,
    preprocessing_config,
    n_components,
    sample_data,
    output_folder,
    names=[],
    n_splits=5,
    n_repeats=20
):
    preprocesser = AbsorbanceSpectraPreProcesser()
    preprocessed_spectra = preprocesser.run(spectra, **preprocessing_config)

    wavelengths = preprocessed_spectra.columns
    merged_data = preprocessed_spectra.merge(
        sample_data,
        left_index=True,
        right_index=True
    )

    X = merged_data[wavelengths]

    predictions_config = get_predictions_config(X, merged_data)
    if len(names) > 0:
        keys = names
    else:
        keys = predictions_config.keys()

    for key in keys:
        predictions_config[key]['save'] = True
        predictions_config[key]['output_folder'] = os.path.join(
            output_folder, key)
        predictions_config[key]['wavelengths'] = wavelengths

    predictions = {}
    for key in keys:
        try:
            predictions[key] = Prediction(**predictions_config[key])
        except Exception as e:
            print(e)
            continue

    for key, prediction in predictions.items():

        print(key)

        prediction.set_n_components(n_components[key])
        prediction.run_cross_validation(n_splits=n_splits, n_repeats=n_repeats)
        prediction.run_train_test_prediction()

        print('---')

    return predictions


def run_preprocessing_prediction(
    spectra,
    preprocessing_config,
    n_components,
    sample_data,
    trained_predictors,
    output_folder,
    spectra_original,
    names=[]
):
    '''
    '''
    preprocesser = AbsorbanceSpectraPreProcesser()
    preprocessed_spectra_original = preprocesser.run(
        spectra_original, **preprocessing_config)
    preprocessed_spectra = preprocesser.run(
        spectra, **preprocessing_config)[preprocessed_spectra_original.columns]

    wavelengths = preprocessed_spectra.columns
    merged_data = preprocessed_spectra.merge(
        sample_data,
        left_index=True,
        right_index=True
    )

    X = merged_data[wavelengths]

    predictions_config = get_predictions_config(X, merged_data)
    if names:
        keys = names
    else:
        keys = predictions_config.keys()

    for key in keys:
        predictions_config[key]['save'] = True
        predictions_config[key]['output_folder'] = os.path.join(
            output_folder, key)

    predictions = {}
    for key in keys:
        try:
            predictions[key] = Prediction(**predictions_config[key])
        except Exception as e:
            print(e)
            continue

    for key, prediction in predictions.items():

        print(key)

        trained_predictor = trained_predictors[key]

        prediction.set_n_components(trained_predictor.n_components)
        prediction.set_pretrained_pls(trained_predictor.pls)
        if prediction.model_name == 'beta-regression':
            prediction.set_pretrained_beta_regression(
                trained_predictor.beta_regression)
        prediction.run_prediction()

        print('---')

    return predictions

# Correlation Alignment CORAL


def coral(Xs, Xt):
    cov_src = np.cov(Xs, rowvar=False) + 1 * np.eye(Xs.shape[1])
    cov_tar = np.cov(Xt, rowvar=False) + 1 * np.eye(Xt.shape[1])
    A_coral = np.dot(np.linalg.cholesky(cov_tar),
                     np.linalg.inv(np.linalg.cholesky(cov_src)))
    Xs_new = np.dot(Xs, A_coral)
    return Xs_new


def run_preprocessing_coral_training_prediction(
    spectra_src,
    spectra_tar,
    preprocessing_config,
    n_components,
    sample_data_src,
    sample_data_tar,
    output_folder,
    names=[],
):
    preprocesser = AbsorbanceSpectraPreProcesser()

    preprocessed_spectra_src = preprocesser.run(
        spectra_src, **preprocessing_config)
    wavelengths_src = preprocessed_spectra_src.columns
    merged_data_src = preprocessed_spectra_src.merge(
        sample_data_src,
        left_index=True,
        right_index=True
    )
    X_src = merged_data_src[wavelengths_src]

    preprocessed_spectra_tar = preprocesser.run(
        spectra_tar, **preprocessing_config)[preprocessed_spectra_src.columns]
    wavelengths_tar = preprocessed_spectra_tar.columns
    merged_data_tar = preprocessed_spectra_tar.merge(
        sample_data_tar,
        left_index=True,
        right_index=True
    )
    X_tar = merged_data_tar[wavelengths_tar]

    X_src_aligned = coral(X_src.values, X_tar.values)
    X_src_aligned = pd.DataFrame(X_src_aligned, columns=X_src.columns)

    predictions_config_src = get_predictions_config(
        X_src_aligned, merged_data_src)
    predictions_config_tar = get_predictions_config(X_tar, merged_data_tar)
    if names:
        keys = names
    else:
        keys = predictions_config_tar.keys()

    for key in keys:
        predictions_config_src[key]['save'] = True
        predictions_config_src[key]['output_folder'] = os.path.join(
            output_folder, key)
        predictions_config_src[key]['wavelengths'] = wavelengths_src

        predictions_config_tar[key]['save'] = True
        predictions_config_tar[key]['output_folder'] = os.path.join(
            output_folder, key)
        predictions_config_tar[key]['wavelengths'] = wavelengths_tar

    predictions_tar = {}
    predictions_src = {}
    for key in keys:
        try:
            predictions_src[key] = Prediction(**predictions_config_src[key])
            predictions_tar[key] = Prediction(**predictions_config_tar[key])
        except Exception as e:
            print(e)
            continue

    for key in keys:

        prediction_src = predictions_src[key]
        prediction_tar = predictions_tar[key]

        print(key)

        prediction_src.set_n_components(n_components[key])
        prediction_src.run_train()

        prediction_tar.set_pretrained_pls(prediction_src.pls)
        if prediction_tar.model_name == 'beta-regression':
            prediction_tar.set_pretrained_beta_regression(
                prediction_src.beta_regression)
        prediction_tar.run_prediction()

        print('---')

    return predictions_src, predictions_tar
