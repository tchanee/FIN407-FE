from causal_impact.analysis import CausalImpact

import pandas as pd


class BSTSForCausalInferenceWrapper():
    """
    Bayesian Structural Time-Series Causal Inference
    """
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy(deep=True)
        self.ci = None

    def fit(self, at_lag: int, pre_effect_lag_window: int, post_effect_lag_window: int):
        """
        Analyse the time series data
        
        Args:
            at_lag: The lag at which the effect is to be measured.
            pre_effect_lag_window: The number of lags before the effect to be considered for the analysis.
            This excludes the lag at which the effect is to be measured.
            post_effect_lag_window: The number of lags after the effect to be considered for the analysis, including the lag at which the effect is to be measured.
        
        Returns:
            the caller object.
        """
        assert at_lag > 0
        assert at_lag + post_effect_lag_window <= len(self.data)
        
        # periods
        start = max(at_lag - pre_effect_lag_window, 0)
        end   = at_lag + post_effect_lag_window - 1
        pre_period  = [start, at_lag - 1]
        post_period = [at_lag, end]
        
        self.ci = CausalImpact(self.data.iloc[start:end+1], pre_period, post_period)      
        self.ci.run()
        return self

    def summary(self, textual: str =False):
        """
        Generates a summary of the analysis.
        
        Args:
            Textual: If True, return a textual summary. If False, return a numeric summary. Default is False.
        """
        print(self.ci.summary() if not(textual) else self.ci.summary(output="report"))
    
    def plot(self, figsize=(8, 7), fname=None):
        """
        Plot the analysis.
        """
        self.ci.plot(figsize=figsize, fname=fname, panels=["original", "pointwise"])


def asset_tweet_pair_analysis(excess_rets: pd.DataFrame, tweet: pd.Series, 
                              pre_effect_lag_window: int =14, post_effect_lag_window: int =3):
    """_summary_

    Args:
        excess_rets (pd.DataFrame): _description_
        tweet (pd.Series): _description_
        pre_effect_lag_window (int, optional): _description_. Defaults to 14.
        post_effect_lag_window (int, optional): _description_. Defaults to 5.

    Returns:
        _type_: _description_
    """
    
    prepared_data = excess_rets.drop(columns=["date"]).astype(float)
    
    effect_lag = None
    for idx, date in enumerate(excess_rets.date):
        if date >= tweet.date.values[0]:
            effect_lag = idx
            break
    if effect_lag is None:
        return None
    
    bsts_model = BSTSForCausalInferenceWrapper(prepared_data)
    bsts_model.fit(effect_lag, pre_effect_lag_window, post_effect_lag_window)
        
    return bsts_model
