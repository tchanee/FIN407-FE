from causalimpact.analysis import CausalImpact
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
    
    def plot(self):
        """
        Plot the analysis.
        """
        return self.ci.plot()