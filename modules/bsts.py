from causalimpact import CausalImpact

class BSTSForCausalInferenceWrapper():
    """
    Bayesian Structural Time-Series Causal Inference
    """
    def __init__(self, ts, at_lag, pre_effect_lag_window, post_effect_lag_window):
        self.ts = ts
        self.preperiod = range(at_lag - pre_effect_lag_window, at_lag)
        self.postperiod = range(at_lag, at_lag + post_effect_lag_window)
        self.result = None

    def analyse(self):
        """
        Analyse the time series data
        """
        self.result = CausalImpact(self.ts, self.preperiod, self.postperiod, prior_level_sd=None)
        return self

    def summary(self, textual=False):
        """
        Generate a summary of the analysis
        """
        return self.result.summary() if not(textual) else self.result.summary(output="report")
    
    def plot(self):
        """
        Plot the analysis
        """
        return self.result.plot()