import wrds

import pandas as pd
import numpy as np
import datetime as dt

from traceback import print_exc
from typing import Dict

###########################################
# Asset Dataframes Processing Functions
###########################################

def linear_interpolation(df: pd.DataFrame, cols: list):
    """Interpolates missing values in a dataframe using linear interpolation.

    Args:
        df (pd.DataFrame): the dataframe.
        cols (List, optional): the columns to interpolate.

    Returns:
        pandas.DataFrame: the dataframe with the interpolated columns.
    """
    # prepare for interpolation
    min_date = df.date.min()
    max_date = df.date.max()
    df = df.set_index("date")
    date_range = pd.date_range(min_date, max_date, freq="D")
    df = df.reindex(date_range, fill_value=np.nan)

    # interpolate missing values
    if cols is None:
        df = df.interpolate(method="linear")
    else:
        df[cols] = df[cols].interpolate(method="linear")

    # reset the index
    df = df.reset_index()

    return df

def bsts_prepare_stock(tsla_df: pd.DataFrame):
    """Preprocesses the TSLA dataframe for the BSTS analysis?

    Args:
        tsla_df (pd.DataFrame, optional): the dataframe to preprocess.

    Returns:
        pd.DataFrame: the preprocessed dataframe.
    """
    tsla_df = tsla_df.dropna(axis=0, subset=["ret"]).reset_index(drop=True)
    tsla_df["date"] = pd.to_datetime(tsla_df["date"], format="%Y-%m-%d")
    tsla_df = tsla_df.drop(columns=["Unnamed: 0"])
    tsla_df = tsla_df[["date", "ret"]]

    # interpolate missing values
    tsla_df = linear_interpolation(tsla_df, cols=["ret"])

    return tsla_df

def bsts_prepare_f3(factors_df: pd.DataFrame):
    """Preprocesses the Carhart 4-factor dataframe for the BSTS analysis.

    Args:
        factors_df (pd.DataFrame): the dataframe to preprocess.

    Returns:
        pd.DataFrame: the preprocessed dataframe.
    """
    factors_df = factors_df.rename(
        columns={"Mkt-RF": "m", "SMB": "smb", "HML": "hml", "RF": "rf"})
    factors_df["date"] = pd.to_datetime(
        factors_df["Unnamed: 0"].astype(str), format="%Y%m%d")
    factors_df = factors_df.drop(columns=["Unnamed: 0"])

    for col in factors_df.columns[:-1]:
        factors_df[col] = factors_df[col] * 1e-2

    # inteprolate missing values
    factors_df = linear_interpolation(factors_df, cols=["m", "smb", "hml", "rf"])

    return factors_df

def bsts_prepare_f6(factors_df: pd.DataFrame):
    """Preprocesses the Carhart 4-factor dataframe for the BSTS analysis.

    Args:
        factors_df (pd.DataFrame): the dataframe to preprocess.

    Returns:
        pd.DataFrame: the preprocessed dataframe.
    """
    factor_cols = ["m", "smb", "hml", "rf", "umd", "rmw", "cma"]
    
    factors_df = factors_df.rename(columns={"mktrf": "m"})
    factors_df["date"] = pd.to_datetime(factors_df.date.astype(str), format="%Y-%m-%d")

    # inteprolate missing values
    factors_df = linear_interpolation(factors_df, cols=factor_cols)
    factors_df = factors_df.dropna(subset=factor_cols)

    return factors_df

def bsts_prepare_crypto(crypto_df: pd.DataFrame):
    """Preprocesses the BTC dataframe for the BSTS analysis.

    Args:
        crypto_df (pd.DataFrame, optional): the dataframe to preprocess.

    Returns:
        pd.DataFrame: the preprocessed dataframe.
    """

    crypto_df = crypto_df.convert_dtypes()

    crypto_df = crypto_df.drop(columns=["Open", "Volume", "Close", "Adj Close"])
    crypto_df = crypto_df.rename(
        columns={"High": "hiprc", "Low": "loprc", "Date": "date"})
    crypto_df["midprc"] = (crypto_df.hiprc + crypto_df.loprc) / 2
    crypto_df["ret"] = crypto_df.midprc.pct_change()
    crypto_df["date"] = pd.to_datetime(crypto_df["date"], format="%Y-%m-%d")
    crypto_df = crypto_df.dropna(subset=["ret"])

    return crypto_df

###########################################
# Asset Dataframes Transformation Functions
###########################################

def bsts_prepare_data(asset_df: pd.DataFrame, factors_df: pd.DataFrame):
    start_date = max(factors_df.date.min(), asset_df.date.min())
    end_date = min(factors_df.date.max(), asset_df.date.max())
    asset_df = asset_df.query("date >= @start_date and date <= @end_date")
    factors_df = factors_df.query("date >= @start_date and date <= @end_date")
    
    asset_df["retx"] = asset_df.ret.values - factors_df.rf.values
    factors_df = factors_df.drop(columns=["rf"])
    merged = asset_df[["retx", "date"]].merge(factors_df, on="date", how="inner")
    
    return merged

#############################
# WRDS loader
#############################


class WRDSLoader():
    def __init__(self, username: str, pgpass: bool = False):
        self.conn = wrds.Connection(wrds_username=username, pgpass=pgpass)

    def connect(self, username: str, pgpass: bool = False):
        """Connects the loader to WRDS if not connected already.

        Args:
            username (str): username to connect to WRDS.
            pgpass (bool, optional): Whether to allow WRDS connector to ask for a '.pgpass' file setup. Defaults to False.

        Returns:
            WRDSLoader: the caller object.
        """
        if self.conn is not None:
            return self.conn
        self.conn = wrds.Connection(wrds_username=username, pgpass=pgpass)
        return self

    def close(self):
        """Closes the connection to WRDS.
        """
        self.conn.close()

    def exec(self, query: str, close_conn: bool = True):
        """Executes a WRDS query SQL query.

        Args:
            query (str): the query to be executed.
            close_conn (bool, optional): Whether to close the connection after executing the query. Defaults to True.

        Returns:
            pandas.DataFrame: the result of the query. 
        """
        try:
            frame = self.conn.raw_sql(query)
            if close_conn:
                self.close()
            return frame
        except Exception as e:
            self.close()
            print_exc()

        return None

    def download_stock_data(self, ticker2permno: Dict[str, int], from_: str, to: str, save_to: str = None, close_conn: bool = True):
        """Loads stock data from CRSP.

        Args:
            permnos (List[str]): list of permnos of stocks whose data is to be loaded.
            from_ (str): start date of the data to be loaded.
            to (str): end date of the data to be loaded.
            save_to (str): path to save the data to. Defaults to None.
            close_conn (bool, optional): Whether to close the connection after executing the query. Defaults to True.

        Returns:
            pandas.DataFrame: the stock data for the given permnos and dates.
        """
        permnos = list(ticker2permno.values())
        query = f"""
                select date, permno, ret
                from crsp.dsf
                where date >= '{from_}' and date <= '{to}' and permno in ({','.join([f"'{permno}'" for permno in permnos])})
        """
        data = self.exec(query, close_conn)

        permno2ticker = {permno: ticker for ticker,
                         permno in ticker2permno.items()}
        frames = {permno2ticker[permno]: group for permno,
                  group in data.groupby("permno")}
        
        # saving the frames to csv files
        if save_to is not None:
            for ticker, frame in frames.items():
                min_year = frame.date.min().strftime("%Y-%m-%d").split('-')[0]
                max_year = frame.date.max().strftime("%Y-%m-%d").split('-')[0]
                frame.to_csv(
                    save_to+f"{ticker}_{min_year}_{max_year}.csv")

        return frames
