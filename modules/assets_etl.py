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
    df = df.reset_index(names=["date"])

    return df


def preproc_tsla_for_bsts(tsla_df: pd.DataFrame):
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


def preproc_ff3_for_bsts(ff_df: pd.DataFrame, after: str = "2010-01-01"):
    """Preprocesses the Fama-French 3-factor dataframe for the BSTS analysis.

    Args:
        ff_df (pd.DataFrame): the dataframe to preprocess.

    Returns:
        pd.DataFrame: the preprocessed dataframe.
    """
    ff_df = ff_df.rename(
        columns={"Mkt-RF": "m", "SMB": "smb", "HML": "hml", "RF": "rf"})
    ff_df["date"] = pd.to_datetime(
        ff_df["Unnamed: 0"].astype(str), format="%Y%m%d")
    ff_df = ff_df.drop(columns=["Unnamed: 0"])

    for col in ff_df.columns[:-1]:
        ff_df[col] = ff_df[col] * 1e-2

    # select only dates after
    ff_df = ff_df.query(f"date >= '{after}'")

    # inteprolate missing values
    ff_df = linear_interpolation(ff_df, cols=["m", "smb", "hml", "rf"])

    return ff_df


def preproc_btc_for_bsts(btc_df: pd.DataFrame):
    """Preprocesses the BTC dataframe for the BSTS analysis.

    Args:
        btc_df (pd.DataFrame, optional): the dataframe to preprocess.

    Returns:
        pd.DataFrame: the preprocessed dataframe.
    """

    btc_df = btc_df.convert_dtypes()

    btc_df = btc_df.drop(columns=["Open", "Volume", "Close", "Adj Close"])
    btc_df = btc_df.rename(
        columns={"High": "hiprc", "Low": "loprc", "Date": "date"})
    btc_df["midprc"] = (btc_df.hiprc + btc_df.loprc) / 2
    btc_df["ret"] = btc_df.midprc.pct_change()
    btc_df["date"] = pd.to_datetime(btc_df["date"], format="%Y-%m-%d")
    btc_df = btc_df.dropna(subset=["ret"])

    return btc_df

###########################################
# Asset Dataframes Transformation Functions
###########################################


def prepare_for_bsts(asset_df: pd.DataFrame, ff_df: pd.DataFrame):
    start_date = max(ff_df.date.min(), asset_df.date.min())
    end_date = min(ff_df.date.max(), asset_df.date.max())

    asset_df = asset_df.query("date >= @start_date and date <= @end_date")
    ff_df = ff_df.query("date >= @start_date and date <= @end_date")
    asset_df["retx"] = asset_df.ret.values - ff_df.rf.values
    return asset_df[["retx", "date"]].merge(ff_df, on="date", how="inner")

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
                frame.to_csv(
                    save_to+f"{ticker}_{from_.split('-')[0]}_{to.split('-')[0]}.csv")

        return frames
