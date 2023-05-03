import wrds

import datetime as dt

import pandas as pd
import numpy as np

from traceback import print_exc
from typing import Dict

###########################################
# Asset Dataframes Processing Functions
###########################################

def preprocess_tsla(tsla_df: pd.DataFrame =None, read_from: str =None):
    assert tsla_df is not None or read_from is not None, "Either tsla_df or read_from must be provided."
    assert tsla_df is None or  read_from is None, "Only one of tsla_df or read_from must be provided."
    
    # read the dataframe from the file if provided
    if read_from is not None:
        tsla_df = pd.read_csv(read_from, index_col=0)
    
    tsla_df = tsla_df.dropna(axis=0, subset=["ret"]).reset_index(drop=True)
    tsla_df = tsla_df.convert_dtypes()
    tsla_df["date"] = pd.to_datetime(tsla_df["date"], format="%Y-%m-%d")
    return tsla_df

def preprocess_rf(rf_df: pd.DataFrame =None, read_from: str =None):
    assert rf_df is not None or read_from is not None, "Either rf_df or read_from must be provided."
    assert rf_df is None or  read_from is None, "Only one of rf_df or read_from must be provided."
    
    # read the dataframe from the file if provided
    if read_from is not None:
        rf_df = pd.read_csv(read_from, index_col=0)
    
    rf_df = rf_df.convert_dtypes()
    
    # from monthly values to daily values, where the month's values are duplicated into the days of the month
    rf_df["date"] = rf_df["date"].apply(lambda date: pd.to_datetime(date, format="%Y-%m-%d").replace(day=1))
    rf_df["dummy"] = "rf"
    rf_df = rf_df.pivot(index="date", columns="dummy", values="rf")
    
    start_date = rf_df.index.min() - pd.DateOffset(day=1)
    end_date = rf_df.index.max() + pd.DateOffset(day=31)
    dates = pd.date_range(start_date, end_date, freq='D', name="date")
    rf_df = rf_df.reindex(dates, method="ffill")
    rf_df = rf_df.reset_index()
    rf_df.columns.rename("", inplace=True)
    
    return rf_df

def preprocess_mkt(mkt_df: pd.DataFrame =None, read_from: str =None):
    assert mkt_df is not None or read_from is not None, "Either mkt_df or read_from must be provided."
    assert mkt_df is None or  read_from is None, "Only one of mkt_df or read_from must be provided."
    
    # read the dataframe from the file if provided
    if read_from is not None:
        mkt_df = pd.read_csv(read_from, index_col=0)
    
    mkt_df = mkt_df.convert_dtypes()
    mkt_df["date"] = pd.to_datetime(mkt_df["date"], format="%Y-%m-%d")
    
    return mkt_df

def preprocess_btc(btc_df: pd.DataFrame =None, read_from: str =None):
    assert btc_df is not None or read_from is not None, "Either btc_df or read_from must be provided."
    assert btc_df is None or  read_from is None, "Only one of btc_df or read_from must be provided."
    
    # read the dataframe from the file if provided
    if read_from is not None:
        btc_df = pd.read_csv(read_from)
        
    btc_df = btc_df.convert_dtypes()
        
    btc_df = btc_df.drop(columns=["Open", "Volume", "Close", "Adj Close"])
    btc_df = btc_df.rename(columns={"High": "hiprc", "Low": "loprc", "Date": "date"})
    btc_df["midprc"] = (btc_df.hiprc + btc_df.loprc) / 2
    btc_df["date"] = pd.to_datetime(btc_df["date"], format="%Y-%m-%d")
    
    return btc_df

def merge_data(stock_df: pd.DataFrame, rf_df: pd.DataFrame, mkt_df: pd.DataFrame):
    stock_excess_ret = pd.DataFrame({"date": stock_df.date, "xsret": stock_df.ret - rf_df.rf})
    mkt_excess_ret   = pd.DataFrame({"date": stock_df.date, "mkt_xsret": mkt_df.mkt_ret - rf_df.rf})
    return pd.merge(stock_excess_ret, mkt_excess_ret, on="date", how="left")

def arg_nearest_after(date: dt.datetime, dates: pd.Series):
    date_diffs = dates - date
    for index in date_diffs.argsort():
        if date_diffs.iloc[index].days >= 0:
            return index
        
    return None
    
def prepare_for_bsts(data: pd.DataFrame, effect_date: dt.datetime):
    effect_lag = int(arg_nearest_after(effect_date, data.date))
    data = data.drop(columns=["date"]).astype(float)
    return data, effect_lag

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
    
    def get_table(self, table: str, library: str ="crsp", close_conn: bool = True):
        """Loads a WRDS table.

        Args:
            table (str): the table to be loaded.
            close_conn (bool, optional): Whether to close the connection after loading the table. Defaults to True.
            library (str): the library of the table to be loaded. Defaults to "crsp".

        Returns:
            _type_: _description_
        """
        try:
            table = self.conn.get_table(library=library, table=table)
            if close_conn:
                self.close()
            return table
        except Exception as e:
            self.close()
            print_exc()
        return None
    
    def download_stock_data(self, ticker2permno: Dict[str, int], from_: str, to: str, save_to: str =None, close_conn: bool =True):
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
        
        permno2ticker = {permno:ticker for ticker, permno in ticker2permno.items()}
        frames = {permno2ticker[permno]:group for permno, group in data.groupby("permno")}
        
        # saving the frames to csv files
        if save_to is not None:
            for ticker, frame in frames.items():
                frame.to_csv(save_to+f"{ticker}_{from_.split('-')[0]}_{to.split('-')[0]}.csv")
        
        return frames
    
    def download_rf_data(self, from_: str, to: str, save_to: str =None, close_conn: bool = True):
        """Loads risk-free rate data from CRSP.
        
        Args:
            from_ (str): start date of the data to be loaded.
            to (str): end date of the data to be loaded.
            save_to (str): path to save the data to. Default to None.
            close_conn (bool, optional): Whether to close the connection after executing the query. Defaults to True.
        
        Returns:
            pandas.DataFrame: the risk-free rate data for the given dates.
        """
        # https://www.crsp.org/products/documentation/crsp-risk-free-rates-file
        query = f"""
            select  mcaldt, tmytm 
            from crsp.tfz_mth_rf            
            where kytreasnox = 2000001 and mcaldt >= '{from_}'and mcaldt <= '{to}'
        """
        frame = self.exec(query, close_conn)     
        frame["tmytm"] = np.exp(frame["tmytm"]/12 /100) - 1
        frame = frame.rename(columns={"mcaldt":"date", "tmytm":"rf"})
        
        # saving the frames to csv files
        if save_to is not None:
            frame.to_csv(save_to+f"rf_{from_.split('-')[0]}_{to.split('-')[0]}.csv")

        return frame
    
    def download_market_pf_data(self, from_: str, to: str, save_to: str=None, close_conn: bool=True):
        """Returns the value-weighted market return from CRSP.

        Args:
            from_ (str): start date of the data to be loaded.
            to (str): end date of the data to be loaded.
            save_to (str, optional): path to save the data to. Defaults to None.
            close_conn (bool, optional): Whether to close the connection after executing the query. Defaults to True.

        Returns:
            pandas.DataFrame: 
        """
        query = f"""
            select  date,vwretd from crsp.dsi 
            where date>='{from_}' and date<='{to}'
        """
        frame = self.exec(query, close_conn)
        frame = frame.rename(columns={'vwretd':'mkt_ret'})
        
        # save the frame to csv file
        if save_to is not None:
            frame.to_csv(save_to+f"market_{from_.split('-')[0]}_{to.split('-')[0]}.csv")
        
        return frame
    
    def download_sp500_data(self, from_: str, to: str, save_to: str=None, close_conn: bool =True):
        """_summary_

        Args:
            from_ (str): _description_
            to (str): _description_
            save_to (str, optional): _description_. Defaults to None.
            close_conn (bool, optional): _description_. Defaults to True.
        """
        query = f"""
            select a.permno, a.date, a.ret, a.shrout, a.prc, b.siccd 
            from crsp.dsf as a left join crsp.msenames as b 
                on a.permno=b.permno and b.namedt<=a.date and a.date<=b.nameendt
            where a.date between '{from_}' and '{to}'
        """
        frame = self.exec(query, close_conn=False) # close_conn=False because we will use the connection again
        frame = frame.convert_dtypes()
        frame["date"] = pd.to_datetime(frame["date"], format="%Y-%m-%d")
        
        # Restrict to SP500 
        sp500_stocks = self.get_table(table='msp500list', close_conn=close_conn) # connection reused here
        
        frame = pd.merge(sp500_stocks, frame, on=['permno'], how='left')
        frame['ending'] = frame['ending'].fillna(frame.date.max())
        frame = frame[(frame.date >= frame.start) & (frame.date <= frame.ending)]
        
        # save the frame to csv file
        if save_to is not None:
            frame.to_csv(save_to+"SP500.csv")
        
        return frame
    