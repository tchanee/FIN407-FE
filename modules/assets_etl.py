import wrds

import pandas as pd

from traceback import print_exc
from typing import Dict

###########################################
# asset dataframes preprocessing functions
###########################################

def preprocess_tsla(tsla_df: pd.DataFrame = None, read_from: str = None):
    assert tsla_df is not None or read_from is not None, "Either tsla_df or read_from must be provided."
    assert tsla_df is None or  read_from is None, "Only one of tsla_df or read_from must be provided."
    
    # read the dataframe from the file if provided
    if read_from is not None:
        tsla_df = pd.read_csv(read_from, index_col=0)
    
    tsla_df = tsla_df.dropna(axis=0, subset=["ret"]).reset_index(drop=True)
    tsla_df = tsla_df.convert_dtypes()
    tsla_df["date"] = tsla_df["date"].apply(lambda date_str: pd.to_datetime(date_str, format="%Y-%m-%d"))
    return tsla_df

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
            return self.conn.raw_sql(query)
        except Exception as e:
            self.close()
            print_exc()
            
        if close_conn:
            self.close()
        
        return None
    
    def download_stock_data(self, ticker2permno: Dict[str, int], from_: str, to: str, save_to: str =None, close_conn: bool =True):
        """Loads stock data from CRSP.
        
        Args:
            permnos (List[str]): list of permnos of stocks whose data is to be loaded.
            from_ (str): start date of the data to be loaded.
            to (str): end date of the data to be loaded.
            save_to (str): path to save the data to.
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