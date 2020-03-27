import os
import pandas as pd
def dump_to_folder(df,path):
    """
    Write a dataframe to csv, and any dataframe-valued fields
    are written as separate csvs, indexed by the field name
    """
    if not os.path.exists(path):
        os.makedirs(path)
        
    cols=df.columns.values
    df_cols=[]
    non_df_cols=[]
    for col in cols:
        val=df[col][ df[col].notnull()].values[0]
        if isinstance(val,pd.DataFrame):
            df_cols.append(col)
        else:
            non_df_cols.append(col)

    df_nodf=df.loc[:,non_df_cols].copy()
    
    for df_col in df_cols:
        fn_fld=df_col+"_fn"
        df_nodf[fn_fld]=["%s-%s.csv"%(df_col,idx) for idx in df_nodf.index.values]

        for idx,row in df.iterrows():
            if row[df_col].values is None: continue
            fn=os.path.join(path,df_nodf.loc[idx,fn_fld])
            row[df_col].to_csv(fn,index=False)

    df_nodf.to_csv(os.path.join(path,'master.csv'))

def read_from_folder(path):
    """
    reverse of dump_to_folder.
    """
    df_out=pd.read_csv(os.path.join(path,'master.csv'))

    fn_cols=[col for col in df.columns.values if col.endswith('_fn')]

    def ingest_csv(fn):
        return pd.read_csv(os.path.join(path,fn))
    
    for fn_col in fn_cols:
        df_col=fn_col.replace('_fn','')
        df_out[df_col]=df[fn_col].apply(ingest_csv)

        del df_out[fn_col]

