import pandas as pd
import pymongo

import db.stock_constants as const

DB = "ai-broker"
STOCK_COLLECTION = "stock"
PROCESSED_STOCK_COLLECTION = "processed_stock"
LOCAL_URL = "mongodb://localhost:27017/"
REMOTE_URL = "mongodb://admin:<pswd>@ds125574.mlab.com:25574/ai-broker"

"""Symbols with large history (over 5200 days)"""
SELECTED_SYMBOLS_LIST = ['AVNW', 'AWRE', 'BPFH', 'CALL', 'CALM', 'CAMP', 'CARV', 'CASH', 'CASI', 'CASS', 'CENX', 'CERN',
                         'CERS', 'CETV', 'CFNB', 'CHKE', 'CHKP', 'CHNR', 'CLWT', 'CMCO', 'CMCSA', 'CMCT', 'CNMD',
                         'CREE', 'CRZO', 'CTAS', 'CUBA', 'CVTI', 'CYTR', 'DAVE', 'DEST', 'DJCO', 'DLHC', 'DSPG', 'DSWL',
                         'DWCH', 'DXYN', 'EDUC', 'EEFT', 'EEI', 'ELSE', 'ELTK', 'EMCI', 'EMITF', 'EMKR', 'EML', 'EMMS',
                         'ENG', 'EVLV', 'FBNC', 'FELE', 'FFIC', 'FFIN', 'FISV', 'FITB', 'FIZZ', 'FLEX', 'FLIC', 'FLIR',
                         'FLL', 'FMBI', 'FRBK', 'FRME', 'FTEK', 'FTR', 'FULT', 'FUNC', 'FUND', 'GPIC', 'GRIF', 'GSBC',
                         'GT', 'GTIM', 'HA', 'HAFC', 'HAIN', 'HALL', 'HBHC', 'HCSG', 'HMNY', 'HMSY', 'HRTX', 'HSIC',
                         'HUBG', 'HURC', 'HWKN', 'IBKC', 'IBOC', 'ICAD', 'ICCC', 'ICON', 'ICUI', 'IDCC', 'IDRA', 'IDSA',
                         'IDTI', 'IDXX', 'IEP', 'IIIN', 'IIN', 'IIVI', 'IMKTA', 'IMMU', 'INOD', 'INTC', 'INTL', 'INTU',
                         'INVE', 'IPAR', 'IRIX', 'ISCA', 'JBSS', 'JCS', 'JCTCF', 'JJSF', 'JKHY', 'JOUT', 'KBAL', 'KCLI',
                         'KELYA', 'KELYB', 'KEQU', 'KLAC', 'KLIC', 'KOOL', 'KOPN', 'KTCC', 'LCUT', 'LECO', 'LNDC',
                         'LOGI', 'LPTH', 'LRAD', 'LRCX', 'LSCC', 'LSTR', 'LTRE', 'LWAY', 'LYTS', 'MAG', 'MAGS', 'MAR',
                         'MARPS', 'MBFI', 'MDCA', 'MGEE', 'MIND', 'MINI', 'MITK', 'MLAB', 'MLHR', 'MMAC', 'MNST',
                         'MPAA', 'MPB', 'MSEX', 'MSFT', 'MSON', 'MTSC', 'MTSL', 'MU', 'MXWL', 'MYGN', 'MYL', 'NAII',
                         'NANO', 'NATH', 'NAVG', 'NBIX', 'NBN', 'NBTB', 'NEOG', 'NEON', 'NNBR', 'NTRS', 'NVAX', 'NVEC',
                         'ODFL', 'ODP', 'OFIX', 'OHGI', 'OLED', 'ONB', 'PATK', 'PAYX', 'PCAR', 'PCH', 'PEBK', 'PEBO',
                         'PEGA', 'PENN', 'PGNX', 'PHII', 'PHIIK', 'PLCE', 'PLUS', 'PLXS', 'PMD', 'PNBK', 'PNTR', 'POPE',
                         'POWI', 'POWL', 'PPBI', 'PPC', 'PROV', 'PRPH', 'PTC', 'PTEN', 'PTSI', 'PTX', 'PWOD', 'QUMU',
                         'RADA', 'RAVE', 'RAVN', 'RDCM', 'REFR', 'REGN', 'RELL', 'RELV', 'RICK', 'RITT', 'RMCF', 'RNST',
                         'RNWK', 'ROST', 'RRD', 'RYAAY', 'SAFM', 'SASR', 'SBCF', 'SBGI', 'SBUX', 'SCHL', 'SCHN', 'SFNC',
                         'SIEB', 'SIRI', 'SIVB', 'SMIT', 'SMRT', 'SMTC', 'SNHY', 'SNPS', 'SONC', 'SPAR', 'SSYS', 'STAA',
                         'SYMC', 'SYNL', 'THFF', 'THRM', 'TILE', 'TRIB', 'TTEK', 'TWIN', 'TWMC', 'UBCP', 'UHAL', 'ULBI',
                         'USAK', 'USAP', 'USEG', 'USLM', 'VIRC', 'VVUS', 'WDC', 'WDFC', 'WEN', 'WERN', 'WETF', 'WEYS',
                         'WIRE', 'WRLD', 'WSBC', 'WSCI', 'WSFS', 'YRCW']


def create_db_connection(remote=False, db_name=DB):
    if not remote:
        url = LOCAL_URL
    else:
        url = REMOTE_URL
    mongo_client = pymongo.MongoClient(url)
    db_conn = mongo_client[db_name]
    return db_conn


def stock_collection(db_conn, processed=True):
    if processed:
        return db_conn[PROCESSED_STOCK_COLLECTION]
    else:
        return db_conn[STOCK_COLLECTION]


def find_by_tickers_to_dateframe_parse_to_df_list(db_conn, symbol_list, processed=True):
    data = stock_collection(db_conn, processed).find({const.SYMBOL: {"$in": symbol_list}})
    df_list = []
    for document in data:
        document.pop(const.ID, None)
        document.pop(const.SYMBOL, None)
        df = pd.DataFrame.from_dict(document, orient=const.INDEX)
        df = df.astype(float)
        df_list.append(df)
    if len(df_list) == 0:
        raise Exception('No data with any ticker of ' + symbol_list + ' was found.')
    return df_list
