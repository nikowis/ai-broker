import time

import numpy as np
import pandas as pd

import alpha
import stock_constants as const
from db_access import create_db_connection, stock_collection

TI_BBANDS = 'BBANDS'
TI_RSI = 'RSI'
TI_STOCH = 'STOCH'
TI_MACD = 'MACD'
TI_EMA = 'EMA'
TI_SMA = 'SMA'

SYMBOL_KEY = "symbol"

API_MAX_PER_MINUTE_CALLS = 5
API_MAX_DAILY = 400

SYMBOLS = ['AAME', 'AAON', 'AAPL', 'AAWW', 'AAXJ', 'ABCB', 'ABIO', 'ABMD', 'ACAD', 'ACET', 'ACFN', 'ACGL',
           'ACHC', 'ACHN', 'ACIW', 'ACLS', 'ACNB', 'ACOR', 'ACTG', 'ACUR', 'ACWI', 'ACWX', 'ACXM', 'ADBE', 'ADES',
           'ADI', 'ADMP', 'ADP', 'ADRA', 'ADRD', 'ADRE', 'ADRU', 'ADSK', 'ADTN', 'ADXS', 'AEGN', 'AEHR', 'AEIS', 'AETI',
           'AEY', 'AEZS', 'AFSI', 'AGEN', 'AGNC', 'AGYS', 'AHPI', 'AIMC', 'AINV', 'AIRT', 'AKAM', 'AKRX', 'ALCO',
           'ALGN', 'ALGT', 'ALKS', 'ALLT', 'ALNY', 'ALOT', 'ALQA', 'ALSK', 'ALTR', 'ALXN', 'AMAG', 'AMAT', 'AMCN',
           'AMD', 'AMED', 'AMGN', 'AMKR', 'AMNB', 'AMOT', 'AMOV', 'AMRB', 'AMRN', 'AMSC', 'AMSF', 'AMSWA', 'AMTX',
           'AMWD', 'AMZN', 'ANAT', 'ANCX', 'ANDE', 'ANGO', 'ANIK', 'ANIP', 'ANSS', 'ANY', 'APDN', 'APEI', 'APOG',
           'APRI', 'APWC', 'ARAY', 'ARCB', 'ARCC', 'ARCI', 'ARCW', 'ARDM', 'AREX', 'ARII', 'ARKR', 'ARLP', 'ARNA',
           'AROW', 'ARQL', 'ARTW', 'ARTX', 'ARWR', 'ASCMA', 'ASFI', 'ASML', 'ASNA', 'ASRV', 'ASRVP', 'ASTC', 'ASTE',
           'ASTI', 'ASUR', 'ASYS', 'ATAI', 'ATAX', 'ATEC', 'ATHN', 'ATHX', 'ATLC', 'ATLO', 'ATNI', 'ATRC', 'ATRI',
           'ATRM', 'ATRO', 'ATRS', 'ATSG', 'ATTU', 'ATVI', 'AUBN', 'AUDC', 'AVAV', 'AVHI', 'AVID', 'AVNW', 'AWRE',
           'AXAS', 'AXDX', 'AXGN', 'AXTI', 'AZPN', 'BABY', 'BANF', 'BANFP', 'BANR', 'BASI', 'BBBY', 'BBGI', 'BBOX',
           'BCOR', 'BCPC', 'BCRX', 'BDGE', 'BDMS', 'BDSI', 'BEAT', 'BECN', 'BELFA', 'BFIN', 'BGCP', 'BGFV', 'BIDU',
           'BIIB', 'BIOC', 'BIOL', 'BIOS', 'BJRI', 'BKCC', 'BKSC', 'BLDP', 'BLDR', 'BLFS', 'BLIN', 'BLKB', 'BMRC',
           'BMRN', 'BMTC', 'BNCL', 'BNSO', 'BOCH', 'BOFI', 'BOKF', 'BOOM', 'BOSC', 'BOTJ', 'BPFH', 'BPOP', 'BPOPM',
           'BPTH', 'BREW', 'BRID', 'BRKS', 'BSET', 'BSPM', 'BSQR', 'BSRR', 'BSTC', 'BUSE', 'BYFC', 'CADC', 'CAKE',
           'CALI', 'CALL', 'CALM', 'CAMP', 'CARV', 'CASH', 'CASI', 'CASM', 'CASS', 'CBAN', 'CBFV', 'CBLI', 'CBRL',
           'CCBG', 'CCCL', 'CCNE', 'CCOI', 'CCRN', 'CDNS', 'CDTI', 'CECE', 'CENTA', 'CENX', 'CERN', 'CERS', 'CETV',
           'CEVA', 'CFFN', 'CFNB', 'CGEN', 'CHI', 'CHKE', 'CHKP', 'CHMG', 'CHNR', 'CHW', 'CHY', 'CINF', 'CIZN', 'CJJD',
           'CLCT', 'CLDX', 'CLFD', 'CLMT', 'CLUB', 'CLWT', 'CMCO', 'CMCSA', 'CMCT', 'CME', 'CNMD', 'COBZ', 'CONN',
           'CORT', 'COST', 'COWN', 'CPAH', 'CPHC', 'CPLP', 'CPRT', 'CPRX', 'CPSI', 'CPSS', 'CPST', 'CRAI', 'CRAY',
           'CREE', 'CREG', 'CRWS', 'CRZO', 'CSFL', 'CSGP', 'CSGS', 'CSPI', 'CSQ', 'CSWC', 'CTAS', 'CTWS', 'CTXS',
           'CUBA', 'CUI', 'CUTR', 'CVBF', 'CVCO', 'CVCY', 'CVGI', 'CVGW', 'CVLT', 'CVLY', 'CVTI', 'CVV', 'CYCC',
           'CYCCP', 'CYRN', 'CYTK', 'CYTR', 'CYTX', 'DAVE', 'DENN', 'DEST', 'DGICA', 'DGII', 'DGLY', 'DHIL', 'DJCO',
           'DLHC', 'DORM', 'DOX', 'DRAD', 'DRRX', 'DSPG', 'DSWL', 'DVAX', 'DVCR', 'DWCH', 'DXLG', 'DXPE', 'DXYN',
           'EBIX', 'EBMT', 'EBSB', 'EBTC', 'ECOL', 'ECPG', 'ECTE', 'EDUC', 'EEFT', 'EEI', 'EGBN', 'EGHT', 'EGLE',
           'EHTH', 'ELGX', 'ELSE', 'ELTK', 'EMCF', 'EMCI', 'EMITF', 'EMKR', 'EML', 'EMMS', 'ENDP', 'ENG', 'ENSG',
           'ENTG', 'EPAY', 'EQIX', 'ERIE', 'ESEA', 'ESGR', 'ESIO', 'ESLT', 'EVK', 'EVLV', 'EWBC', 'EXAS', 'EXEL',
           'EXFO', 'EXLS', 'EXPD', 'EZPW', 'FALC', 'FBIZ', 'FBMS', 'FBNC', 'FCNCA', 'FDEF', 'FEIM', 'FELE', 'FFBC',
           'FFHL', 'FFIC', 'FFIN', 'FFIV', 'FFNW', 'FISI', 'FISV', 'FITB', 'FIZZ', 'FLEX', 'FLIC', 'FLIR', 'FLL',
           'FLWS', 'FLXS', 'FMBH', 'FMBI', 'FMNB', 'FNHC', 'FNJN', 'FOLD', 'FONR', 'FORTY', 'FOSL', 'FRBK', 'FRED',
           'FRME', 'FTCS', 'FTEK', 'FTR', 'FULT', 'FUNC', 'FUND', 'GASS', 'GBCI', 'GBLI', 'GBNK', 'GCBC', 'GENC',
           'GGAL', 'GLUU', 'GOODO', 'GOODP', 'GOOGL', 'GPIC', 'GPOR', 'GPRE', 'GRBK', 'GRIF', 'GRMN', 'GROW', 'GRVY',
           'GSBC', 'GSIT', 'GT', 'GTIM', 'HA', 'HAFC', 'HAIN', 'HALL', 'HALO', 'HBIO', 'HBMD', 'HBNC', 'HBP', 'HCCI',
           'HCKT', 'HCSG', 'HEES', 'HELE', 'HFBC', 'HFBL', 'HIFS', 'HMNF', 'HMNY', 'HMSY', 'HRTX', 'HSIC', 'HSII',
           'HSKA', 'HTLD', 'HTLF', 'HUBG', 'HURC', 'HURN', 'HWKN', 'HYGS', 'IART', 'IBB', 'IBCP', 'IBKC', 'IBKR',
           'IBOC', 'ICAD', 'ICCC', 'ICFI', 'ICLN', 'ICLR', 'ICON', 'ICUI', 'IDCC', 'IDRA', 'IDSA', 'IDSY', 'IDTI',
           'IDXX', 'IEP', 'IESC', 'IEUS', 'IFEU', 'IFGL', 'IGLD', 'IGOV', 'III', 'IIIN', 'IIJI', 'IIN', 'IIVI', 'IKNX',
           'ILMN', 'IMGN', 'IMKTA', 'IMMR', 'IMMU', 'IMMY', 'IMNP', 'IMOS', 'INAP', 'INBK', 'INFI', 'INFN', 'INO',
           'INOD', 'INSM', 'INSY', 'INTC', 'INTG', 'INTL', 'INTU', 'INTX', 'INVE', 'INVT', 'INWK', 'IOSP', 'IPAR',
           'IPAS', 'IRBT', 'IRDM', 'IRIX', 'ISBC', 'ISCA', 'ISHG', 'ISIG', 'ISSC', 'JAZZ', 'JBHT', 'JBLU', 'JBSS',
           'JCOM', 'JCS', 'JCTCF', 'JJSF', 'JKHY', 'JOBS', 'JOUT', 'JRJC', 'JSM', 'KBAL', 'KCAP', 'KCLI', 'KELYA',
           'KELYB', 'KEQU', 'KERX', 'KFFB', 'KFRC', 'KGJI', 'KINS', 'KIRK', 'KLAC', 'KLIC', 'KNDI', 'KONA', 'KOOL',
           'KOPN', 'KOSS', 'KRNY', 'KTCC', 'KVHI', 'LBTYA', 'LBTYB', 'LBTYK', 'LCNB', 'LCUT', 'LECO', 'LION', 'LMAT',
           'LMNR', 'LMNX', 'LNDC', 'LOAN', 'LOGI', 'LPTH', 'LQDT', 'LRAD', 'LRCX', 'LSBK', 'LSCC', 'LSTR', 'LTBR',
           'LTRE', 'LTRX', 'LTXB', 'LULU', 'LUNA', 'LWAY', 'LXRX', 'LYTS', 'MAG', 'MAGS', 'MAMS', 'MANH', 'MANT', 'MAR',
           'MARK', 'MARPS', 'MASI', 'MAT', 'MATW', 'MAYS', 'MBCN', 'MBFI', 'MBTF', 'MBWM', 'MCBC', 'MCRI', 'MDCA',
           'MDCO', 'MDLZ', 'MDRX', 'MEET', 'MEIP', 'MFNC', 'MFSF', 'MGEE', 'MIDD', 'MIND', 'MINI', 'MITK', 'MKSI',
           'MKTX', 'MLAB', 'MLHR', 'MLNX', 'MLVF', 'MMAC', 'MMLP', 'MMSI', 'MNDO', 'MNGA', 'MNKD', 'MNST', 'MNTA',
           'MNTX', 'MOFG', 'MORN', 'MOSY', 'MPAA', 'MPB', 'MRLN', 'MRTN', 'MSEX', 'MSFT', 'MSON', 'MSTR', 'MTSC',
           'MTSL', 'MU', 'MVIS', 'MXIM', 'MXWL', 'MYGN', 'MYL', 'MYRG', 'NAII', 'NANO', 'NATH', 'NATI', 'NAUH', 'NAVG',
           'NBIX', 'NBN', 'NBTB', 'NECB', 'NEO', 'NEOG', 'NEON', 'NEPT', 'NFBK', 'NFEC', 'NICE', 'NNBR', 'NSEC', 'NTRI',
           'NTRS', 'NTWK', 'NUAN', 'NURO', 'NUVA', 'NVAX', 'NVCN', 'NVDA', 'NVEC', 'NXTM', 'NYMT', 'NYMX', 'NYNY',
           'OBAS', 'ODFL', 'ODP', 'OFIX', 'OFLX', 'OHAI', 'OHGI', 'OHRP', 'OIIM', 'OLBK', 'OLED', 'OMAB', 'OMCL',
           'OMEX', 'ONB', 'ONEQ', 'ONTX', 'OPHC', 'ORRF', 'OSBC', 'OVLY', 'PACW', 'PAGG', 'PATK', 'PAYX', 'PBIP',
           'PCAR', 'PCH', 'PEBK', 'PEBO', 'PEGA', 'PEIX', 'PENN', 'PETS', 'PFBC', 'PFBI', 'PFBX', 'PFSW', 'PGC', 'PGNX',
           'PGTI', 'PHII', 'PHIIK', 'PICO', 'PKBK', 'PKOH', 'PLAB', 'PLBC', 'PLCE', 'PLPC', 'PLUS', 'PLXS', 'PMBC',
           'PMD', 'PNBK', 'PNFP', 'PNNT', 'PNQI', 'PNRG', 'PNTR', 'PODD', 'POPE', 'POWI', 'POWL', 'PPBI', 'PPC', 'PROV',
           'PRPH', 'PRTS', 'PSEC', 'PTC', 'PTEN', 'PTIE', 'PTNR', 'PTSI', 'PTX', 'PWOD', 'QKLS', 'QQQX', 'QQXT', 'QTEC',
           'QUIK', 'QUMU', 'RADA', 'RAIL', 'RAND', 'RAVE', 'RAVN', 'RBCAA', 'RBCN', 'RCII', 'RCKY', 'RCMT', 'RDCM',
           'RDWR', 'RECN', 'REFR', 'REGN', 'REIS', 'RELL', 'RELV', 'RIBT', 'RICK', 'RIGL', 'RITT', 'RMCF', 'RMTI',
           'RNST', 'RNWK', 'ROCK', 'ROLL', 'ROST', 'ROYL', 'RRD', 'RRGB', 'RSYS', 'RTIX', 'RYAAY', 'SAFM', 'SAFT',
           'SAIA', 'SAL', 'SALM', 'SASR', 'SATS', 'SBAC', 'SBBX', 'SBCF', 'SBFG', 'SBGI', 'SBLK', 'SBNY', 'SBRA',
           'SBSI', 'SBUX', 'SCHL', 'SCHN', 'SCON', 'SCOR', 'SCSC', 'SFNC', 'SFST', 'SGEN', 'SGMA', 'SGMO', 'SGMS',
           'SGYP', 'SHBI', 'SHEN', 'SHIP', 'SHPG', 'SIEB', 'SIFI', 'SIMO', 'SINA', 'SINO', 'SIRI', 'SIVB', 'SKYW',
           'SLAB', 'SLMBP', 'SLP', 'SMED', 'SMIT', 'SMMF', 'SMRT', 'SMSI', 'SMTC', 'SMTX', 'SNCR', 'SNFCA', 'SNHY',
           'SNMX', 'SNPS', 'SNSS', 'SOHU', 'SONA', 'SONC', 'SORL', 'SOXX', 'SP', 'SPAR', 'SPCB', 'SPTN', 'SRCE', 'SRCL',
           'SSFN', 'SSYS', 'STAA', 'STMP', 'SYMC', 'SYNA', 'SYNL', 'SYNT', 'TAST', 'TATT', 'TCBI', 'TGTX', 'THFF',
           'THRM', 'TILE', 'TOWN', 'TREE', 'TRIB', 'TRIL', 'TROV', 'TROW', 'TRS', 'TRST', 'TRUE', 'TSBK', 'TTEK',
           'TTGT', 'TTMI', 'TTWO', 'TUES', 'TUSA', 'TWER', 'TWIN', 'TWMC', 'TZOO', 'UBCP', 'UBFO', 'UHAL', 'UIHC',
           'ULBI', 'UMBF', 'UMPQ', 'UNAM', 'UNB', 'UNFI', 'USAK', 'USAP', 'USAT', 'USATP', 'USEG', 'USLM', 'VBFC',
           'VIRC', 'VOD', 'VOXX', 'VRML', 'VVUS', 'WASH', 'WDC', 'WDFC', 'WEN', 'WERN', 'WETF', 'WEYS', 'WIRE', 'WLDN',
           'WLFC', 'WRLD', 'WSBC', 'WSBF', 'WSCI', 'WSFS', 'YRCW', 'ZAGG', 'ZION', 'ZIOP', 'ZN', 'ZUMZ']

API_KEYS = ['ULDORYWPDU2S2E6X', 'yM2zzAs6_DxdeT86rtZY', 'TX1OLY36K73S9MS9', 'I7RUE3LA4PSXDJU6', '41KVI2PCCMZ09Y69']


class Importer:

    def __init__(self) -> None:
        super().__init__()
        self.minute_count = 0
        self.daily_count = 0
        self.api_key_index = 0
        self.db = create_db_connection()
        self.api = alpha.AlphaVantage(API_KEYS[self.api_key_index])

    def json_to_df(self, json):
        json.pop(const.ID, None)
        json.pop(const.SYMBOL, None)
        df = pd.DataFrame.from_dict(json, orient=const.INDEX)
        df = df.astype(float)
        return df

    def df_to_json(self, df, ticker):
        json = df.to_dict(const.INDEX)
        json[const.SYMBOL] = ticker
        return json

    def import_one(self, sym):
        if stock_collection(self.db, False).count({SYMBOL_KEY: sym}) > 0:
            print('Found object with symbol ', sym)
        else:
            print('Didnt find object with symbol ', sym)
            raw_json = self.api.data_raw(sym).json(object_pairs_hook=self.remove_dots)
            keys = list(raw_json.keys())
            if len(keys) < 2:
                print('Symbol ', sym, 'not existing in alpha vantage')
                print(str(raw_json))
                return
            time_series_key = keys[1]
            time_series = raw_json[time_series_key]
            time_series[SYMBOL_KEY] = sym
            stock_collection(self.db, False).insert(time_series)
            self.increment_counters_sleep()

    def increment_counters_sleep(self):
        self.minute_count = self.minute_count + 1
        self.daily_count = self.daily_count + 1
        if self.daily_count >= API_MAX_DAILY:
            self.minute_count = 0
            self.daily_count = 0
            self.api_key_index = self.api_key_index + 1
            self.api = alpha.AlphaVantage(API_KEYS[self.api_key_index])
            print('####################CHANGING API KEY##################')
            time.sleep(10)
        if self.minute_count >= API_MAX_PER_MINUTE_CALLS:
            print('Sleeping.')
            time.sleep(65)
            self.minute_count = 0

    def remove_dots(self, items):
        result = {}
        for key, value in items:
            key = key.replace('.', ' ')
            result[key] = value
        return result

    def import_all(self, symbols):
        for sym in symbols:
            self.import_one(sym)

    def import_all_technical_indicators(self, tickers):
        for ticker in tickers:
            json = stock_collection(self.db, False).find_one({const.SYMBOL: ticker})
            df = self.json_to_df(json)
            self.import_technical_indicator(ticker, df, TI_SMA, const.SMA_10_COL, time_period=10)
            self.import_technical_indicator(ticker, df, TI_SMA, const.SMA_20_COL, time_period=20)
            self.import_technical_indicator(ticker, df, TI_EMA, const.EMA_10_COL, time_period=10)
            self.import_technical_indicator(ticker, df, TI_EMA, const.EMA_20_COL, time_period=20)
            self.import_technical_indicator(ticker, df, TI_MACD, const.MACD_COL)
            self.import_technical_indicator(ticker, df, TI_STOCH, const.STOCH_K_COL)
            self.import_technical_indicator(ticker, df, TI_RSI, const.RSI_10_COL, time_period=10)
            self.import_technical_indicator(ticker, df, TI_RSI, const.RSI_20_COL, time_period=20)
            self.import_technical_indicator(ticker, df, TI_BBANDS, const.BBANDS_10_RLB_COL, time_period=10)
            self.import_technical_indicator(ticker, df, TI_BBANDS, const.BBANDS_20_RLB_COL, time_period=20)

    def import_technical_indicator(self, ticker, df, indicator, col_name, time_period=None):
        if col_name not in df.columns:
            print('Importing ', indicator, ' for ', ticker)
            raw_json = self.api.technical_indicator(ticker, indicator, time_period=time_period).json(
                object_pairs_hook=self.remove_dots)
            keys = list(raw_json.keys())
            if len(keys) < 2:
                print('Symbol ', ticker, 'not existing in alpha vantage')
                print(str(raw_json))
                return
            time_series_key = keys[1]
            time_series = raw_json[time_series_key]
            indicator_df = self.json_to_df(time_series)
            if 'MACD' == indicator:
                prefix = col_name + ' '
                df[col_name] = indicator_df[indicator]
                df[prefix + 'Hist'] = indicator_df['MACD_Hist']
                df[prefix + 'Signal'] = indicator_df['MACD_Signal']
            elif 'STOCH' == indicator:
                prefix = indicator + ' '
                df[prefix + 'SlowK'] = indicator_df['SlowK']
                df[prefix + 'SlowD'] = indicator_df['SlowD']
            elif 'BBANDS' == indicator:
                prefix = str(time_period) + '-' + indicator + ' '
                df[prefix + 'Real Lower Band'] = indicator_df['Real Lower Band']
                df[prefix + 'Real Upper Band'] = indicator_df['Real Upper Band']
                df[prefix + 'Real Middle Band'] = indicator_df['Real Middle Band']
            else:
                df[col_name] = indicator_df[indicator]
            processed_json = self.df_to_json(df, ticker)
            stock_collection(self.db, False).remove({const.SYMBOL: ticker})
            stock_collection(self.db, False).insert(processed_json)
            self.increment_counters_sleep()

    def process_data(self):
        stock_collection_raw = stock_collection(self.db, False)
        stock_processed_collection = stock_collection(self.db, True)

        for stock in stock_collection_raw.find():
            symbol = stock[const.SYMBOL]
            df = self.json_to_df(stock)
            if const.SMA_10_COL in df.columns and const.BBANDS_20_RUB_COL in df.columns:
                df[const.LABEL_COL] = df[const.ADJUSTED_CLOSE_COL].shift(-const.FORECAST_DAYS)
                df[const.DAILY_PCT_CHANGE_COL] = (df[const.LABEL_COL] - df[const.ADJUSTED_CLOSE_COL]) / df[
                    const.ADJUSTED_CLOSE_COL] * 100.0
                df[const.HL_PCT_CHANGE_COL] = (df[const.HIGH_COL] - df[const.LOW_COL]) / df[
                    const.HIGH_COL] * 100
                df[const.LABEL_DISCRETE_COL] = df[const.DAILY_PCT_CHANGE_COL].apply(
                    lambda pct: np.NaN if pd.isna(pct)
                    else const.FALL_VALUE if pct < -const.TRESHOLD else const.RISE_VALUE if pct > const.TRESHOLD else const.IDLE_VALUE)
                df[const.LABEL_BINARY_COL] = df[const.DAILY_PCT_CHANGE_COL].apply(
                    lambda pct: np.NaN if pd.isna(pct)
                    else const.FALL_VALUE if pct < 0 else const.IDLE_VALUE if pct >= 0 else const.RISE_VALUE)
                processed_dict = self.df_to_json(df, symbol)
                stock_processed_collection.insert(processed_dict)
                print('Processed ', symbol)


if __name__ == "__main__":
    print(len(SYMBOLS))
    imp = Importer()
    # imp.import_all(SYMBOLS)
    # imp.process_data()
    imp.import_all_technical_indicators(SYMBOLS)
