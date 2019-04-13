import api_to_db_importer
import stock_constants as const

TARGET_DIR = './../target'
CSV_FILES_DIR = TARGET_DIR + '/data'

BINARY_BALANCED_SYMS = ['ABIO', 'ACET', 'ACFN', 'ACOR', 'ACUR', 'ADES', 'AEZS', 'AGYS', 'ALCO', 'ALKS', 'AMAG', 'AMCN',
                        'AMOV', 'AMRN', 'AMSC', 'AREX', 'ARNA', 'ASCMA', 'ASNA', 'ASTC', 'ASTE', 'ASYS', 'ATHN', 'AVAV',
                        'BBBY', 'BBOX', 'BCRX', 'BIDU', 'BIOL', 'BLDP', 'BSPM', 'CALI', 'CASI', 'CECE', 'CGEN', 'CHNR',
                        'CLDX', 'CONN', 'CPRX', 'CRAI', 'CRZO', 'CVGI', 'CYTR', 'DGICA', 'DGLY', 'DXPE', 'EMMS', 'ENDP',
                        'ENG', 'ESEA', 'FCNCA', 'FONR', 'FOSL', 'FRED', 'FTEK', 'GRBK', 'HEES', 'HUBG', 'HWKN', 'HYGS',
                        'ICAD', 'IDRA', 'IDSA', 'IIIN', 'IMGN', 'IMMR', 'INFI', 'INFN', 'INO', 'INSM', 'INTX', 'INWK',
                        'ISHG', 'JCS', 'KERX', 'KIRK', 'KOOL', 'LWAY', 'LYTS', 'MAG', 'MARK', 'MDCO', 'MDRX', 'MEIP',
                        'MNKD', 'MNTX', 'MOSY', 'MPAA', 'MVIS', 'MXWL', 'NFEC', 'NUAN', 'NYMX', 'OFIX', 'OLED', 'OMEX',
                        'PEGA', 'PEIX', 'PGNX', 'PHIIK', 'PICO', 'PLPC', 'POWL', 'PRTS', 'PTNR', 'RAIL', 'RBCN', 'REFR',
                        'REGN', 'RIGL', 'RMTI', 'ROYL', 'SCON', 'SCOR', 'SINA', 'SMRT', 'SMSI', 'SNSS', 'SOHU', 'SORL',
                        'SYNA', 'USAK', 'USAP', 'USEG', 'USLM', 'VRML', 'VVUS', 'ZIOP']

BALANCED_SYMS31 =['ACET', 'AGYS', 'ALCO', 'AMOV', 'ASCMA', 'ASTE', 'ATHN', 'AVAV', 'BBBY', 'CECE', 'CRAI', 'DGICA', 'ENDP', 'FCNCA', 'HUBG', 'HWKN', 'IIIN', 'JCS', 'KIRK', 'LWAY', 'MDRX', 'MPAA', 'NUAN', 'OFIX', 'PEGA', 'PICO', 'PLPC', 'POWL', 'PTNR', 'SCOR', 'USLM']
BALANCED_SYMS22 = ['ACET', 'ALCO', 'AMOV', 'ASTE', 'ATHN', 'AVAV', 'BBBY', 'CRAI', 'DGICA', 'ENDP', 'FCNCA', 'HUBG', 'HWKN', 'MDRX', 'MPAA', 'NUAN', 'OFIX', 'PEGA', 'PICO', 'POWL', 'SCOR', 'USLM']
BALANCED_SYMS11 = ['ALCO', 'AMOV', 'BBBY', 'DGICA', 'FCNCA', 'HUBG', 'HWKN', 'NUAN', 'OFIX', 'SCOR', 'USLM']

SELECTED_SYMS = 'USLM'

EPSILON = 0.01

import plot_helper as plth

def main():
    symbols = api_to_db_importer.SYMBOLS
    df_list = api_to_db_importer.Importer().import_data_from_files(symbols, CSV_FILES_DIR)

    balanced_syms = []

    for i in range(0, len(symbols)):
        sym = symbols[i]
        df = df_list[i]

        bincount = df[const.LABEL_BINARY_COL].value_counts(normalize=True)
        discretecount = df[const.LABEL_DISCRETE_COL].value_counts(normalize=True)

        bin_fall = bincount.loc[0.0]
        bin_raise = bincount.loc[1.0]

        dis_fall = discretecount.loc[0.0]
        dis_keep = discretecount.loc[1.0]
        dis_raise = discretecount.loc[2.0]

        if bin_fall >= 0.5-EPSILON and bin_fall <= 0.5+EPSILON:
            balanced_syms.append(sym)
            plth.plot_company_summary(df, sym)
    print(len(balanced_syms))
    print(balanced_syms)

if __name__ == '__main__':
    main()
    print('FINISHED')
