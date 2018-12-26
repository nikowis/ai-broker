import time

import alpha
from db_access import create_db_connection, stock_collection

SYMBOL_KEY = "symbol"

API_MAX_PER_MINUTE_CALLS = 5
API_MAX_DAILY = 500

SYMBOLS = ["AAIT", "AAL", "AAME", "AAOI", "AAON", "AAPL", "AAVL", "AAWW", "AAXJ", "ABAC", "ABAX", "ABCB", "ABCD",
           "ABCO", "ABCW", "ABDC", "ABGB", "ABIO", "ABMD", "ABTL", "ABY", "ACAD", "ACAS", "ACAT", "ACET", "ACFC",
           "ACFN", "ACGL", "ACHC", "ACHN", "ACIW", "ACLS", "ACNB", "ACOR", "ACRX", "ACSF", "ACST", "ACTA",
           "ACTG", "ACTS", "ACUR", "ACWI", "ACWX", "ACXM", "ADAT", "ADBE", "ADEP", "ADES", "ADHD", "ADI", "ADMA",
           "ADMP", "ADMS", "ADNC", "ADP", "ADRA", "ADRD", "ADRE", "ADRU", "ADSK", "ADTN", "ADUS", "ADVS", "ADXS",
           "ADXSW", "AEGN", "AEGR", "AEHR", "AEIS", "AEPI", "AERI", "AETI", "AEY", "AEZS", "AFAM", "AFCB", "AFFX",
           "AFH", "AFMD", "AFOP", "AFSI", "AGEN", "AGII", "AGIIL", "AGIO", "AGNC", "AGNCB", "AGNCP", "AGND", "AGRX",
           "AGTC", "AGYS", "AGZD", "AHGP", "AHPI", "AIMC", "AINV", "AIQ", "AIRM", "AIRR", "AIRT", "AIXG", "AKAM",
           "AKAO", "AKBA", "AKER", "AKRX", "ALCO", "ALDR", "ALDX", "ALGN", "ALGT", "ALIM", "ALKS", "ALLT",
           "ALNY", "ALOG", "ALOT", "ALQA", "ALSK", "ALTR", "ALXA", "ALXN", "AMAG", "AMAT", "AMBA", "AMBC", "AMBCW",
           "AMCC", "AMCF", "AMCN", "AMCX", "AMD", "AMDA", "AMED", "AMGN", "AMIC", "AMKR", "AMNB", "AMOT", "AMOV",
           "AMPH", "AMRB", "AMRI", "AMRK", "AMRN", "AMRS", "AMSC", "AMSF", "AMSG", "AMSGP", "AMSWA", "AMTX", "AMWD",
           "AMZN", "ANAC", "ANAD", "ANAT", "ANCB", "ANCI", "ANCX", "ANDE", "ANGI", "ANGO", "ANIK", "ANIP", "ANSS",
           "ANTH", "ANY", "AOSL", "APAGF", "APDN", "APDNW", "APEI", "APOG", "APOL", "APRI", "APTO",
           "APWC", "AQXP", "ARAY", "ARCB", "ARCC", "ARCI", "ARCP", "ARCPP", "ARCW", "ARDM", "ARDX", "AREX", "ARGS",
           "ARIA", "ARII", "ARIS", "ARKR", "ARLP", "ARMH", "ARNA", "AROW", "ARQL", "ARTW",
           "ARTX", "ARUN", "ARWR", "ASBB", "ASBI", "ASCMA", "ASEI", "ASFI", "ASMB", "ASMI", "ASML", "ASNA", "ASPS",
           "ASPX", "ASRV", "ASRVP", "ASTC", "ASTE", "ASTI", "ASUR", "ASYS", "ATAI", "ATAX", "ATEA", "ATEC", "ATHN",
           "ATHX", "ATLC", "ATLO", "ATML", "ATNI", "ATNY", "ATOS", "ATRA", "ATRC", "ATRI", "ATRM", "ATRO", "ATRS",
           "ATSG", "ATTU", "ATVI", "AUBN", "AUDC", "AUMA", "AUMAU", "AUMAW", "AUPH", "AUXL", "AVAV", "AVEO", "AVGO",
           "AVHI", "AVID", "AVNR", "AVNW", "AWAY", "AWRE", "AXAS", "AXDX", "AXGN", "AXJS", "AXPW", "AXPWW", "AXTI",
           "AZPN", "BABY", "BAGR", "BAMM", "BANF", "BANFP", "BANR", "BANX", "BASI", "BBBY", "BBC", "BBCN", "BBEP",
           "BBEPP", "BBGI", "BBLU", "BBNK", "BBOX", "BBP", "BBRG", "BBRY", "BBSI", "BCBP", "BCLI", "BCOM", "BCOR",
           "BCOV", "BCPC", "BCRX", "BDBD", "BDCV", "BDE", "BDGE", "BDMS", "BDSI", "BEAT", "BEAV", "BEBE", "BECN",
           "BELFA", "BELFB", "BFIN", "BGCP", "BGFV", "BGMD", "BHBK", "BIB", "BICK", "BIDU", "BIIB", "BIND", "BIOC",
           "BIOD", "BIOL", "BIOS", "BIRT", "BIS", "BJRI", "BKCC", "BKEP", "BKEPP", "BKMU", "BKSC", "BKYF", "BLCM",
           "BLDP", "BLDR", "BLFS", "BLIN", "BLKB", "BLMN", "BLMT", "BLRX", "BLUE", "BLVD", "BLVDU", "BLVDW", "BMRC",
           "BMRN", "BMTC", "BNCL", "BNCN", "BNDX", "BNFT", "BNSO", "BOBE", "BOCH", "BOFI", "BOKF", "BONA", "BONT",
           "BOOM", "BOSC", "BOTA", "BOTJ", "BPFH", "BPFHP", "BPFHW", "BPOP", "BPOPM", "BPOPN", "BPTH", "BRCD", "BRCM",
           "BRDR", "BREW", "BRID", "BRKL", "BRKR", "BRKS", "BRLI", "BSDM", "BSET", "BSF", "BSFT", "BSPM", "BSQR",
           "BSRR", "BSTC", "BTUI", "BUR", "BUSE", "BV", "BVA", "BVSN", "BWEN", "BWFG", "BWINA", "BWINB", "BWLD", "BYBK",
           "BYFC", "BYLK", "CA", "CAAS", "CAC", "CACB", "CACC", "CACG", "CACGU", "CACGW", "CACH", "CACQ", "CADC",
           "CADT", "CADTR", "CADTU", "CADTW", "CAKE", "CALA", "CALD", "CALI", "CALL", "CALM", "CAMB", "CAMBU", "CAMBW",
           "CAMP", "CAMT", "CAPN", "CAPNW", "CAR", "CARA", "CARB", "CARO", "CART", "CARV", "CARZ", "CASH", "CASI",
           "CASM", "CASS", "CASY", "CATM", "CATY", "CATYW", "CAVM", "CBAK", "CBAN", "CBAY", "CBDE", "CBF", "CBFV",
           "CBIN", "CBLI", "CBMG", "CBMX", "CBNJ", "CBNK", "CBOE", "CBPO", "CBRL", "CBRX", "CBSH", "CBSHP", "CBST",
           "CBSTZ", "CCBG", "CCCL", "CCCR", "CCIH", "CCLP", "CCMP", "CCNE", "CCOI", "CCRN", "CCUR", "CCXI", "CDC",
           "CDK", "CDNA", "CDNS", "CDTI", "CDW", "CDXS", "CDZI", "CECE", "CECO", "CELG", "CELGZ", "CEMI", "CEMP",
           "CENT", "CENTA", "CENX", "CERE", "CERN", "CERS", "CERU", "CETV", "CEVA", "CFA", "CFBK", "CFFI", "CFFN",
           "CFGE", "CFNB", "CFNL", "CFO", "CFRX", "CFRXW", "CFRXZ", "CG", "CGEN", "CGIX", "CGNX", "CGO", "CHCI", "CHCO",
           "CHDN", "CHEF", "CHEV", "CHFC", "CHFN", "CHI", "CHKE", "CHKP", "CHLN", "CHMG", "CHNR", "CHOP", "CHRS",
           "CHRW", "CHSCM", "CHSCN", "CHSCO", "CHSCP", "CHTR", "CHUY", "CHW", "CHXF", "CHY", "CHYR", "CIDM", "CIFC",
           "CIMT", "CINF", "CISAW", "CISG", "CIZ", "CIZN", "CJJD", "CKEC", "CKSW", "CLAC", "CLACU", "CLACW", "CLBH",
           "CLCT", "CLDN", "CLDX", "CLFD", "CLIR", "CLMS", "CLMT", "CLNE", "CLNT", "CLRB", "CLRBW", "CLRO", "CLRX",
           "CLSN", "CLTX", "CLUB", "CLVS", "CLWT", "CMCO", "CMCSA", "CMCSK", "CMCT", "CME", "CMFN", "CMGE", "CMLS",
           "CMPR", "CMRX", "CMSB", "CMTL", "CNAT", "CNBKA", "CNCE", "CNDO", "CNET", "CNIT", "CNLM", "CNLMR", "CNLMU",
           "CNLMW", "CNMD", "CNOB", "CNSI", "CNSL", "CNTF", "CNTY", "CNV", "CNXR", "CNYD", "COB", "COBK", "COBZ",
           "COCO", "COHR", "COHU", "COKE", "COLB", "COLM", "COMM", "COMT", "CONE", "CONN", "COOL", "CORE", "CORI",
           "CORT", "COSI", "COST", "COVS", "COWN", "COWNL", "CPAH", "CPGI", "CPHC", "CPHD", "CPHR", "CPIX", "CPLA",
           "CPLP", "CPRT", "CPRX", "CPSI", "CPSS", "CPST", "CPTA", "CPXX", "CRAI", "CRAY", "CRDC", "CRDS", "CRDT",
           "CREE", "CREG", "CRESW", "CRESY", "CRIS", "CRME", "CRMT", "CRNT", "CROX", "CRRC", "CRRS", "CRTN", "CRTO",
           "CRUS", "CRVL", "CRWN", "CRWS", "CRZO", "CSBK", "CSCD", "CSCO", "CSF", "CSFL", "CSGP", "CSGS", "CSII",
           "CSIQ", "CSOD", "CSPI", "CSQ", "CSRE", "CSTE", "CSUN", "CSWC", "CTAS", "CTBI", "CTCM", "CTCT", "CTG", "CTHR",
           "CTIB", "CTIC", "CTRE", "CTRL", "CTRN", "CTRP", "CTRX", "CTSH", "CTSO", "CTWS", "CTXS", "CU", "CUBA", "CUI",
           "CUNB", "CUTR", "CVBF", "CVCO", "CVCY", "CVGI", "CVGW", "CVLT", "CVLY", "CVTI", "CVV", "CWAY", "CWBC",
           "CWCO", "CWST", "CXDC", "CY", "CYAN", "CYBE", "CYBR", "CYBX", "CYCC", "CYCCP", "CYHHZ", "CYNO", "CYOU",
           "CYRN", "CYTK", "CYTR", "CYTX", "CZFC", "CZNC", "CZR", "CZWI", "DAEG", "DAIO", "DAKT", "DARA", "DATE",
           "DAVE", "DAX", "DBVT", "DCIX", "DCOM", "DCTH", "DENN", "DEPO", "DERM", "DEST", "DFRG", "DFVL", "DFVS",
           "DGAS", "DGICA", "DGICB", "DGII", "DGLD", "DGLY", "DGRE", "DGRS", "DGRW", "DHIL", "DHRM", "DIOD", "DISCA",
           "DISCB", "DISCK", "DISH", "DJCO", "DLBL", "DLBS", "DLHC", "DLTR", "DMLP", "DMND", "DMRC", "DNBF", "DNKN",
           "DORM", "DOVR", "DOX", "DPRX", "DRAD", "DRAM", "DRIV", "DRNA", "DRRX", "DRWI", "DRWIW", "DRYS", "DSCI",
           "DSCO", "DSGX", "DSKX", "DSKY", "DSLV", "DSPG", "DSWL", "DTLK", "DTSI", "DTUL", "DTUS", "DTV", "DTYL",
           "DTYS", "DVAX", "DVCR", "DWA", "DWAT", "DWCH", "DWSN", "DXCM", "DXGE", "DXJS", "DXKW", "DXLG", "DXM", "DXPE",
           "DXPS", "DXYN", "DYAX", "DYNT", "DYSL", "EA", "EAC", "EARS", "EBAY", "EBIO", "EBIX", "EBMT", "EBSB", "EBTC",
           "ECHO", "ECOL", "ECPG", "ECTE", "ECYT", "EDAP", "EDGW", "EDS", "EDUC", "EEFT", "EEI", "EEMA", "EEME", "EEML",
           "EFII", "EFOI", "EFSC", "EFUT", "EGAN", "EGBN", "EGHT", "EGLE", "EGLT", "EGOV", "EGRW", "EGRX", "EGT",
           "EHTH", "EIGI", "ELGX", "ELNK", "ELON", "ELOS", "ELRC", "ELSE", "ELTK", "EMCB", "EMCF", "EMCG", "EMCI",
           "EMDI", "EMEY", "EMIF", "EMITF", "EMKR", "EML", "EMMS", "EMMSP", "ENDP", "ENFC", "ENG", "ENOC", "ENPH",
           "ENSG", "ENT", "ENTA", "ENTG", "ENTR", "ENVI", "ENZN", "ENZY", "EOPN", "EPAX", "EPAY", "EPIQ", "EPRS",
           "EPZM", "EQIX", "ERI", "ERIC", "ERIE", "ERII", "EROC", "ERS", "ERW", "ESBF", "ESBK", "ESCA", "ESCR", "ESCRP",
           "ESEA", "ESGR", "ESIO", "ESLT", "ESMC", "ESPR", "ESRX", "ESSA", "ESSX", "ESXB", "ESYS", "ETFC", "ETRM",
           "EUFN", "EVAL", "EVAR", "EVBS", "EVEP", "EVK", "EVLV", "EVOK", "EVOL", "EVRY", "EWBC", "EXA", "EXAC", "EXAS",
           "EXEL", "EXFO", "EXLP", "EXLS", "EXPD", "EXPE", "EXPO", "EXTR", "EXXI", "EYES", "EZCH", "EZPW", "FALC",
           "FANG", "FARM", "FARO", "FAST", "FATE", "FB", "FBIZ", "FBMS", "FBNC", "FBNK", "FBRC", "FBSS", "FCAP", "FCBC",
           "FCCO", "FCCY", "FCEL", "FCFS", "FCHI", "FCLF", "FCNCA", "FCS", "FCSC", "FCTY", "FCVA", "FCZA", "FCZAP",
           "FDEF", "FDIV", "FDML", "FDUS", "FEIC", "FEIM", "FELE", "FEMB", "FES", "FEUZ", "FEYE", "FFBC", "FFBCW",
           "FFHL", "FFIC", "FFIN", "FFIV", "FFKT", "FFNM", "FFNW", "FFWM", "FGEN", "FHCO", "FIBK", "FINL", "FISH",
           "FISI", "FISV", "FITB", "FITBI", "FIVE", "FIVN", "FIZZ", "FLAT", "FLDM", "FLEX", "FLIC", "FLIR", "FLL",
           "FLML", "FLWS", "FLXN", "FLXS", "FMB", "FMBH", "FMBI", "FMER", "FMI", "FMNB", "FNBC", "FNFG", "FNGN", "FNHC",
           "FNJN", "FNLC", "FNRG", "FNSR", "FOLD", "FOMX", "FONE", "FONR", "FORD", "FORM", "FORR", "FORTY", "FOSL",
           "FOX", "FOXA", "FOXF", "FPRX", "FPXI", "FRAN", "FRBA", "FRBK", "FRED", "FREE", "FRGI", "FRME", "FRP", "FRPH",
           "FRPHV", "FRPT", "FRSH", "FSAM", "FSBK", "FSBW", "FSC", "FSCFL", "FSFG", "FSFR", "FSGI", "FSLR", "FSNN",
           "FSRV", "FSTR", "FSYS", "FTCS", "FTD", "FTEK", "FTGC", "FTHI", "FTLB", "FTNT", "FTR", "FTSL", "FTSM", "FUEL",
           "FULL", "FULLL", "FULT", "FUNC", "FUND", "FV", "FWM", "FWP", "FWRD", "FXCB", "FXEN", "FXENP", "GABC", "GAI",
           "GAIA", "GAIN", "GAINO", "GAINP", "GALE", "GALT", "GALTU", "GALTW", "GAME", "GARS", "GASS", "GBCI", "GBDC",
           "GBIM", "GBLI", "GBNK", "GBSN", "GCBC", "GCVRZ", "GDEF", "GENC", "GENE", "GEOS", "GERN", "GEVA", "GEVO",
           "GFED", "GFN", "GFNCP", "GFNSL", "GGAC", "GGACR", "GGACU", "GGACW", "GGAL", "GHDX", "GIFI", "GIGA", "GIGM",
           "GIII", "GILD", "GILT", "GK", "GKNT", "GLAD", "GLADO", "GLBS", "GLBZ", "GLDC", "GLDD", "GLDI", "GLMD",
           "GLNG", "GLPI", "GLRE", "GLRI", "GLUU", "GLYC", "GMAN", "GMCR", "GMLP", "GNBC", "GNCA", "GNCMA", "GNMA",
           "GNMK", "GNTX", "GNVC", "GOGO", "GOLD", "GOMO", "GOOD", "GOODN", "GOODO", "GOODP", "GOOG", "GOOGL", "GPIC",
           "GPOR", "GPRE", "GPRO", "GRBK", "GRFS", "GRID", "GRIF", "GRMN", "GROW", "GRPN", "GRVY", "GSBC", "GSIG",
           "GSIT", "GSM", "GSOL", "GSVC", "GT", "GTIM", "GTIV", "GTLS", "GTWN", "GTXI", "GUID", "GULF", "GULTU", "GURE",
           "GWGH", "GWPH", "GYRO", "HA", "HABT", "HAFC", "HAIN", "HALL", "HALO", "HART", "HAS", "HAWK", "HAWKB", "HAYN",
           "HBAN", "HBANP", "HBCP", "HBHC", "HBIO", "HBK", "HBMD", "HBNC", "HBNK", "HBOS", "HBP", "HCAC", "HCACU",
           "HCACW", "HCAP", "HCBK", "HCCI", "HCKT", "HCOM", "HCSG", "HCT", "HDNG", "HDP", "HDRA", "HDRAR", "HDRAU",
           "HDRAW", "HDS", "HDSN", "HEAR", "HEES", "HELE", "HEOP", "HERO", "HFBC", "HFBL", "HFFC", "HFWA", "HGSH",
           "HIBB", "HIFS", "HIHO", "HIIQ", "HILL", "HIMX", "HKTV", "HLIT", "HLSS", "HMHC", "HMIN", "HMNF", "HMNY",
           "HMPR", "HMST", "HMSY", "HMTV", "HNH", "HNNA", "HNRG", "HNSN", "HOFT", "HOLI", "HOLX", "HOMB", "HOTR",
           "HOTRW", "HOVNP", "HPJ", "HPTX", "HQY", "HRTX", "HRZN", "HSGX", "HSIC", "HSII", "HSKA", "HSNI", "HSOL",
           "HSON", "HSTM", "HTBI", "HTBK", "HTBX", "HTCH", "HTHT", "HTLD", "HTLF", "HTWO", "HTWR", "HUBG", "HURC",
           "HURN", "HWAY", "HWBK", "HWCC", "HWKN", "HYGS", "HYLS", "HYND", "HYZD", "HZNP", "IACI", "IART", "IBB",
           "IBCA", "IBCP", "IBKC", "IBKR", "IBOC", "IBTX", "ICAD", "ICCC", "ICEL", "ICFI", "ICLD", "ICLDW", "ICLN",
           "ICLR", "ICON", "ICPT", "ICUI", "IDCC", "IDRA", "IDSA", "IDSY", "IDTI", "IDXX", "IEP", "IESC", "IEUS",
           "IFAS", "IFEU", "IFGL", "IFNA", "IFON", "IFV", "IGLD", "IGOV", "IGTE", "III", "IIIN", "IIJI", "IILG", "IIN",
           "IIVI", "IKAN", "IKGH", "IKNX", "ILMN", "IMDZ", "IMGN", "IMI", "IMKTA", "IMMR", "IMMU", "IMMY", "IMNP",
           "IMOS", "IMRS", "INAP", "INBK", "INCR", "INCY", "INDB", "INDY", "INFA", "INFI", "INFN", "INGN", "ININ",
           "INNL", "INO", "INOD", "INPH", "INSM", "INSY", "INTC", "INTG", "INTL", "INTLL", "INTU", "INTX", "INVE",
           "INVT", "INWK", "IOSP", "IPAR", "IPAS", "IPCC", "IPCI", "IPCM", "IPDN", "IPGP", "IPHS", "IPKW", "IPWR",
           "IPXL", "IQNT", "IRBT", "IRDM", "IRDMB", "IRDMZ", "IRG", "IRIX", "IRMD", "IROQ", "IRWD", "ISBC", "ISCA",
           "ISHG", "ISIG", "ISIL", "ISIS", "ISLE", "ISM", "ISNS", "ISRG", "ISRL", "ISSC", "ISSI", "ISTR", "ITCI",
           "ITIC", "ITRI", "ITRN", "IVAC", "IVAN", "IXYS", "JACK", "JACQ", "JACQU", "JACQW", "JAKK", "JASN", "JASNW",
           "JASO", "JAXB", "JAZZ", "JBHT", "JBLU", "JBSS", "JCOM", "JCS", "JCTCF", "JD", "JDSU", "JGBB", "JIVE", "JJSF",
           "JKHY", "JMBA", "JOBS", "JOEZ", "JOUT", "JRJC", "JRVR", "JSM", "JST", "JTPY", "JUNO", "JVA", "JXSB", "JYNT",
           "KALU", "KANG", "KBAL", "KBIO", "KBSF", "KCAP", "KCLI", "KE", "KELYA", "KELYB", "KEQU", "KERX", "KEYW",
           "KFFB", "KFRC", "KFX", "KGJI", "KIN", "KINS", "KIRK", "KITE", "KLAC", "KLIC", "KLXI", "KMDA", "KNDI", "KONA",
           "KONE", "KOOL", "KOPN", "KOSS", "KPTI", "KRFT", "KRNY", "KTCC", "KTEC", "KTOS", "KTWO", "KUTV", "KVHI",
           "KWEB", "KYTH", "KZ", "LABC", "LABL", "LACO", "LAKE", "LALT", "LAMR", "LANC", "LAND", "LARK", "LAWS", "LAYN",
           "LBAI", "LBIX", "LBRDA", "LBRDK", "LBRKR", "LBTYA", "LBTYB", "LBTYK", "LCNB", "LCUT", "LDRH", "LDRI", "LE",
           "LECO", "LEDS", "LEVY", "LEVYU", "LEVYW", "LFUS", "LFVN", "LGCY", "LGCYO", "LGCYP", "LGIH", "LGND", "LHCG",
           "LIME", "LINC", "LINE", "LION", "LIOX", "LIQD", "LIVE", "LJPC", "LKFN", "LKQ", "LLEX", "LLNW", "LLTC",
           "LMAT", "LMBS", "LMCA", "LMCB", "LMCK", "LMIA", "LMNR", "LMNS", "LMNX", "LMOS", "LMRK", "LNBB", "LNCE",
           "LNCO", "LNDC", "LOAN", "LOCM", "LOCO", "LOGI", "LOGM", "LOJN", "LONG", "LOOK", "LOPE", "LORL", "LOXO",
           "LPCN", "LPHI", "LPLA", "LPNT", "LPSB", "LPSN", "LPTH", "LPTN", "LQDT", "LRAD", "LRCX", "LSBK", "LSCC",
           "LSTR", "LTBR", "LTRE", "LTRPA", "LTRPB", "LTRX", "LTXB", "LULU", "LUNA", "LVNTA", "LVNTB", "LWAY", "LXRX",
           "LYTS", "MACK", "MAG", "MAGS", "MAMS", "MANH", "MANT", "MAR", "MARA", "MARK", "MARPS", "MASI", "MAT", "MATR",
           "MATW", "MAYS", "MBCN", "MBFI", "MBFIP", "MBII", "MBLX", "MBRG", "MBSD", "MBTF", "MBUU", "MBVT", "MBWM",
           "MCBC", "MCBK", "MCEP", "MCGC", "MCHP", "MCHX", "MCOX", "MCRI", "MCRL", "MCUR", "MDAS", "MDCA", "MDCO",
           "MDIV", "MDLZ", "MDM", "MDRX", "MDSO", "MDSY", "MDVN", "MDVXU", "MDWD", "MDXG", "MEET", "MEIL", "MEILW",
           "MEILZ", "MEIP", "MELA", "MELI", "MELR", "MEMP", "MENT", "MEOH", "MERC", "MERU", "METR", "MFI", "MFLX",
           "MFNC", "MFRI", "MFRM", "MFSF", "MGCD", "MGEE", "MGI", "MGIC", "MGLN", "MGNX", "MGPI", "MGRC", "MGYR",
           "MHGC", "MHLD", "MHLDO", "MICT", "MICTW", "MIDD", "MIFI", "MIK", "MIND", "MINI", "MITK", "MITL", "MKSI",
           "MKTO", "MKTX", "MLAB", "MLHR", "MLNK", "MLNX", "MLVF", "MMAC", "MMLP", "MMSI", "MMYT", "MNDL", "MNDO",
           "MNGA", "MNKD", "MNOV", "MNRK", "MNRO", "MNST", "MNTA", "MNTX", "MOBI", "MOBL", "MOCO", "MOFG", "MOKO",
           "MOLG", "MOMO", "MORN", "MOSY", "MPAA", "MPB", "MPEL", "MPET", "MPWR", "MRCC", "MRCY", "MRD", "MRGE", "MRKT",
           "MRLN", "MRNS", "MRTN", "MRTX", "MRVC", "MRVL", "MSBF", "MSCC", "MSEX", "MSFG", "MSFT", "MSG", "MSLI",
           "MSON", "MSTR", "MTBC", "MTEX", "MTGE", "MTGEP", "MTLS", "MTRX", "MTSC", "MTSI", "MTSL", "MTSN", "MU",
           "MULT", "MVIS", "MWIV", "MXIM", "MXWL", "MYGN", "MYL", "MYOS", "MYRG", "MZOR", "NAII", "NAME", "NANO",
           "NATH", "NATI", "NATL", "NATR", "NAUH", "NAVG", "NAVI", "NBBC", "NBIX", "NBN", "NBS", "NBTB", "NBTF", "NCIT",
           "NCLH", "NCMI", "NCTY", "NDAQ", "NDLS", "NDRM", "NDSN", "NECB", "NEO", "NEOG", "NEON", "NEOT", "NEPT",
           "NERV", "NETE", "NEWP", "NEWS", "NEWT", "NFBK", "NFEC", "NFLX", "NGHC", "NGHCP", "NHTB", "NICE", "NICK",
           "NILE", "NKSH", "NKTR", "NLNK", "NLST", "NMIH", "NMRX", "NNBR", "NPBC", "NPSP", "NRCIA", "NRCIB", "NRIM",
           "NRX", "NSEC", "NSIT", "NSPH", "NSSC", "NSTG", "NSYS", "NTAP", "NTCT", "NTES", "NTGR", "NTIC", "NTK", "NTLS",
           "NTRI", "NTRS", "NTRSP", "NTWK", "NUAN", "NURO", "NUTR", "NUVA", "NVAX", "NVCN", "NVDA", "NVDQ", "NVEC",
           "NVEE", "NVEEW", "NVFY", "NVGN", "NVMI", "NVSL", "NWBI", "NWBO", "NWBOW", "NWFL", "NWLI", "NWPX", "NWS",
           "NWSA", "NXPI", "NXST", "NXTD", "NXTDW", "NXTM", "NYMT", "NYMTP", "NYMX", "NYNY", "OBAS", "OBCI", "OCC",
           "OCFC", "OCLR", "OCLS", "OCRX", "OCUL", "ODFL", "ODP", "OFED", "OFIX", "OFLX", "OFS", "OGXI", "OHAI", "OHGI",
           "OHRP", "OIIM", "OKSB", "OLBK", "OLED", "OMAB", "OMCL", "OMED", "OMER", "OMEX", "ONB", "ONCY", "ONEQ",
           "ONFC", "ONNN", "ONTX", "ONTY", "ONVI", "OPB", "OPHC", "OPHT", "OPOF", "OPTT", "OPXA", "ORBC", "ORBK",
           "OREX", "ORIG", "ORIT", "ORLY", "ORMP", "ORPN", "ORRF", "OSBC", "OSBCP", "OSHC", "OSIR", "OSIS", "OSM",
           "OSN", "OSTK", "OSUR", "OTEL", "OTEX", "OTIC", "OTIV", "OTTR", "OUTR", "OVAS", "OVBC", "OVLY", "OVTI",
           "OXBR", "OXBRW", "OXFD", "OXGN", "OXLC", "OXLCN", "OXLCO", "OXLCP", "OZRK", "PAAS", "PACB", "PACW", "PAGG",
           "PAHC", "PANL", "PARN", "PATIV", "PATK", "PAYX", "PBCP", "PBCT", "PBHC", "PBIB", "PBIP", "PBMD", "PBPB",
           "PBSK", "PCAR", "PCBK", "PCCC", "PCH", "PCLN", "PCMI", "PCO", "PCOM", "PCRX", "PCTI", "PCTY", "PCYC", "PCYG",
           "PCYO", "PDBC", "PDCE", "PDCO", "PDEX", "PDFS", "PDII", "PDLI", "PEBK", "PEBO", "PEGA", "PEGI", "PEIX",
           "PENN", "PENX", "PEOP", "PERF", "PERI", "PERY", "PESI", "PETM", "PETS", "PETX", "PFBC", "PFBI", "PFBX",
           "PFIE", "PFIN", "PFIS", "PFLT", "PFMT", "PFPT", "PFSW", "PGC", "PGNX", "PGTI", "PHII", "PHIIK", "PHMD",
           "PICO", "PIH", "PINC", "PKBK", "PKOH", "PKT", "PLAB", "PLAY", "PLBC", "PLCE", "PLCM", "PLKI", "PLMT", "PLNR",
           "PLPC", "PLPM", "PLTM", "PLUG", "PLUS", "PLXS", "PMBC", "PMCS", "PMD", "PME", "PMFG", "PNBK", "PNFP", "PNNT",
           "PNQI", "PNRA", "PNRG", "PNTR", "PODD", "POOL", "POPE", "POWI", "POWL", "POZN", "PPBI", "PPC", "PPHM",
           "PPHMP", "PPSI", "PRAA", "PRAH", "PRAN", "PRCP", "PRFT", "PRFZ", "PRGN", "PRGNL", "PRGS", "PRGX", "PRIM",
           "PRKR", "PRLS", "PRMW", "PROV", "PRPH", "PRQR", "PRSC", "PRSS", "PRTA", "PRTK", "PRTO", "PRTS", "PRXI",
           "PRXL", "PSAU", "PSBH", "PSCC", "PSCD", "PSCE", "PSCF", "PSCH", "PSCI", "PSCM", "PSCT", "PSCU", "PSDV",
           "PSEC", "PSEM", "PSIX", "PSMT", "PSTB", "PSTI", "PSTR", "PSUN", "PTBI", "PTBIW", "PTC", "PTCT", "PTEN",
           "PTIE", "PTLA", "PTNR", "PTNT", "PTRY", "PTSI", "PTX", "PULB", "PVTB", "PVTBP", "PWOD", "PWRD", "PWX",
           "PXLW", "PZZA", "QABA", "QADA", "QADB", "QAT", "QBAK", "QCCO", "QCLN", "QCOM", "QCRH", "QDEL", "QGEN",
           "QINC", "QIWI", "QKLS", "QLGC", "QLIK", "QLTI", "QLTY", "QLYS", "QNST", "QQEW", "QQQ", "QQQC", "QQQX",
           "QQXT", "QRHC", "QRVO", "QSII", "QTEC", "QTNT", "QTNTW", "QTWW", "QUIK", "QUMU", "QUNR", "QURE", "QVCA",
           "QVCB", "QYLD", "RADA", "RAIL", "RAND", "RARE", "RAVE", "RAVN", "RBCAA", "RBCN", "RBPAA", "RCII", "RCKY",
           "RCMT", "RCON", "RCPI", "RCPT", "RDCM", "RDEN", "RDHL", "RDI", "RDIB", "RDNT", "RDUS", "RDVY", "RDWR",
           "RECN", "REDF", "REFR", "REGI", "REGN", "REIS", "RELL", "RELV", "REMY", "RENT", "REPH", "RESN", "REXI",
           "REXX", "RFIL", "RGCO", "RGDO", "RGDX", "RGEN", "RGLD", "RGLS", "RGSE", "RIBT", "RIBTW", "RICK", "RIGL",
           "RITT", "RITTW", "RIVR", "RJET", "RLJE", "RLOC", "RLOG", "RLYP", "RMBS", "RMCF", "RMGN", "RMTI", "RNA",
           "RNET", "RNST", "RNWK", "ROBO", "ROCK", "ROIA", "ROIAK", "ROIC", "ROIQ", "ROIQU", "ROIQW", "ROKA", "ROLL",
           "ROSE", "ROSG", "ROST", "ROVI", "ROYL", "RP", "RPRX", "RPRXW", "RPRXZ", "RPTP", "RPXC", "RRD", "RRGB",
           "RRST", "RSTI", "RSYS", "RTGN", "RTIX", "RTK", "RTRX", "RUSHA", "RUSHB", "RUTH", "RVBD", "RVLT", "RVNC",
           "RVSB", "RWLK", "RXDX", "RXII", "RYAAY", "SAAS", "SABR", "SAEX", "SAFM", "SAFT", "SAGE", "SAIA", "SAJA",
           "SAL", "SALE", "SALM", "SAMG", "SANM", "SANW", "SANWZ", "SAPE", "SASR", "SATS", "SAVE", "SBAC", "SBBX",
           "SBCF", "SBCP", "SBFG", "SBGI", "SBLK", "SBLKL", "SBNY", "SBNYW", "SBRA", "SBRAP", "SBSA", "SBSI", "SBUX",
           "SCAI", "SCHL", "SCHN", "SCLN", "SCMP", "SCOK", "SCON", "SCOR", "SCSC", "SCSS", "SCTY", "SCVL", "SCYX",
           "SEAC", "SEED", "SEIC", "SEMI", "SENEA", "SENEB", "SEV", "SFBC", "SFBS", "SFLY", "SFM", "SFNC", "SFST",
           "SFXE", "SGBK", "SGC", "SGEN", "SGI", "SGMA", "SGMO", "SGMS", "SGNL", "SGNT", "SGOC", "SGRP", "SGYP",
           "SGYPU", "SGYPW", "SHBI", "SHEN", "SHIP", "SHLD", "SHLDW", "SHLM", "SHLO", "SHOO", "SHOR", "SHOS", "SHPG",
           "SIAL", "SIBC", "SIEB", "SIEN", "SIFI", "SIFY", "SIGA", "SIGI", "SIGM", "SILC", "SIMG", "SIMO", "SINA",
           "SINO", "SIRI", "SIRO", "SIVB", "SIVBO", "SIXD", "SKBI", "SKIS", "SKOR", "SKUL", "SKYS", "SKYW", "SKYY",
           "SLAB", "SLCT", "SLGN", "SLM", "SLMAP", "SLMBP", "SLP", "SLRC", "SLTC", "SLVO", "SLXP", "SMAC", "SMACR",
           "SMACU", "SMBC", "SMCI", "SMED", "SMIT", "SMLR", "SMMF", "SMPL", "SMRT", "SMSI", "SMT", "SMTC", "SMTP",
           "SMTX", "SNAK", "SNBC", "SNC", "SNCR", "SNDK", "SNFCA", "SNHY", "SNMX", "SNPS", "SNSS", "SNTA", "SOCB",
           "SOCL", "SODA", "SOFO", "SOHO", "SOHOL", "SOHOM", "SOHU", "SONA", "SONC", "SONS", "SORL", "SOXX", "SP",
           "SPAN", "SPAR", "SPCB", "SPDC", "SPEX", "SPHS", "SPIL", "SPKE", "SPLK", "SPLS", "SPNC", "SPNS", "SPOK",
           "SPPI", "SPPR", "SPPRO", "SPPRP", "SPRO", "SPRT", "SPSC", "SPTN", "SPU", "SPWH", "SPWR", "SQBG", "SQBK",
           "SQI", "SQNM", "SQQQ", "SRCE", "SRCL", "SRDX", "SREV", "SRNE", "SRPT", "SRSC", "SSB", "SSBI", "SSFN", "SSH",
           "SSNC", "SSRG", "SSRI", "SSYS", "STAA", "STB", "STBA", "STBZ", "STCK", "STEM", "STFC", "STKL", "STLD",
           "STLY", "STML", "STMP", "STNR", "STPP", "STRA", "STRL", "STRM", "STRN", "STRS", "STRT", "STRZA", "STRZB",
           "STX", "STXS", "SUBK", "SUMR", "SUNS", "SUPN", "SURG", "SUSQ", "SUTR", "SVA", "SVBI", "SVVC", "SWHC", "SWIR",
           "SWKS", "SWSH", "SYBT", "SYKE", "SYMC", "SYMX", "SYNA", "SYNC", "SYNL", "SYNT", "SYPR", "SYRX", "SYUT",
           "SZMK", "SZYM", "TACT", "TAIT", "TAPR", "TASR", "TAST", "TATT", "TAX", "TAXI", "TAYD", "TBBK", "TBIO", "TBK",
           "TBNK", "TBPH", "TCBI", "TCBIL", "TCBIP", "TCBIW", "TCBK", "TCCO", "TCFC", "TCPC", "TCRD", "TCX", "TDIV",
           "TEAR", "TECD", "TECH", "TECU", "TEDU", "TENX", "TERP", "TESO", "TESS", "TFM", "TFSC", "TFSCR", "TFSCU",
           "TFSCW", "TFSL", "TGA", "TGE", "TGEN", "TGLS", "TGTX", "THFF", "THLD", "THOR", "THRM", "THRX", "THST",
           "THTI", "TICC", "TIGR", "TILE", "TINY", "TIPT", "TISA", "TITN", "TIVO", "TKAI", "TKMR", "TLF", "TLMR",
           "TLOG", "TNAV", "TNDM", "TNGO", "TNXP", "TOPS", "TORM", "TOUR", "TOWN", "TQQQ", "TRAK", "TRCB", "TRCH",
           "TREE", "TRGT", "TRIB", "TRIL", "TRIP", "TRIV", "TRMB", "TRMK", "TRNS", "TRNX", "TROV", "TROVU", "TROVW",
           "TROW", "TRS", "TRST", "TRTL", "TRTLU", "TRTLW", "TRUE", "TRVN", "TSBK", "TSC", "TSCO", "TSEM", "TSLA",
           "TSRA", "TSRE", "TSRI", "TSRO", "TST", "TSYS", "TTEC", "TTEK", "TTGT", "TTHI", "TTMI", "TTOO", "TTPH", "TTS",
           "TTWO", "TUBE", "TUES", "TUSA", "TVIX", "TVIZ", "TWER", "TWIN", "TWMC", "TWOU", "TXN", "TXRH", "TYPE",
           "TZOO", "UACL", "UAE", "UBCP", "UBFO", "UBIC", "UBNK", "UBNT", "UBOH", "UBSH", "UBSI", "UCBA", "UCBI",
           "UCFC", "UCTT", "UDF", "UEIC", "UEPS", "UFCS", "UFPI", "UFPT", "UG", "UGLD", "UHAL", "UIHC", "ULBI", "ULTA",
           "ULTI", "ULTR", "UMBF", "UMPQ", "UNAM", "UNB", "UNFI", "UNIS", "UNTD", "UNTY", "UNXL", "UPI", "UPIP", "UPLD",
           "URBN", "URRE", "USAK", "USAP", "USAT", "USATP", "USBI", "USCR", "USEG", "USLM", "USLV", "USMD", "USTR",
           "UTEK", "UTHR", "UTIW", "UTMD", "UTSI", "UVSP", "VA", "VALU", "VALX", "VASC", "VBFC", "VBIV", "VBLT", "VBND",
           "VBTX", "VCEL", "VCIT", "VCLT", "VCSH", "VCYT", "VDSI", "VECO", "VGGL", "VGIT", "VGLT", "VGSH", "VIA",
           "VIAB", "VIAS", "VICL", "VICR", "VIDE", "VIDI", "VIEW", "VIIX", "VIIZ", "VIMC", "VIP", "VIRC", "VISN",
           "VIVO", "VLCCF", "VLGEA", "VLTC", "VLYWW", "VMBS", "VNDA", "VNET", "VNOM", "VNQI", "VNR", "VNRAP", "VNRBP",
           "VNRCP", "VOD", "VOLC", "VONE", "VONG", "VONV", "VOXX", "VPCO", "VRA", "VRML", "VRNG", "VRNGW", "VRNS",
           "VRNT", "VRSK", "VRSN", "VRTA", "VRTB", "VRTS", "VRTU", "VRTX", "VSAR", "VSAT", "VSCI", "VSCP", "VSEC",
           "VSTM", "VTAE", "VTHR", "VTIP", "VTL", "VTNR", "VTSS", "VTWG", "VTWO", "VTWV", "VUSE", "VVUS", "VWOB", "VWR",
           "VXUS", "VYFC", "WABC", "WAFD", "WAFDW", "WASH", "WATT", "WAVX", "WAYN", "WB", "WBA", "WBB", "WBKC", "WBMD",
           "WDC", "WDFC", "WEBK", "WEN", "WERN", "WETF", "WEYS", "WFBI", "WFD", "WFM", "WGBS", "WHF", "WHFBL", "WHLM",
           "WHLR", "WHLRP", "WHLRW", "WIBC", "WIFI", "WILC", "WILN", "WIN", "WINA", "WIRE", "WIX", "WLB", "WLBPZ",
           "WLDN", "WLFC", "WLRH", "WLRHU", "WLRHW", "WMAR", "WMGI", "WMGIZ", "WOOD", "WOOF", "WPCS", "WPPGY", "WPRT",
           "WRES", "WRLD", "WSBC", "WSBF", "WSCI", "WSFS", "WSFSL", "WSTC", "WSTG", "WSTL", "WTBA", "WTFC", "WTFCW",
           "WTSL", "WVFC", "WVVI", "WWD", "WWWW", "WYNN", "XBKS", "XCRA", "XENE", "XENT", "XGTI", "XGTIW", "XIV",
           "XLNX", "XLRN", "XNCR", "XNET", "XNPT", "XOMA", "XONE", "XOOM", "XPLR", "XRAY", "XTLB", "XXIA", "YDIV",
           "YDLE", "YHOO", "YNDX", "YOD", "YORW", "YPRO", "YRCW", "YY", "Z", "ZAGG", "ZAZA", "ZBRA", "ZEUS", "ZFGN",
           "ZGNX", "ZHNE", "ZINC", "ZION", "ZIONW", "ZIONZ", "ZIOP", "ZIV", "ZIXI", "ZLTQ", "ZN", "ZNGA", "ZSPH", "ZU",
           "ZUMZ"]


class Importer:

    def __init__(self) -> None:
        super().__init__()
        self.minute_count = 0
        self.daily_count = 0
        self.db = create_db_connection()
        self.api = alpha.AlphaVantage('ULDORYWPDU2S2E6X')
        # yM2zzAs6_DxdeT86rtZY
        # TX1OLY36K73S9MS9
        # I7RUE3LA4PSXDJU6
        # ULDORYWPDU2S2E6X


    def import_one(self, sym):
        if stock_collection(self.db, False).count({SYMBOL_KEY: sym}) > 0:
            print('Found object with symbol ', sym)
        else:
            print('Didnt find object with symbol ', sym)
            raw_json = self.api.data_raw(sym).json(object_pairs_hook=self.remove_dots)
            keys = list(raw_json.keys())
            if len(keys) < 2:
                print('Symbol ', sym, 'not existing in alpha vantage')
                return
            time_series_key = keys[1]
            time_series = raw_json[time_series_key]
            time_series[SYMBOL_KEY] = sym
            stock_collection(self.db, False).insert(time_series)
            self.minute_count = self.minute_count + 1
            self.daily_count = self.daily_count + 1
            if self.daily_count >= API_MAX_DAILY:
                raise Exception('Maximum api calls per day reached.')
            if self.minute_count >= API_MAX_PER_MINUTE_CALLS:
                print('Sleeping.')
                time.sleep(60)
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


if __name__ == "__main__":
    imp = Importer()
    imp.import_all(SYMBOLS)