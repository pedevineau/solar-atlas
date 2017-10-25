###################################
# ## INFO to archive :

# 1. philosophical problem - read CF1.6 only OR compatibility read OR ALL read but You must know the config??
# recommendation: read --> CF1.6 only, readCompatible --> swiss knife, LOGS obligatory warning that it is used!
#   in readCompatible - possible to set own name for any dimension + bounds for it
# DECISION to archive: Only CF 1.6 and small subset of future non-CF1.6 compliant files (forecasts), the goal of this
#  lib IS NOT to be practical SWISS KNIFE for lots of history.
#  => tests for nclib: himawari .nc files
# DECISION 2: 2015-12-18 primarily for us, use CF1.6 where it helps us only

# 2. API - Explicit is better, but too explicit is verbose - the reasonable line in one case is freaky in another
# recommendation: 2 ways of API - dict verbose (too explicit) + class style compact (fogged / hidden / shortcut)
# DECISION - dimension name should be given explicitly

# 3. Continuous / discrete = center(pixel) / boundaries / cell_boundaries
# as parameters - each typical dimension will
# DECISION: Should be explicit switch, but with defaults

# 4. FULL names required: no lat, but latitude - the lat, latit, ... aliased needs to be handled internally
# DECISION: this is internal convention

# 5. Do we handle "projected_x_coordinate" attribute in any named dimension? Or do we REQUIRE exact dimension name?
# recommendation - if used object in compatibility read - perform alias name checking + search for
#  "projected_x_coordinate" / y counterpart; in strict mode, only check attributes!
# DECISION: "projected_x_coordinate" / y are treated in creation of .nc file ONLY, no need to check in reading

# 6. Checking units? This makes zicher
# DECISION: Only time dimensions make internal checks <-- to ask when implement and test again

# 7. DECISION: REUSE templates

# 8. DECISION: TimeZone: UTC hard given --> convert to it internally

# 9. Add interpolation
# DECISION: DONE

# 10. How to handle UNLIMITED? (AOD prefix)

# 11. meteorologist: at the end of time they are aggregating (avg) - TODO: get the .nc file example
# 12. nclib created files should be readable by GDAL - This is practical requirement # TODO: write test for it

# all the demonstration in dict style - versatile, VERBOSE, explicit, for modelling primarily

# 13. no leap year ? - NOT NOW - maybe in later versions

class DataSet:
    pass
ds = DataSet("file.nc")
ds.metainfo  # fill Value, dataType
# time variable --> extent

ds.read({
    "dfb":{"start":0, "end":20, "step":2, "beginning_day":"1900-01-01", "end_inclusive":True, "start_inclusive":True},
    "dfb":{"start":0, "end":20, "step":2, "indices":True},
    "dfb":{"enumeration":(0,3,1,5,6,5,4,4), },
    "dfb":{"enumeration":(0,3,1,5,6,5,4,4), "indices":True},
    "dfb":{"beginning_day":"1900-01-01"},
    ## CONVENTION = defaults
    # DFB is "centerpixeled" ONLY - explicitly (we need to avoid time interpolation)
    # default start = start dfb dimension in file (required when creating!)
    # default end = end dfb dimension in file (required when creating!)
    # default step = when reading - step dfb dimension in file
    # default beginning_day - autodetect from file, OR our default when creating = gregorian "1900-01-01"
    # default indices = False - we try to be as explicit as possible
    # default enumeration is None - if None, enumeration is inactive (it is NON-OPTIMAL!!)
    # default end_inclusive - TRUE

    # if given and file has _time_ dimension --> ? raise Exception ? uderstand it

    # if not given at all: ? take ALL / raise exception / ignore the dimension (NO) ?


    "slot":{"start":0, "end":5,  "step":2},
    "slot":{"enumeration":(1,2,5,9,12)},
    # convention = defaults
    # slot is "centerpixeled" ONLY - explicitly (we need to avoid time interpolation)
    # default start = start dfb dimension in file (required when creating!)
    # default end = end dfb dimension in file (required when creating!)
    # default step = when reading - step dfb dimension in file
    # default enumeration is None - if None, enumeration is inactive (it is NON-OPTIMAL!!)


    # .nc with time dimension ONLY = external 3D with datetime!
    # For our files --> raise Exception
    # because time would require slotmapping knowledge --> ? dict {slotNo: "HH:mm"} (search for desc in the dimension?)
    "time" :{"ISO":"datetimeFrom/datetimeTo/step", "approximate":True},  # this will enumeration the absolute dfb in autoadjusted calendar in .nc file
    "time" :{"start":"2010-01-01 23:50", "end": "2015-01-01 00:00", "step": "1H", "approximate":False},
    "time": {"start":datetime("2010-01-01 23:50"), "end":datetime("2015-01-01 00:00"), "step":timedelta("1H"), "approximate":False},  # this will enumerate the absolute dfb in autoadjusted calendar in .nc file
    # how to give one day? From 2010-01-01 to 2010-01-01 - gives all the day? = inclusive end
    # approximate - "the call for problem" - How is returned the info about the deviation from REQUESTED time?


    # space coordinates - iba pre lat /lon kodovane
    "latitude"    :{"start":90.,   "end":-84.,"step":-0.1},
    "longitude"   :{"start":-179.9,"end":0.,  "step":None},  # autoadjusted from file
    # "longitudeCircular" :{"start":0.,    "end":360.,"step":None},  # autoadjusted from file
    # pravdepodobne zbytocne Circular - vsetko a vzdy sa da robit cez lon - circular vies autodetektnut a abstrahovat
    # ZATIAL len bounds - ALE UVAZ, ZE v buducnosti sa prida centerpixel prepocitavanie


    # space coordinates - iba pre kvazi m kodovane
    "x" :{"start":-2000000.,"end":1500000.,"step":2000.},
    "y" :{"start":0.       ,"end":500000., "step":None},  # autoadjusted step
    # ma sa to volat proj_x alebo proj_y
    "projection_x_coordinate" :{"start":-2000000.,"end":1500000.,"step":2000.},
    "projection_y_coordinate" :{"start":0.       ,"end":500000., "step":None},  # autoadjusted step
    # Diskusia -ako to nazvat, pozri CF1.6

    "x" :{"start":-2000000.,"end":1500000.,"step":2000., "meaning": "center", "interpolation":"bilinear" },
    "x" :{"start":-2000000.,"end":1500000.,"step":2000., "meaning": "bound", "interpolation":"nearest_neighbour"},
    "x" :{"start":-2000000.,"end":1500000.,"step":2000., "meaning": "index", "interpolation":"bilinear"},
    # default end inclusive


    # indexovane = cez pixle iterujeme
    # space coordinates - column / row
    # compatibility functionality
    "col" :{"start":0 ,"end":2500,  "step":1},
    "row" :{"start":0 ,"end":10000, "step":None},  # autoadjusted
    # default end inclusive


    "img_x" :{"start":0 ,"end":2500,  "step":1},
    "img_y" :{"start":0 ,"end":10000, "step":None},  # autoadjusted
    # ma byt img_x a img_y


    "forecast":{"start":0., "end":12., "step":0.5},  # v hodinach?
})


ads =AdvancedDataSet("folder", "file_pattern*.nc", template, allow_missing=True)
ads.metadata  # returns {} header parts which are in common for all required files
ads.read()  # like DataSet

ds = DataSet("filename.nc")


# možnosť:
#
# start - ako pre range() - ak None, od začiatku dimenzie v súbore
# end - ako pre range() - ak None, do konca dimenzie v súbore
# step - ako pre range() - ak None, step ako je v súbore
#
# enumeration -tuple - vymenované hodnoty - bud je enumerate alebo start, end a step
#
# indices -True/False - či ideme od začitku dimenzie v súbore = či ideme po indexe začínajúcom 0
#
# cell_boundary -True/False - či sme zadávali začiatky pixlov
#
# beginning_day - date - pre dfb - odkedy začína DFB


# WAY OF DICTIONARY (+)simple (+)flexible (-)too verbose
# (-)so flexible, user needs to read really ALL the documentation to know how to write working dict

ds.read({  # this method is like the previous one, but space coordinates are refferenced
    # ak by sme šli po 10-tich dňoch v 61-dňovom archíve - iterujeme relatívne
     "dfb" :{"start":0, "end":5,   "step":1, "beginning_day":"1980-01-01", "end_inclusive":True, "start_inclusive":True},  # this will read on (0,2,4)-th values of dfb in file
     "dfb" :{"start":0, "end":5,   "step":2, "beginning_day":"1900-01-01"},  # this takes days 1900-01-01, 1900-01-03, 1900-01-05
     # CF1.6 podporuje celú plejádu kalendárov - no leap, gregorian, ... hours since 1980-01-01
     "dfb" :{"start":0, "end":None,"step":2, "beginning_day":None},  # autoadjust to the calendar in .nc file, then take 0-th, 2-nd and 4-th ... to the end ; pri CReate použi defaultny calendar
     "dfb" :{"enumeration":(10,12,15,18,1,0),  "beginning_day":None},  # this will enumerate the absolute dfb in autoadjusted calendar in .nc file

     # dfb indexy -relatívne v súbore
     "dfb" :{"enumeration":(10,12,15,18,1,0), "indices":True},  # this will enumerate the absolute dfb in autoadjusted calendar in .nc file

     # slot setting style
     "slot":{"start":0, "end":5,  "step":2},
     "slot":{"enumeration":(1,2,5,9,12)},

     # toto ma byť použité LEN pre súbor, ktorý je 3D cez datetime! Naše DFB+ slot súbory musia pri tomto vyhlásiť chybu!
     # time - vyžaduje poznanie slotmappingu --> cez dict {slotNo: "HH:mm"} (aj nezadané, pohľadá v desc v súbore)
     "time" :{"ISO":"datetimeFrom/datetimeTo/step", "approximate":"1H"},  # this will enumerate the absolute dfb in autoadjusted calendar in .nc file
     "time" :{"start":"2010-01-01 23:50", "end": "2015-01-01 00:00", "step": timedelta("1H"), "approximate":False},
     "time": {"start":datetime("2010-01-01 23:50"), "end":datetime("2015-01-01 00:00"), "step":timedelta("1H"), "approximate":False},  # this will enumerate the absolute dfb in autoadjusted calendar in .nc file
     # ako zadat od 2010-01-01 do 2010-01-01 (vrati cely den) ?
     # approximate - NIE

     # space coordinates - iba pre lat /lon kodovane
     "latitude"    :{"start":90.,   "end":-84.,"step":-0.1},
     "longitude"   :{"start":-179.9,"end":0.,  "step":None},  # autoadjusted from file
     # longitudeCircular --> AUTODETECT!
     # "longitudeCircular" :{"start":0.,    "end":360.,"step":None},  # autoadjusted from file

     # space coordinates - iba pre kvazi m kodovane
     "x" :{"start":-2000000.,"end":1500000.,"step":2000.},
     "y" :{"start":0.       ,"end":500000., "step":None},  # autoadjusted step
     # ma sa to volat proj_x alebo proj_y
     "projection_x_coordinate" :{"start":-2000000.,"end":1500000.,"step":2000.},
     "projection_y_coordinate" :{"start":0.       ,"end":500000., "step":None},  # autoadjusted step
     # Diskusia -ako to nazvat, pozri CF1.6

     "x" :{"start":-2000000.,"end":1500000.,"step":2000., "meaning": "center", "spatial_interpolation": "nearest_neighbour"},
     "x" :{"start":-2000000.,"end":1500000.,"step":2000., "meaning": "bound", "spatial_interpolation": "bilinear"},
     "x" :{"start":-2000000.,"end":1500000.,"step":2000., "meaning": "index", "spatial_interpolation": "average"},


     # space coordinates - column / row
     # compatibility functionality
     "col" :{"start":0 ,"end":2500,  "step":1},
     "row" :{"start":0 ,"end":10000, "step":None},  # autoadjusted

     "img_x" :{"start":0 ,"end":2500,  "step":1},
     "img_y" :{"start":0 ,"end":10000, "step":None},  # autoadjusted
     # ma byt img_x a img_y

    "forecast":{"start":0., "end":12., "step":0.5},
    # v hodinach?  # TODO: neskôr zisti

     # DECISION - ma sa zadať explicitne názov dimenzie?
})


# WAY OF KWARGS, THEN DICTIONARY (+)simpler (-)still too verbose
# (-)user needs to read all the documentation to know the keys
ds.read(
     dfb={"start":0, "end":5,  "step":2, "beginning_day":"1980-01-01"},  # this will read on (0,2,4)-th values of dfb in file
     slot={"start":0, "end":5,  "step":2},
     time={"ISO":"datetimeFrom/datetimeTo/step", "approximate":True, "slotmap":{1:"00:00",144:"23:50",}},  # this will enumerate the absolute dfb in autoadjusted calendar in .nc file
     lat        ={"start":90.,   "end":-84.,"step":-0.1},
     lon        ={"start":-179.9,"end":0.,  "step":None},  # autoadjusted
     img_x={"start":-2000000.,"end":1500000.,"step":2000.},
     img_y={"start":0.       ,"end":500000., "step":None},  # autoadjusted step
     col={"start":0 ,"end":2500,  "step":1},
     row={"start":0 ,"end":10000, "step":None},  # autoadjusted
     forecast={"start":0., "end":12., "step":0.5}  # v hodinach?
)

# WAY OF KWARGS AND MICRO-OBJECTS (+)possible default values (+)very able (enough possibilities) (-)objects
# (+) users reads only doc for what interests him
class Dimension:
    def __init__(self):
        pass
    def __iter__(self):
        if self.enumeration is not None:
            return self.enumeration
        elif self.start is not None and self.end is not None:
            return range(self.start,self.end, self.step)
        else:
            raise Exception("Bad definition of DFB")


# microobject creation OR (generator) function - ALL ARE ITERABLE!
ds.read('ghi',
     dfb(start=0, end=5, step=2, beginning_day="1980-01-01", indices=True),  # this will read on (0,2,4)-th values of dfb in file
     dfb[0:5:2, {"beginning_day":"1980-01-01","indices":False}],  # possibly AND/OR numpy alike slice / range!
     dfb[0:5, {"indices":True}],
     dfb[:],

     slot(start=0, end=5, step=2),
     slot(0,5,2),  # possibly AND/OR range alike!
     slot[0:5:2],  # possibly AND/OR numpy alike slice / range!
     slot[:],

     time(ISO="datetimeFrom/datetimeTo/step", approximate=True, slotmap={1:"00:00",144:"23:50",}),  # this will enumerate the absolute dfb in autoadjusted calendar in .nc file
     time(start="2010-01-01 23:50", end="2015-01-01 00:00", step="1H", slotmap={1:"00:00",144:"23:50",}, approximate=False),  # this will enumerate the absolute dfb in autoadjusted calendar in .nc file
     time(start=datetime("2010-01-01 23:50"), end=datetime("2015-01-01 00:00"), step=timedelta("1H"), slotmap={1:"00:00",144:"23:50",}, approximate=False),  # this will enumerate the absolute dfb in autoadjusted calendar in .nc file

     lat        (start=90.,    end=-84., step=-0.1),
     lon        (start=-179.9, end=0.,   step=None),  # autoadjusted
     lat        (90.,  -84.,  -0.1),
     lat        [90. : -84. : -0.1],
     img_x(start=-2000000., end=1500000., step=2000.),
     img_y(start=0.       , end=500000.,  step=None),  # step autoadjusted from file!
     img_x[-2000000. : 1500000. : 2000.],
     col(start=0 , end=2500, step=1),
     row(start=0 , end=10000, step=None),  # autoadjusted

     forecast(start=0., end=12., step=0.5),
     forecast[0. : 12. : 0.5],  # v hodinach?
     forecast[:],

     extent={'time':{},}
)


ds.read('ghi', extent = {'time': timeseries('2000-01-01', '2000-02-31'),
                         'latitude': coords(14, 14),
                         'longitude': bounds(0,5)},
                         forecast_period=coords(1, 72))

# WAY OF NAMEDTUPLES (+)mandatory simple (-)NO default values = MANDATORY explicit signature
# (-)only one signature for ALL cases = mandatory extra verbosity

# Like the above, but ALL variables must be given = AUTOADJUST NEEDS explicit None,
# default dictionary setting needs explicit empty dict {}


##
# 1. máme podporovať ECMWF forecasty - absolútne nie CF1.6 compliant?
#   /neptun/home0/data/ecmwf_forecasts/*
# DECISION : vid "máme podporovať CF1.6? + ak vieš, čo je v tom súbore a menovite si to vypýtaš, má to fungovať!
# 2. odporúčanie založiť konvenciu pre forecasty nasledovne:
#  dimenzie: time, forecast_period => zase len 4D (tak odporúča Braňo)
# 3. time=slot, dfb, time=timestamp in double -- vždy čítaj units, býva v tom confusion
#  pozri nc_headers.odt v tomto adresári
# 4. AOD sú príklad UNLIMITED dát --> postupne prichádzajú (aerosoly)
# 5. "lat":range(1,2,3) <-- ZLE! Lebo neoptimalne!



######################################
# Example of real reading, writing and creating

extentCreate, extentWrite, extentRead = {}, {}, {}
class DataSetException(Exception):
    pass
class MemoryError(Exception):  # chces nacitat tolko, volneho mas tolko # = cekovanie volnej RAM
    pass

# lets create an .nc file
# DataSet --> context manager - enter, exit,
ds = HeterogenousDataSet("folder/regexp*.nc", "folder2/regexp*.nc", "fol")  # lazy init; no declaration of segmentation! uses metadata, can interpolate # LATER implementation
# nespájať directory / path s regex

ds = HeterogenousDataSet(DataSet(), DataSet())  # lazy init; can interpolate, works on DataSet # Priority
ds = DataSet("filename.nc")  # lazy init - nothing to do
ds = DataSet(folders=["foldername"], "prefix", segmentation=[Segmentation_Space, Segmentation_Time])  # lazy init - nothing to do
ds = ClimatologyDataSet(folders=["foldername"], "prefix", segmentation=[Segmentation_Space, Segmentation_Time])  # lazy init - nothing to do

# predpoklad: dataset je vo vnútri homogénny = rovnaké rozlíšenie, ...
ds.metadata()  # returns {} - no data in nonexistent file
# NEW: Tomáš: viac segmentácií použiť len pre 1 space segmentáciu a rôzne časové segmentácie (operačný archív a ročné archívy)
# Sú susedné files

ds.create(extent=extentCreate)  # extent see below
ds.metadata()  # returns dict alike extentCreate, with fill Value, dataTypes, time variables used, all extents

try:
    with ds as local_ds:
        local_ds.write("my_variable", my_variable_ndarray, extent=extentWrite)  # write the variable with name "my_variable"
except DataSetException as e:
    print("Problem writing into .nc file: ", str(e))
ds.metadata()  # returns almost extentCreate + data about my_variable

# UNIVERSAL read method - no compatibility checking - that is done in create ONLY
ndArray = ds.read("my_variable", asked_extent=extentRead)
ds.read("dni", asked_extent={"dfb":{}, "slot":{}, "latitude":{}, "logitude":{}})
ndArray = ds.read("my_variable")  # reads the proxy of the
# suggestion - metadata are always available
ndArray.metadata  # <-- this will be hardly added into ndArray, it will carry the ds.metadata() dict


ads = AdvancedDataSet("/folder/AOD*regexp*.nc")
ads.read("ghi", asked_extent={"time":{"ISO": "8601"},
                        "latitude":{"start":0, "end":89, "step":0.2},
                        "longitude":{"start":0, "end":89, "step":0.2}
                              })
ads.write("ghi", extent={"time":{"ISO":"8601"},
                        "latitude":{"start":0, "end":89, "step":0.2},
                        "longitude":{"start":0, "end":89, "step":0.2}
                       }, ndArray)
# joins the files together to produce the ndarray


extent = {  # This have defaults:
"dfb" :{"start":None, "end":None, "step":None, "enumeration":None, "calendar":None, "end_inclusive":True,
        "start_inclusive":True, "interpolation":None, "meaning":"bounds / indices / centers"},
# DEFAULTS:
# start = None = beginning of the dimension
# end = None = end of dimension
# step = None = iterate over the dimension from start to end
# enumeration = None = inactive
# calendar = None = autodetect calendar from file (example of custom calendar: "1900-01-01", date("1900-01-01")) # TODO: Pozri ako sa volá atribút v CF1.6
# end_inclusive = True = end inclusive
# start_inclusive = True = start inclusive
# meaning = "center" = we consider the values representing the center of the dfb = no interpolation;
    # this is FORCED for dfb; this can change in future


"slot" :{"start":None, "end":None, "step":None, "enumeration":None, "end_inclusive":True,
         "start_inclusive":True, "interpolation":None, "meaning":"bounds / indices / centers"},
# DEFAULTS:
# start = None = beginning of the dimension
# end = None = end of dimension
# step = None = iterate over the dimension from start to end
# enumeration = None = inactive
# end_inclusive = True = end inclusive
# start_inclusive = True = start inclusive
# meaning = "center" = we consider the values representing the center of the slot = no interpolation;
    # this is FORCED for slot; this can change in future


"time" :None,  # time is inactive in default
"time" :{"ISO":None, "start":None, "end":None, "step":None, "enumeration":None, "beginning_day":None, "end_inclusive":True,
         "start_inclusive":True, "interpolation":None, "meaning":"bounds / indices / centers"},
# DEFAULTS:
# ISO = None = inactive; ISO8601 string - e.g. "datetimeFrom/datetimeTo/step"
# start = None = beginning of the dimension
# end = None = end of dimension
# step = None = iterate over the dimension from start to end; can be also timedelta
# enumeration = None = inactive
# beginning_day = None = autodetect calendar from file (example of custom calendar: "1900-01-01", date("1900-01-01"))
# end_inclusive = True = end inclusive
# start_inclusive = True = start inclusive
# meaning = "center" = we consider the values representing the center of the time = no interpolation;
    # this is FORCED for time; this can change in future

# BEHAVIOUR: For .nc files using (date)time ONLY. For files with dfb + slot --> raise Exception


"latitude / longitude / x / y / proj_x / proj_y / img_x / img_y / row / col" :
    {"start":None, "end":None, "step":None, "enumeration": None, "end_inclusive":True,
     "start_inclusive":True, "interpolation":None, "meaning":"bounds / indices / centers"},
# DEFAULTS:
# start = None = beginning of the dimension
# end = None = end of dimension
# step = None = iterate over the dimension from start to end; can be also datetime
# enumeration = None = inactive
# beginning_day = None = autodetect calendar from file (example of custom calendar: "1900-01-01", date("1900-01-01"))
# end_inclusive = True = end inclusive
# start_inclusive = True = start inclusive
# meaning = "center" = we consider the values representing the center of the pixel TODO:
# interpolation = None = if interpolation needed (other step then files own OR files with not the same resolution)
    # Exception is raised; possible: "nearest_neighbour", "bilinear"
    # If the interpolations for both space dimensions are not the same, Error is raised

# BEHAVIOUR: For .nc files using lat/latit/latitude and lon/logit/logitude ONLY. For other files --> raise Exception
# longitudeCircular --> AUTODETECT!

"forecast" :{"start":None, "end":None, "step":None, "enumeration": None, "end_inclusive":True,
             "start_inclusive":True, "interpolation":None, "meaning":"bounds / indices / centers"},
# DEFAULTS:
# start = None = beginning of the dimension
# end = None = end of dimension
# step = None = iterate over the dimension from start to end # zapina interpolaci - IBA ked treba
    # pri create je v pohode 1 pri dfb, slot, img pixel, img coordinates
# enumeration = None = inactive ; cannot be used with bounds
# beginning_day = None = autodetect calendar from file (example of custom calendar: "1900-01-01", date("1900-01-01"))
# end_inclusive = True = end inclusive
# start_inclusive = True = start inclusive
# meaning = "center" = we consider the values representing the center of the forecast = no interpolation;
    # this is FORCED for forecast; this can change in future

# hour units? That is used by ECMWF
# DECISION: ONLY in hours!
}


# Where to use CF1.6 & where not? - Should by thoroughly documented in nclib API2 also and enforce it
# CF1.6 should be considered + enforced only when creating new .nc

# Tomas DECISION: UNLIMITED ONLY in "time" dimension through datetime (no "dfb" /"slot") - in that case bounds does not
# exist = creating, writing & reading through enum OR all OR interval OF indices or the values
# DECISION Tomas: UNLIMITED = can mean also the non-monotonic series; the step surely is non-regular => no bounds set =
# cannot be used with "meaning":"bounds", just "centers" or "indices"

# implicit / explicit way is the matter of ongoing decision

# the ndArray is proxied / lazy in itself? Or should they be eager?
# OWN DECISION: the lazyness is a virtue!

# DECISION Tomáš: indices - on NON-SEGMENTED variable only!
# DECISION Tomáš: minimize mount of default headers
# DECISION Tomáš: time - requires knowing user requiring the existing file content
# DECISION Tomáš: cumulative create (!) - all is in that one method, which immediatelly crashes if something not OK

# TODO: Inner convention: Ma sa to volat proj_x alebo proj_y - JANO (Braňo none decision)
# TODO: ako zadat od 2010-01-01 do 2010-01-01 (vrati cely den) ?

# TODO: "opajcni"Petove Templaty + segmentaciu

## GLOBALS CF1.6 definition
CF16_DEFAULT_GLOBALS_DEFINITION = {
    "Conventions" : "CF-1.6",
    "title" : None,  # hardly requested from user!
    "source" : None,  # hardly requested from user!
    "history" : "Tue May 26 16:48:56 2015 file created",  # autogenerated always after edit!
    "institution" : "GeoModel Solar s.r.o." ,
    "references" : "http://geomodelsolar.eu/",
    "comments" : None,  # hardly requested from user

    # "license" : "...",  # To docs: V outputoch von by malo byť...
}


## DEFAULT FOR DIMENSIONS
# units will be checked against udunits standard http://www.unidata.ucar.edu/software/udunits/udunits.txt
# "nv" dimension is automatic; also *_bonds is automatic for continuous dimension interpretation
DEFAULT_TIME_DEFINITION = {
    "continuous" : True,
    "units" : "hours since 1900-01-01 00:00:0.0",
    "long_name" : "time",
    "datatype" : "f8",  # CRITICAL!  # TODO: How about dates ? is the conversion numpy specific or netCDF supports it?
    "standard_name" : "time",  # TODO: musí byť práve z tabuľky standard name - ak tam je.. inak stačí iba jedno: long_name OR standard name
    # "extent" : None,  # if None --> unlimited, {"start":, "end":, "step":, "enumeration":, "ISO":}  # moved to createDimension
    # "axis" : "T",  # TODO: for outer usage - this is at least recommended
}
# TODO: add compatibility level for old numpy (inner conversion to "f8")

DEFAULT_DFB_DEFINITION = {
    "continuous" : False,
    "long_name" : "Day from beginning",
    "calendar" : "gregorian",  # in dfb only ; OR "standard"
    "units" : "days since 1980-01-01",  # or "m" or "degrees east" or  "hours since 1900-01-01 00:00:0.0" OR "W m**-2"
    "datatype" : "i4",  # CRITICAL!
    "standard_name" : "",  # TODO: musí byť práve z tabuľky standard name - ak tam je.. inak stačí iba jedno: long_name OR standard name
}
DEFAULT_SLOT_DEFINITION = {
    "continuous" : False,
    # "extent" : range(1, 144+1),
    "datatype" : "u2",  # CRITICAL!
    "long_name" : "Time slot number",
    "units" : "slot number",
    # THIS CANNOT BE SET! "standard_name" : "",  # TODO: standard name tabulka nepozná; stačí long_name
}
# if defined forecast dimension, time dimension standard name must be changed to "forecast_reference_time"
DEFAULT_FORECAST_PERIOD_DEFINITION = {
    "continuous" : False,
    "long_name" : "forecast dimension",
    "standard_name" : "forecast_period",  # TODO: musí byť práve z tabuľky standard name - ak tam je.. inak stačí iba jedno: long_name OR standard name
    "datatype" : "i2",
    "units" : "minute",
}
DEFAULT_LATITUDE_DEFINITION = {
    "continuous" : True,
    "units" : "degrees north",
    "valid_range" : (-90., 90.),
    "long_name" : "latitude",  # je celkom voľné - vhodné zadať
    "standard_name" : "latitude",  # TODO: musí byť práve z tabuľky standard name - ak tam je.. inak stačí iba jedno: long_name OR standard name
    "datatype" : "f8",  # CRITICAL!

    # "bounds": None,  # will be autocreated if continuous
}
DEFAULT_LONGITUDE_DEFINITION = {
    "continuous" : True,
    "units" : "degrees east", # degrees west means circular
    "valid_range" : (-180., 180.),
    "long_name" : "longitude",  # je celkom voľné - vhodné zadať
    "standard_name" : "longitude",  # TODO: musí byť práve z tabuľky standard name - ak tam je.. inak stačí iba jedno: long_name OR standard name
    "datatype" : "f8",  # CRITICAL!

    # "bounds": None,  # will be autocreated if continuous
}
# TODO: a ďalšie


## PROJECTION
# will be generated based on other variables: if lat/lon, latlon grid
# will be used; OR projected grid will be used; raw elsewhere
# AK nebude LATLON -- NEPOUŽIŤ!  Inak áno. Otravuje to život
DEFAULT_COORDINATE_REFERENCE_SYSTEM_LATLON = {
    "long_name" : "Parameters of WGS 1984 datum",
    "grid_mapping_name" : "latitude_longitude",
    "longitude_of_prime_meridian" : 0.,
    "semi_major_axis" : 6378137.,
    "inverse_flattening" : 298.257223563,
    "datatype" : "i2",  # Required
}
DEFAULT_COORDINATE_REFERENCE_SYSTEM_RAW = {
    "long_name" : "Unprojected data",
    "grid_mapping_name" : "line_column",
    "datatype" : "i2",  # Required
}
# in this case the subsatellite point "longitude_of_projection_origin" MUST be given!
DEFAULT_COORDINATE_REFERENCE_SYSTEM_VERTICAL_PERSPECTIVE = {
    "long_name" : "Parameters of the projection",
    "grid_mapping_name" : "vertical_perspective",  # "geostationary" ?
    "latitude_of_projection_origin" : 0.,
    "longitude_of_projection_origin" : 140.,
    "perspective_point_height" : 35785831,
    "false_easting" : 0.,
    "false_northing" : 0.,
    "crs_wkt" : "PROJCS[\"unnamed\",\n    GEOGCS[\"unnamed ellipse\",\n        DATUM[\"unknown\",\n            SPHEROID[\"unnamed\",6378169,295.4880658970008]],\n        PRIMEM[\"Greenwich\",0],\n        UNIT[\"degree\",0.0174532925199433]],\n    PROJECTION[\"Geostationary_Satellite\"],\n    PARAMETER[\"central_meridian\",140],\n    PARAMETER[\"satellite_height\",35785831],\n    PARAMETER[\"false_easting\",0],\n    PARAMETER[\"false_northing\",0],\n    UNIT[\"Meter\",1]]",
    "datatype" : "i2",  # Required

    "GeoTransform" : (-5500000., 2000., 0., 5500000., 0., -2000.) ,  # TODO: odporúčané Jano
}
# For special OUTPUT cases (conferences, ...), this may be additional fields provided
# maybe this could be for inspiration / template to be near the hand
# TODO: a ďalšie - podľa projekcie


# DECISION Tomáš : must be able to set ANY parameters to data variable - but not mandatory anytime
DEFAULT_GHI = {
    # mandatory
    "_FillValue" : None,  # CRITICAL! Required --> will fill "fill_value" also - in netCDF4 i-face
    "units" : None ,  # CRITICAL! Required checked against udunits --> http://www.unidata.ucar.edu/software/udunits/udunits.txt
    # "standard_name" : None,  # e.g. "air_temperature", checked against valid standard names
    "long_name" : None,  # e.g. "Global Horizontal Irradiation (GHI) average daily sum", required from user
    "grid_mapping" : "coordinate_reference_system",
    "cell_methods" : None,  # e.g. "time: mean (interval: 1 day comment: hourly sum averages) latitude: longitude: mean"
    "datatype" : None,  # CRITICAL!! Required
    "dimensions" : (),

    # optional
    "zlib" : True,  # will be autodetected from complevel
    "complevel" : 4,  # Ignored if zlib=False; sets also "compression" : True in nc headers
    "shuffle" : True,  # This significantly improves compression. Default is True. Ignored if zlib=False.
    "fletcher32" : False,  # Fletcher32 HDF5 checksum algorithm | TODO ????
    "contiguous" : False,  # If the optional keyword contiguous is True, the variable data is stored contiguously on disk. Default False. Setting to True for a variable with an unlimited dimension will trigger an error.
    "chunksizes" : None,  # chunksizes cannot be set if contiguous=True. # TODO: rychlost citania?
    "endian" : 'native',
    "least_significant_digit" : None,

    "scale_factor" : None,  # Only for some special cases, when it can shrik the data
    "add_offset" : None,  # Only for some special cases, when it can shrik the data
}

## DECISION Tomas: Must be simple usage, it may be blackbox
class _Segmentation:
    """Carries strict logic about the segmentation"""
    # možné riešenie cez flagy space a time segmentation
    def filename_part(self, extent):
        """Creates filenames for given extent"""
        raise Exception("not implemented")
    def partial_regexp(self, extent):
        """Returns the regexp to use for filenames"""
        raise Exception("not implemented")
    def filename_extent_part(self, extent):
        """Returns the filename part as key and extent part as value"""
        raise Exception("not implemented")
class _TimeSegmentation():
    pass
class _SpaceSegmentation(_Segmentation):
    pass
class SpaceSegmentation5to5(_SpaceSegmentation):
    pass
class SpaceSegmentation1to1(_SpaceSegmentation):
    pass
class OperationalTimeSegmentation(_TimeSegmentation):
    def get_file_name_part(self, extent):
        names = set()
        time = extent.get("time", None)
        if "enumeration" in time:

            return names
        dfb = extent.get("dfb", None)
    def get_partial_regexp(self, extent):
        pass

## How to create
ds.create(file_pattern="haha",
          segmentation=(SpaceSegmentation5to5(),OperationalTimeSegmentation()),
          globals_definition=dict(CF16_DEFAULT_GLOBALS_DEFINITION, {"title": "haha"}),  # joining the dictionaries
          dimension_definition={"dfb":{"extent":{"enumeration":range(13589, 14589)}},
                                "longitude":{"extent":{"start":0, "end":180}},  # extent musí byť vždy
                                "slot":{"extent":{"enumeration":range(1,145)}},
                                "latitude": {"extent":{"enumeration":range(1,145)}}},  # empty means default for given name. if name not resolved and insufficient --> ERROR
                                # neexistuje minimum dimenzií
          variables_definition={"ghi":{"_FillValue" : -999, "units": "W m**-2",
                                "long_name" : "GHI","cell_methods" : "time: mean (interval: 1 day comment: hourly sum averages) latitude: longitude: mean",
                               "datatype" : "f4", "dimensions" : ["dfb", "slot", "latitude", "longitude"]},},
          projection=DEFAULT_COORDINATE_REFERENCE_SYSTEM_LATLON,  # default is LATLON - but it is validated. if not latitude/longitude used --> ERROR
          overwrite_level=False,  # if truncate existing files
          )

#e.g.
{"extent":{"ISO":"20161201/20170101/P1M", "end_inclusive":True, "start_inclusive":True}}
{"extent":{"start":datetime.datetime(2016,11,01), "end":datetime.datetime(2016,12,01), "step":"P1M / DAILY / MONTHLY/ YEARLY / 15MINUTES", "enumeration":None,}}


# TODO: ASK: bounds + end_inclusive =True: ak end za začiatkom bounds nejakého pixla, ale jeho koniec nie je pokrytý = VEZME HO
# obdobne pri start_inclusive
# centers + end_inclusive =True: ak stred pixla == end - vezme ho!
# indices + end_inclusive =True: ber AJ end index



# ##############################################
# ### LEVEL 2 nclib API2:
# TODO: climatology --> create for it separate defaults
# TODO: XRAY as support for large nArrays

# 4. Meteorology: TMY - čo tak použiť months since 1900-01-01 ? Priamo ako integer? Alebo aj float s rozumnými bounds?
#   TODO: bude to CF compliant? BUDE!!! http://www.unidata.ucar.edu/software/udunits/udunits.txt
#   TODO: TOMáš Bude to jednoduché na použitie?
#   TODO: vypýtaj TMY príklad meteo dát od Tomáša (nie sú v dátach, čo má Braňo)
# 5. získaj príklady:
#  súbory s meteo dátami, kde sú dáta zarovnané na KONIEC dňa