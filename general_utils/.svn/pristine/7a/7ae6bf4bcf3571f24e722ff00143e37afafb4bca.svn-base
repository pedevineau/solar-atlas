'''
Created on Mar 17, 2013

@author: tomas
'''


import psycopg2

from general_utils import db_utils

from general_utils.basic_logger import make_logger
logger = make_logger(__name__)
import logging
logging.getLogger().setLevel(logging.DEBUG)



#out table initialisation
def out_tab_init(dsn, output_table, output_table_descr, output_fields):
    outfields_allowed=["date","time","datetime","slot","ghi_c", "ghi", "ghi_cor", "dni_c", "dni", "dni_cor", "cli", "ci", "ci_flag", "ktm", "lb", "lbland", "lbclass"]
    outfields_allowed_params={ "siteid": ["integer NOT NULL", ""],\
                            "date": ["date NOT NULL", ""],\
                            "time": ["time NOT NULL",""],\
                            "datetime": ["timestamp", ""],\
                            "slot": ["integer",""],\
                            "ghi_c": ["real", "global horizontal clear sky irradiance"],\
                            "ghi": ["real", "global horizontal irradiance"],\
                            "ghi_cor": ["real", "global horizontal irradiance, Perez ranking correction (depricated)"],\
                            "dni_c": ["real", "direct normal clear sky irradiance"],\
                            "dni": ["real", "direct normal irradiance"],\
                            "dni_cor": ["real", "direct normal irradiance, Perez ranking correction (depricated)"],\
                            "cli": ["real", "spectral cloud index"],\
                            "ci": ["real", "cloud index"],\
                            "ci_flag": ["integer", "cloud index flag: 0-no (h0 <0), 1 - from data, 2 - interpolated, 3 - extrapolated, 4 - inter/extra-polated more than one hour"],\
                            "ktm": ["real", "clearsky index"],\
                            "lb": ["real", "lower bound"],\
                            "lbland": ["real", "lower bound without snow"],\
                            "lbclass": ["integer", "lower bound calss: 0-UNKNOWN, 1-LAND, 2-SNOW, 3-LANDSNOW"],\
                            }

    #create list of required fields
    fields_required = ["siteid", "date", "time", "datetime", "slot"]
    for fld in output_fields:
        if (fld not in fields_required) and (fld in outfields_allowed):
            fields_required.append(fld)
    
    #create new if not exist
    out_tab_exist = db_utils.db_dtable_exist(dsn, output_table)
    if not(out_tab_exist):
        success = db_create_out_table(dsn, output_table, output_table_descr, fields_required, outfields_allowed_params)
        if success:
            logger.info( "Create output table: success" ) 
        else:
            logger.error( "Create output table: failure" ) 
            return False

    #check for table fields
    table_flds = db_utils.db_table_get_fields(dsn, output_table)
    
    missing_flds = []
    for fld in fields_required:
        try:
            table_flds.index(fld)
        except:
            missing_flds.append(fld)
    
    for fld in missing_flds:
        logger.debug( "adding field %s to existing table:"% (fld) ) 
        type, descr = outfields_allowed_params[fld]
        
        #add field
        query = "ALTER TABLE \"" + output_table + "\" ADD COLUMN "+fld+" "+ type
        conn = psycopg2.connect(dsn)
        curs = conn.cursor()
        try:
            curs.execute(query)
        except:
            logger.error( "Unable to execute the query, exiting. " + query) 
            conn.close()
            return False
        conn.commit()
        
        #add comment
        query = "COMMENT ON COLUMN \"" + output_table + "\"."+fld+" IS '" + descr + "'"
        try:
            curs.execute(query)
        except:
            logger.error( "Unable to execute the query, exiting. " + query) 
        conn.commit()
        conn.close()
    return True



#create table for outputs
def db_create_out_table(dsn, dbtable, dbtable_desc='NULL', fields_required=[], outfields_allowed_params={}):
    #check db connection
    try:
        conn = psycopg2.connect(dsn)
    except:
        logger.error( "Unable to connect to the database, exiting") 
        return False

    curs = conn.cursor()

    #create table
    fld_str = ""
    separator=""
    for col in fields_required:
        type, comm = outfields_allowed_params[col]
        fld_str += separator+ col+" "+type
        separator=", "
    
    query = "CREATE TABLE \"" + dbtable + "\" ("+fld_str+")"
    #print query
    try:
        curs.execute(query)
    except:
        logger.error( "Unable to execute the query, exiting. " + query) 
        conn.close()
        return False
    conn.commit()

    #add column coments
    for col in fields_required:
        type, comm = outfields_allowed_params[col]
        if comm == "":
            continue
        query = "COMMENT ON COLUMN \"" + dbtable + "\"." + col + " IS '" + comm + "'"
        #print query
        try:
            curs.execute(query)
        except:
            logger.error( "Unable to execute the query, exiting. " + query) 
        conn.commit()

    #add table comment
    if dbtable_desc != 'NULL':
        query = "COMMENT ON TABLE \"" + dbtable + "\" IS '" + dbtable_desc + "'"
        #print query
        try:
            curs.execute(query)
        except:
            logger.error( "Unable to execute the query, exiting. " + query) 
        conn.commit()

    conn.close()
    return True


#drop existing data from output table
def table_drop_data(dsn, out_table, site_id, minslot, maxslot, minD, maxD):
    try:
        conn = psycopg2.connect(dsn)
    except:
        logger.error(  "Unable to connect to the database, exiting.")
        return False

    curs = conn.cursor()
    query = "DELETE FROM \"" + out_table + "\" WHERE siteid=" + str(site_id) + " AND date>='" + str(minD) + "' AND date<='" + str(maxD) + "' AND slot>=" + str(minslot) + " AND slot<=" + str(maxslot)
    try:
        curs.execute(query)
    except:
        logger.error( "Unable to execute the query, skipping."+query)
        conn.close()
        return False

    conn.commit()
    conn.close()
    return True

def output_table_indexes(dsn, aoutput_table, indx_fields=("date", "datetime", "siteid")):
    indexes={}
    for fld in indx_fields:
        db_indx=aoutput_table + "_"+fld+"_idx"
        db_query = "CREATE INDEX " + db_indx + " ON " + aoutput_table + " ("+fld+")"
        indexes[db_indx] = db_query
    return db_utils.db_create_indexes(dsn, indexes)


def output_table_init(dsn, aoutput_table, aoutput_table_descr, aoutput_fields):
    #check output table and remove indexes
    out_table_exists=db_utils.db_dtable_exist(dsn,aoutput_table)
    if not out_table_exists:
        result = out_tab_init(dsn, aoutput_table, aoutput_table_descr, aoutput_fields)
        if not result:
            logger.error( "cannot check/create output table, exit")
            return False
    #remove indexes
    indexes=db_utils.db_table_get_indexes(dsn, aoutput_table)
    db_utils.db_remove_indexes(dsn, indexes)
    return True
