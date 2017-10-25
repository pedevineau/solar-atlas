#! /usr/bin/env python
# read information from ground mesurements and model output database

import sys
from datetime import timedelta, datetime, time
import psycopg2


from general_utils import daytimeconv_num
from general_utils import db_utils
import numpy


#db_dtables_for_siteID(dsn, ID, table="data_tables") - returns list of data tables for given site ID
#db_dtables_for_siteID_param(dsn, ID, parameter, table="data_tables") - returns list of data tables for given site ID and parameter
#db_dtables_get_siteID_for_dtable(dsn, dtable, table="data_tables") - gets ID of site for given datatable

#db_site_coord(dsn, ID, table="site_geom_wgs84", column="geom") - read coordinates of the site
#db_site_Z(dsn, ID, table="site_geom_wgs84", column="elevation") - read Z coordinate of the site
#db_site_ID_by_name_s(dsn, name_sh, table="sites") - get site site ID by short name
#db_site_name_s_by_ID(dsn, ID, table="sites") - get site short name by site ID
#db_site_name_by_ID(dsn, ID, table="sites") - get site long name by site ID

#db_getdata(dsn, table, column, date_begin, date_end, time_start=time(hour=0,minute=0), time_end=time(hour=23,minute=59), use_flag=True) - reads data from database
#db_getdata_multicol(dsn, table, columns_list, date_begin, date_end, time_start=time(hour=0,minute=0), time_end=time(hour=23,minute=59), use_flag=True) - multicolumn version
#db_getdata_avg(dsn, table, column, date_begin, date_end, time_start, time_end, avg_period) - reads data from database averaged for given period
#db_getdata_forDT(dsn, table, column, datDTlist) - reads data from database for list of datetime values
#db_getdata_day_forD(dsn, table, column, datDlist, use_flag=False) - reads data from database for list of date values
#db_getdata_forDT_ave(dsn, table, column, datDTlist, avg_period) - reads data from database for list of datetime values averaged for given period

#db_getdata_sat(dsn, table, column, siteid, date_begin, date_end, time_start, time_end) # reads dhe sat data from database
#db_getdata_sat_forDT_ave(dsn, table, column, datDTlist, avg_period, siteid, strict_select=False) - reads data from database for list of datetime values averaged for given period, query uses also sideid - multiple sites can be stored in one table
#db_get_site_latlonelev(siteID, dsn, sitetable)
#db_get_site_name_description(siteID, dsn, sitetable)

#get time interval of data stored in datatable - the value is read from description table 
#e.g. dsn="dbname=meteo_sites host=hugin user=tomas", tablename="r_pa_10min", descr_table="data_tables"
def db_dtable_time_step(dsn, tablename, descr_table="data_tables", time_step_column="time_step_min"):
    if (tablename==None) or (tablename==''):
        print "Missing table name ..."
        return (None)
    
    
    if not(db_utils.test_dsn(dsn) and db_utils.db_dtable_exist(dsn, tablename) and db_utils.db_dtable_descr_exist(dsn, tablename,descr_table=descr_table)):
        print "Problem processing time step request"
        return (None)
    conn = psycopg2.connect(dsn)        
    query = "SELECT "+time_step_column+" FROM "+descr_table+" WHERE table_name = '"+tablename+"'"
    curs = conn.cursor()
    try:
        curs.execute(query)
    except:
        print "Unable to query time step, exiting."
        print query
        conn.close()
        return(None)
    
    tab=curs.fetchall()
    conn.close()
    if len(tab)>0:
        time_step=tab[0][0]
        #print time_step, type(time_step)
        tdelta=timedelta(minutes=time_step)
        return (tdelta)
    return (None)


#returns list of data tables for given site ID
def db_dtables_for_siteID(dsn, ID, table="data_tables"):
    tablist=[]
    
    if (ID==None):
        print "Missing site ID ..."
        return (tablist)
    
    if(db_utils.test_dsn(dsn)==False):
        print "Unable to connect to the database, exiting."
        return(None)
    
    conn = psycopg2.connect(dsn)
    curs = conn.cursor()
    query="SELECT table_name FROM "+ table +" WHERE site_uid = '"+ str(ID) +"'"
    try:
        curs.execute(query)
    except:
        print "Unable to execute query, exiting."
        conn.close()
        return(tablist)
    
    tab1 = curs.fetchall()
    conn.close()

    if (len(tab1)==0):
        return()
    
    for row in tab1:
        tablist.append(row[0])
        
    return(tablist)    
        
     
#returns list of data tables for given site ID and parameter
def db_dtables_for_siteID_param(dsn, ID, parameter, table="data_tables"):
    tablist=[]
    
    if (ID==None):
        print "Missing site ID ..."
        return (tablist)
    
    if(db_utils.test_dsn(dsn)==False):
        print "Unable to connect to the database, exiting."
        return(None)
    
    conn = psycopg2.connect(dsn)
    curs = conn.cursor()
    query="SELECT table_name FROM "+ table +" WHERE site_uid = '"+ str(ID) +"' AND parameter = '"+parameter+"'"
    try:
        curs.execute(query)
    except:
        print "Unable to execute query, exiting."
        conn.close()
        return(tablist)
    
    tab1 = curs.fetchall()
    conn.close()

    if (len(tab1)==0):
        return()
    
    for row in tab1:
        tablist.append(row[0])
        
    return(tablist)    
        
        
# gets ID of site for given datatable 
def db_dtables_get_siteID_for_dtable(dsn, dtable, table="data_tables"):
    if (dtable=='') or (dtable==None):
        print "Missing data table ..."
        return (None)
    
    if not(db_utils.test_dsn(dsn)):
        print "Unable to connect to the database, exiting."
        return(None)
    
    conn = psycopg2.connect(dsn)
    curs = conn.cursor()
    query="SELECT site_uid FROM "+ table +" WHERE table_name = '"+ dtable +"'"
    try:
        curs.execute(query)
    except:
        print "Unable to execute site ID query, exiting."
        conn.close()
        return(None)
    tab1 = curs.fetchall()
    conn.close()

    if (len(tab1)==0):
        return(None)
    return(tab1[0][0])    

##read coordinates of the site
#def db_site_coord_DEPRICATED(dsn, ID, table="site_geom_wgs84", column="geom"):
#    if (ID==None):
#        print "Missing site ID ..."
#        return (None)
#    
#    if(db_utils.test_dsn(dsn)==False):
#        print "Unable to connect to the database, exiting."
#        return(datGlist)
#    
#    conn = psycopg2.connect(dsn)
#    curs = conn.cursor()
#    querry="SELECT asText("+ column +") FROM "+ table +" WHERE uid = '"+ str(ID) +"'"
#    try:
#        curs.execute(querry)
#    except:
#        print "Unable to execute the querry, exiting."
#        conn.close()
#        return(None)
#    
#    tab1 = curs.fetchall()
#    conn.close()
#
#    if len(tab1)==0:
#        return (None)
#    else:
#        res = (tab1[0][0]).strip('POINT()').split()
#        if (len(res)==2):
#            return (float(res[0]), float(res[1]))
#    return(None)
def db_sites_by_bbox(dsn, table="site_geom_wgs84", bbox=None, return_coordinates=False):
    if (bbox is None):
        print "Missing bbox ..."
        return (None)
    
    if(db_utils.test_dsn(dsn)==False):
        print "Unable to connect to the database, exiting."
        return(None)
    
    conn = psycopg2.connect(dsn)
    curs = conn.cursor()
    querry="SELECT uid, lon, lat FROM "+ table +" WHERE lon > "+ str(bbox.xmin) +" AND lon < "+ str(bbox.xmax) +" AND lat > "+ str(bbox.ymin) +" AND lat < "+ str(bbox.ymax) 
    try:
        curs.execute(querry)
    except:
        print "Unable to execute the query, exiting."
        conn.close()
        return(None)
    
    tab1 = curs.fetchall()
    conn.close()

    if len(tab1)==0:
        return (None)
    else:
        results=[]
        for i in range(0,len(tab1)):
            res = tab1[i]
            if (len(res)==3):
                if return_coordinates:
                    results.append([int(res[0]),float(res[1]), float(res[2])])
                else:
                    results.append(int(res[0]))
                    
            
    return results

#read coordinates of the site
def db_site_coord(dsn, ID, table="site_geom_wgs84"):
    if (ID==None):
        print "Missing site ID ..."
        return (None)
    
    if(db_utils.test_dsn(dsn)==False):
        print "Unable to connect to the database, exiting."
        return(None)
    
    conn = psycopg2.connect(dsn)
    curs = conn.cursor()
    querry="SELECT lon, lat FROM "+ table +" WHERE uid = '"+ str(ID) +"'"
    try:
        curs.execute(querry)
    except:
        print "Unable to execute the querry, exiting."
        conn.close()
        return(None)
    
    tab1 = curs.fetchall()
    conn.close()

    if len(tab1)==0:
        return (None)
    else:
        res = tab1[0]
        if (len(res)==2):
            return (float(res[0]), float(res[1]))
    return(None)

#read Z coordinate of the site
def db_site_Z(dsn, ID, table="site_geom_wgs84", column="elevation"):
    if (ID==None):
        print "Missing site ID ..."
        return (None)
    
    if(db_utils.test_dsn(dsn)==False):
        print "Unable to connect to the database, exiting."
        return(None)
    
    conn = psycopg2.connect(dsn)
    curs = conn.cursor()
    querry="SELECT "+ column +" FROM "+ table +" WHERE uid = '"+ str(ID) +"'"
    try:
        curs.execute(querry)
    except:
        print "Unable to execute the querry, exiting."
        conn.close()
        return(None)
    
    tab1 = curs.fetchall()
    conn.close()

    if len(tab1)==0:
        return (None)
    else:
        res = (tab1[0][0])
        return (res)


#get site site ID by short name
def db_site_ID_by_name_s(dsn, name_sh, table="sites"):
    if (name_sh==None):
        print "Missing site name ..."
        return (None)
    
    if(db_utils.test_dsn(dsn)==False):
        print "Unable to connect to the database, exiting."
        return(None)
    
    conn = psycopg2.connect(dsn)
    curs = conn.cursor()
    querry="SELECT uid FROM "+ table +" WHERE name_short = '"+ name_sh +"'"
    try:
        curs.execute(querry)
    except:
        print "Unable to execute the querry, exiting."
        conn.close()
        return(None)
    
    tab1 = curs.fetchall()
    conn.close()

    if len(tab1)==0:
        return (None)
    else:
        res = tab1[0][0]
        return int(res)
    return(None)


#get site short name by site ID
def db_site_name_s_by_ID(dsn, ID, table="sites"):
    if (ID==None):
        print "Missing site name ..."
        return (None)
    
    if(db_utils.test_dsn(dsn)==False):
        print "Unable to connect to the database, exiting."
        return(None)
    
    conn = psycopg2.connect(dsn)
    curs = conn.cursor()
    querry="SELECT name_short FROM "+ table +" WHERE UID = '"+ str(ID) +"'"
    try:
        curs.execute(querry)
    except:
        print "Unable to execute the querry, exiting."
        conn.close()
        return(None)
    
    tab1 = curs.fetchall()
    conn.close()
    
    if len(tab1)==0:
        return (None)
    else:
        res = tab1[0][0]
        return (res)
    return(None)


#get site long name by site ID
def db_site_name_by_ID(dsn, ID, table="sites"):
    if (ID==None):
        print "Missing site name ..."
        return (None)
    
    if(db_utils.test_dsn(dsn)==False):
        print "Unable to connect to the database, exiting."
        return(None)
    
    conn = psycopg2.connect(dsn)
    curs = conn.cursor()
    querry="SELECT name FROM "+ table +" WHERE UID = '"+ str(ID) +"'"
    try:
        curs.execute(querry)
    except:
        print "Unable to execute the querry, exiting."
        conn.close()
        return(None)
    
    tab1 = curs.fetchall()
    conn.close()
    
    if len(tab1)==0:
        return (None)
    else:
        res = tab1[0][0]
        return (res)
    return(None)


#reads data from database
#inputs are in datetime.date and datetime.time format !!!
#returns two lists "timelist" in datetime type and requested parameter list
def db_getdata_daily(dsn, table, column, date_begin, date_end, time_start=time(hour=0,minute=0), time_end=time(hour=23,minute=59), use_flag=True):
    datDTlist=[]#daytime list
    datGlist=[]#global list

    if(db_utils.test_dsn(dsn)==False):
        print "Unable to connect to the database, exiting."
        return(datDTlist,datGlist)

    conn = psycopg2.connect(dsn)
    curs = conn.cursor()

    # convert date to datetime - only datetime needs to be indexed
    aDT_begin=datetime.combine(date_begin,time(hour=0,minute=0))
    aDT_end=datetime.combine(date_end,time(hour=23,minute=59))

    query="SELECT date, "+column+" FROM "+ table +" WHERE date >= '"+ str(date_begin) +"' AND date <= '"+ str(date_end) + "'  ORDER BY date"
    if use_flag:
        query="SELECT date, "+column+" FROM "+ table +" WHERE date >= '"+ str(date_begin) +"' AND date <= '"+ str(date_end) +"' AND (flg_"+column+"=1 ) ORDER BY date"
    print query
    try:
        curs.execute(query)
    except:
        print "Unable to execute the query, exiting."
        print sys.exc_info()
        conn.close()
        return(datDTlist,datGlist)

    tab1 = curs.fetchall()

    conn.commit()
    conn.close()

    #convert to list (if the result i not empty)
    if (len(tab1)==0):
        return(datDTlist,datGlist)

    for row in tab1:
        datDTlist.append(datetime.combine(row[0],time(hour=12,minute=0)))
        datGlist.append(row[1])

    return(datDTlist,datGlist)


#reads data from database 
#inputs are in datetime.date and datetime.time format !!!
#returns two lists "timelist" in datetime type and requested parameter list
def db_getdata(dsn, table, column, date_begin, date_end, time_start=time(hour=0,minute=0), time_end=time(hour=23,minute=59), use_flag=True):
    datDTlist=[]#daytime list
    datGlist=[]#global list
    
    if(db_utils.test_dsn(dsn)==False):
        print "Unable to connect to the database, exiting."
        return(datDTlist,datGlist)
    
    conn = psycopg2.connect(dsn)
    curs = conn.cursor()
    
    # convert date to datetime - only datetime needs to be indexed
    aDT_begin=datetime.combine(date_begin,time(hour=0,minute=0))
    aDT_end=datetime.combine(date_end,time(hour=23,minute=59))

    query="SELECT date, time, "+column+" FROM "+ table +" WHERE datetime >= '"+ str(aDT_begin) +"' AND datetime <= '"+ str(aDT_end) +"'  AND time>='"+ str(time_start) +"' AND time <='"+ str(time_end) +"'  ORDER BY date, time"
    if use_flag:
        query="SELECT date, time, "+column+" FROM "+ table +" WHERE datetime >= '"+ str(aDT_begin) +"' AND datetime <= '"+ str(aDT_end) +"'  AND time>='"+ str(time_start) +"' AND time <='"+ str(time_end) +"' AND (flg_"+column+"=1 ) ORDER BY date, time"

    try:
        curs.execute(query)
    except:
        print "Unable to execute the query, exiting."
        print query
        print sys.exc_info()
        conn.close()
        return(datDTlist,datGlist)
    
    tab1 = curs.fetchall()

    conn.commit()
    conn.close()
    
    #convert to list (if the result i not empty)
    if (len(tab1)==0):
        return(datDTlist,datGlist)
    
    for row in tab1:
        datDTlist.append(datetime.combine(row[0],row[1]))
        datGlist.append(row[2])

    return(datDTlist,datGlist)


#reads data from database - multicolumn
#inputs are in datetime.date and datetime.time format !!!
#returns twol lists "timelist" in daytime type and requested parameter list
def db_getdata_multicol(dsn, table, columns_list, date_begin, date_end, time_start=time(hour=0,minute=0), time_end=time(hour=23,minute=59), use_flag=True,datetime_original=False):
    datDTlist=[]#daytime list
    datalist=[]#global list
    
    if len(columns_list)<1:
        print "Empty column list:",columns_list
        return(datDTlist,datalist)
    
    if(db_utils.test_dsn(dsn)==False):
        print "Unable to connect to the database, exiting."
        return(datDTlist,datalist)
    
    conn = psycopg2.connect(dsn)
    curs = conn.cursor()
    
    # convert date to datetime - only datetime needs to be indexed
    aDT_begin=datetime.combine(date_begin,time(hour=0,minute=0))
    aDT_end=datetime.combine(date_end,time(hour=23,minute=59))

        
    col_query=columns_list[0]
    flg_condition=""
    if use_flag:
        flg_condition=" AND (flg_"+columns_list[0]+"=1)"
    if(len(columns_list)>1):
        for i in range(1,len(columns_list)):
            col_query+=(","+columns_list[i])
            if use_flag:
                flg_condition+=" AND (flg_"+columns_list[i]+"=1)"
   
   
   
    if datetime_original:
        datetime_str="datetime_original" 
    else:
        datetime_str="datetime"
    query="SELECT "+datetime_str+", "+col_query+" FROM "+ table +" WHERE "+datetime_str+" >= '"+ str(aDT_begin) +"' AND "+datetime_str+" <= '"+ str(aDT_end) +"'  AND time>='"+ str(time_start) +"' AND time <='"+ str(time_end) +"'" + flg_condition + " ORDER BY date, time"


    try:
        curs.execute(query)
    except:
        print "Unable to execute the query, exiting."
        print query
        print sys.exc_info()
        conn.close()
        return(datDTlist,datalist)
    
    tab1 = curs.fetchall()

    conn.commit()
    conn.close()
    
    #convert to list (if the result is not empty)
    if (len(tab1)==0):
        return(datDTlist,datalist)
    
    for row in tab1:
        datDTlist.append(row[0])
    
    for i in range(0,len(columns_list)):
        col_datalist=[]
        for row in tab1:
            col_datalist.append(row[1+i])
        datalist.append(col_datalist)

    return(datDTlist,datalist)


#reads data from database averaged for given period
#returns two lists "timelist" in daytime type and requested parameter list
def db_getdata_avg(dsn, table, column, date_begin, date_end, time_start, time_end, avg_period, use_flag=True):
    datDTlist=[]#daytime list
    datGlist=[]#global list
    half_period=avg_period/2
    
    #init db connection
    if(db_utils.test_dsn(dsn)==False):
        print "Unable to connect to the database, exiting."
        return(datDTlist,datGlist)
    
    conn = psycopg2.connect(dsn)
    curs = conn.cursor()
    
    #loop through days
    day_increm=timedelta(days=1)
    curr_date=date_begin
    while(curr_date<=date_end):
        #loop through days
        dt_start=datetime.combine(curr_date,time_start)
        dt_end=datetime.combine(curr_date,time_end)
        curr_dt=dt_start
        
        while(curr_dt<=dt_end):
            dt_from=curr_dt-half_period
            dt_to=curr_dt+half_period
            
            querry="SELECT AVG("+column+") FROM "+ table +" WHERE datetime >= '"+ str(dt_from) +"' AND datetime <= '"+ str(dt_to)+"'"
            if use_flag:
                querry="SELECT AVG("+column+") FROM "+ table +" WHERE datetime >= '"+ str(dt_from) +"' AND datetime <= '"+ str(dt_to)+"' AND (flg_"+column+"=1 )"
            try:
                curs.execute(querry)
            except:
                print "Unable to execute the querry, skipping."
                curr_dt+=avg_period
                continue
    
            tab1 = curs.fetchall()
            if (tab1[0][0] != None):
                datDTlist.append(curr_dt)
                datGlist.append(tab1[0][0])
            curr_dt+=avg_period
            #end time loop
            
        curr_date+=day_increm
        #end date loop
            
    #close connection
    conn.commit()
    conn.close()            
    #print str(avg_period),str(time_start+(avg_period/2))
    return(datDTlist,datGlist)


#reads data from database for list of datetime values
#returns one list - requested parameter list aligned to input DT list
def db_getdata_forDT(dsn, table, column, datDTlist, use_flag=True):
    datGlist=[]#global list
    
    if(db_utils.test_dsn(dsn)==False):
        print "Unable to connect to the database, exiting."
        return(datGlist)
    
    conn = psycopg2.connect(dsn)
    curs = conn.cursor()
    
    for aDT in datDTlist:
        query="SELECT "+column+" FROM "+ table +" WHERE datetime = '"+ str(aDT) +"'"
        if use_flag:
            query="SELECT "+column+" FROM "+ table +" WHERE datetime = '"+ str(aDT) +"' AND (flg_"+column+"=1 )"
        
        try:
            curs.execute(query)
        except:
            print "Unable to execute the querry, exiting."
            datGlist.append(None)
            continue
                
        tab1 = curs.fetchall()
        if len(tab1)==0:
            datGlist.append(None)
        else:
            datGlist.append(tab1[0][0])

    conn.commit()
    conn.close()
    return(datGlist)



#reads data from database -assumes daily values in DB
#inputs are in datetime.date !!!
#returns two lists:"dateList" in date type and requested parameter list
def db_getdata_day(dsn, table, column, date_begin, date_end, use_flag=False):
    datDlist=[]#daytime list
    datGlist=[]#global list
    
    if(db_utils.test_dsn(dsn)==False):
        print "Unable to connect to the database, exiting."
        return(datGlist)
    
    conn = psycopg2.connect(dsn)
    curs = conn.cursor()
    
    query="SELECT date, "+column+" FROM "+ table +" WHERE date >= '"+ str(date_begin) +"' AND date <= '"+ str(date_end) +"'"
    if use_flag:
        query="SELECT date, "+column+" FROM "+ table +" WHERE date >= '"+ str(date_begin) +"' AND date <= '"+ str(date_end) +"' AND (flg_"+column+"=1 )"
    
    try:
        curs.execute(query)
    except:
        print "Unable to execute the query:", query
        return(datDlist,datGlist)
        
    tab1 = curs.fetchall()
    conn.commit()
    conn.close()

    if (len(tab1)==0):
        return(datDlist,datGlist)
    
    for row in tab1:
        datDlist.append(row[0])
        datGlist.append(row[1])

    return(datDlist,datGlist)




#reads data from database for list of date values -assumes daily values in DB
#returns one list - requested parameter list aligned to input D list
def db_getdata_day_forD(dsn, table, column, datDlist, use_flag=False):
    datGlist=[]#global list
    
    if(db_utils.test_dsn(dsn)==False):
        print "Unable to connect to the database, exiting."
        return(datGlist)
    
    conn = psycopg2.connect(dsn)
    curs = conn.cursor()
    
    for aD in datDlist:
        querry="SELECT "+column+" FROM "+ table +" WHERE date = '"+ str(aD) +"'"
        if use_flag:
            querry="SELECT "+column+" FROM "+ table +" WHERE date = '"+ str(aD) +"' AND (flg_"+column+"=1 )"
        
        try:
            curs.execute(querry)
        except:
            print "Unable to execute the querry, exiting."
            datGlist.append(None)
            continue
                
        tab1 = curs.fetchall()
        if len(tab1)==0:
            datGlist.append(None)
        else:
            datGlist.append(tab1[0][0])

    conn.commit()
    conn.close()

    return(datGlist)


#reads data from database for list of datetime values averaged for given period, this version is for sat data, and siteid is needed
#returns one list - requested parameter list aligned to input DT list
def db_getdata_sat_forDT_ave(dsn, table, column, datDTlist, avg_period, siteid, strict_select=False):
    half_period=avg_period/2
    datGlist=[]#global list


    #derive period for which to read the data
    aDT_min=datDTlist[0]
    aDT_max=datDTlist[0]
    for aDT in datDTlist:
        aDT_min = min(aDT_min, aDT )
        aDT_max = max(aDT_max, aDT )

    #read all data for period    
    datDTlist_all, datGlist_all = db_getdata_sat(dsn, table, column, siteid, aDT_min.date(), aDT_max.date())
    
    #convert aDT to aDTn - to allow for work in numpy
    datDTnlist_all = []
    for aDT in datDTlist_all:
        datDTnlist_all.append(daytimeconv_num.date2num(aDT))
        
    #make numpy arrays
    aDTn_arr = numpy.array(datDTnlist_all)
    aG_arr = numpy.array(datGlist_all)
    
    for aDT in datDTlist:
        dtn_from=daytimeconv_num.date2num(aDT-half_period)
        dtn_to=daytimeconv_num.date2num(aDT+half_period)
        wh = (aDTn_arr >= dtn_from) & (aDTn_arr <= dtn_to)  
        if wh.sum()>0:
            datGlist.append(aG_arr[wh].mean())
        else:   
            datGlist.append(None)

    return datGlist





#reads data from database for list of datetime values averaged for given period
#returns one list - requested parameter list aligned to input DT list
def db_getdata_forDT_ave(dsn, table, column, datDTlist, avg_period, use_flag=True):
    datGlist=[]#global list
    half_period=avg_period/2

    #derive period for which to read the data
    aDT_min=datDTlist[0]
    aDT_max=datDTlist[0]
    for aDT in datDTlist:
        aDT_min = min(aDT_min, aDT )
    for aDT in datDTlist:
        aDT_max = max(aDT_max, aDT )

#    print 'database read', datetime.now()
    #read all data for period    
    datDTlist_all, datGlist_all = db_getdata(dsn, table, column, aDT_min.date(), aDT_max.date(),use_flag=use_flag)
   
    #convert aDT to aDTn - to allow for work in numpy
    datDTnlist_all = []
    for aDT in datDTlist_all:
        datDTnlist_all.append(daytimeconv_num.date2num(aDT))

    #make numpy arrays
    aDTn_arr = numpy.array(datDTnlist_all)
    aG_arr = numpy.array(datGlist_all)
        
#    print 'harmonization to request data', datetime.now()
    #data are split to smaller chunks to make faster processing (selection)
    calculate_chunk_days=10
    aDTn_min = numpy.floor(daytimeconv_num.date2num(aDT_min))
    chunk_aDTn_min=aDTn_min
    chunk_aDTn_max=aDTn_min+calculate_chunk_days
    wh_chunk = (aDTn_arr>=(chunk_aDTn_min-1)) & (aDTn_arr<(chunk_aDTn_max+1))
    aDTn_arr_chunk=aDTn_arr[wh_chunk]
    aG_arr_chunk=aG_arr[wh_chunk]
    
    for aDT in datDTlist:
        aDTn = daytimeconv_num.date2num(aDT)
        dtn_from=daytimeconv_num.date2num(aDT-half_period)
        dtn_to=daytimeconv_num.date2num(aDT+half_period)
        
        if aDTn > chunk_aDTn_max:
            chunk_aDTn_min +=   calculate_chunk_days
            chunk_aDTn_max +=   calculate_chunk_days
            wh_chunk = (aDTn_arr>=(chunk_aDTn_min-1)) & (aDTn_arr<(chunk_aDTn_max+1))
            aDTn_arr_chunk=aDTn_arr[wh_chunk]
            aG_arr_chunk=aG_arr[wh_chunk]
            
        wh = (aDTn_arr_chunk >= dtn_from) & (aDTn_arr_chunk <= dtn_to)  
        if wh.sum()>0:
            datGlist.append(aG_arr_chunk[wh].mean())
        else:   
            datGlist.append(None)

#    print 'data read finnished', datetime.now()
    return(datGlist)




#reads data from database 
#this version is for sat data, and siteid is needed
#returns two lists - requested parameter list and DT list
def db_getdata_sat(dsn, table, column, siteid, date_begin, date_end, time_start=time(hour=0,minute=0), time_end=time(hour=23,minute=59)):
    datDTlist=[]#daytime list
    datGlist=[]#global list
    
    if not(db_utils.test_dsn(dsn)):
        print "Unable to connect to the database, exiting."
        return(datGlist)
    
    conn = psycopg2.connect(dsn)
    curs = conn.cursor()
    
    query="SELECT date, time, "+column+" FROM "+ table +" WHERE date >= '"+ str(date_begin) +"' AND date <= '"+ str(date_end) +"'  AND time>='"+ str(time_start) +"' AND time <='"+ str(time_end) +"' AND siteid ="+ str(siteid)+"  ORDER BY date, time"
    try:
        curs.execute(query)
    except:
        print "Unable to execute the query, exiting."
        print sys.exc_info()
        conn.close()
        return(datDTlist,datGlist)
    conn.commit()
    tab1 = curs.fetchall()
    conn.close()
    
    #convert to list (if the result if it is not empty)
    if (len(tab1)==0):
        return(datDTlist,datGlist)
    for row in tab1:
        datDTlist.append(datetime.combine(row[0],row[1]))
        datGlist.append(row[2])
    return(datDTlist,datGlist)        



#reads data from database 
#this version is for sat data, and siteid is needed
#returns two lists - requested parameter list and DT list
def db_getdata_sat_multicol(dsn, table, columns_list, siteid, date_begin, date_end, time_start=time(hour=0,minute=0), time_end=time(hour=23,minute=59), use_siteid=True):
    datDTlist=[]#daytime list
    datalist=[]#global list
    
    if len(columns_list)<1:
        print "Empty column list:",columns_list
        return(datDTlist,datalist)
    
    if not(db_utils.test_dsn(dsn)):
        print "Unable to connect to the database, exiting."
        return(None)
    
    conn = psycopg2.connect(dsn)
    curs = conn.cursor()
    
    # convert date to datetime - only datetime needs to be indexed
    aDT_begin=datetime.combine(date_begin,time(hour=0,minute=0))
    aDT_end=datetime.combine(date_end,time(hour=23,minute=59))
    
    col_query=columns_list[0]
    if(len(columns_list)>1):
        for i in range(1,len(columns_list)):
            col_query+=(","+columns_list[i])
    
    if use_siteid:
        query="SELECT datetime, "+col_query+" FROM "+ table +" WHERE datetime >= '"+ str(aDT_begin) +"' AND datetime <= '"+ str(aDT_end) +"'  AND time>='"+ str(time_start) +"' AND time <='"+ str(time_end) +"' AND siteid ="+ str(siteid)+" ORDER BY date, time"
    else:
        query="SELECT datetime, "+col_query+" FROM "+ table +" WHERE datetime >= '"+ str(aDT_begin) +"' AND datetime <= '"+ str(aDT_end) +"'  AND time>='"+ str(time_start) +"' AND time <='"+ str(time_end) +"'  ORDER BY date, time"
        
    try:
        curs.execute(query)
    except:
        print "Unable to execute the query, exiting."
        conn.close()
        print query
        return(datDTlist,datalist)
    conn.commit()
    tab1 = curs.fetchall()
    conn.close()
    
    #convert to list (if the result is not empty)
    if (len(tab1)==0):
        return(datDTlist,datalist)
    
    for row in tab1:
        datDTlist.append(row[0])
    
    for i in range(0,len(columns_list)):
        col_datalist=[]
        for row in tab1:
            col_datalist.append(row[1+i])
        datalist.append(col_datalist)

    return(datDTlist,datalist)


 

#
##reads data from database for list of datetime values averaged for given period, this version is for sat data, and siteid is needed
##returns one list - requested parameter list aligned to input DT list
#def db_getdata_sat_forDT_ave_old(dsn, table, column, datDTlist, avg_period, siteid, strict_select=False):
#    datGlist=[]#global list
#    half_period=avg_period/2
#    
#    if(db_utils.test_dsn(dsn)==False):
#        print "Unable to connect to the database, exiting."
#        return(datGlist)
#    
#    conn = psycopg2.connect(dsn)
#    curs = conn.cursor()
#    
#    for aDT in datDTlist:
#        dt_from=aDT-half_period
#        dt_to=aDT+half_period
#        
#        query="SELECT AVG("+column+"), count(*) FROM "+ table +" WHERE datetime >= '"+ str(dt_from) +"' AND datetime <= '"+ str(dt_to)+"' AND siteid="+ str(siteid)
#        #print query
#        try:
#            curs.execute(query)
#        except:
#            print "Unable to execute the querry, exiting.\n"+query
#            datGlist.append(None)
#            continue
#                
#        tab1 = curs.fetchall()
#        if len(tab1)==0:
#            datGlist.append(None)
#        else:
#            value, count=tab1[0]
#            if not(strict_select):
#                datGlist.append(value)
#            else:    
#                if (count >2):
#                    datGlist.append(value)
#                else:
#                    datGlist.append(None)
#    conn.commit()
#    conn.close()
#    return(datGlist)


#reads data from database for list of datetime values averaged for given period, this version is for sat data, and siteid is needed
#returns list of X column values - requested parameter list aligned to input DT list
def db_getdata_sat_forDT_ave_multicol(dsn, table, columns, datDTlist, avg_period, siteid, strict_select=False):
    datoutlist=[]#global list
    half_period=avg_period/2
    
    if(db_utils.test_dsn(dsn)==False):
        print "Unable to connect to the database, exiting."
        return(None)
    
    conn = psycopg2.connect(dsn)
    curs = conn.cursor()
    
    unsucces_result_line=[]
    for i in range(1,len(columns)):
        unsucces_result_line.append(None)
    
    for aDT in datDTlist:
        dt_from=aDT-half_period
        dt_to=aDT+half_period

        col_query="AVG("+columns[0]+") "
        if(len(columns)>1):
            for i in range(1,len(columns)):
                col_query+=(",AVG("+columns[i]+")" )
                    
                    
        query="SELECT "+col_query+", count(*) FROM "+ table +" WHERE datetime >= '"+ str(dt_from) +"' AND datetime <= '"+ str(dt_to)+"' AND siteid="+ str(siteid)
        
        try:
            curs.execute(query)
        except:
            print "Unable to execute the querry, exiting.\n"+query
            datoutlist.append(unsucces_result_line)
            continue
                
        tab1 = curs.fetchall()
        if len(tab1)==0:
            datoutlist.append(unsucces_result_line)
        else:
            resultLine=tab1[0]
            values=resultLine[:-1]
            count=resultLine[-1]
            if not(strict_select):
                datoutlist.append(values)
            else:    
                if (count >1):
                    datoutlist.append(values)
                else:
                    datoutlist.append(unsucces_result_line)
    conn.commit()
    conn.close()
    return(datoutlist)


#reads data from database for list of datetime values averaged for given period, this version is for sat data, and siteid is needed
#returns one list - requested parameter list aligned to input DT list
def db_getdata_sat_forDT_ave_std(dsn, table, column_avg, column_std, datDTlist, avg_period, std_period, siteid, strict_select=False):
    datGlist=[]#global list - avg
    datGlist_std=[]#global list - stdev
    half_period_avg=avg_period/2
    half_period_std=std_period/2
    
    if(db_utils.test_dsn(dsn)==False):
        print "Unable to connect to the database, exiting."
        return(datGlist)
    
    conn = psycopg2.connect(dsn)
    curs = conn.cursor()
    
    for aDT in datDTlist:
        dt_from=aDT-half_period_avg
        dt_to=aDT+half_period_avg
        
        if half_period_avg == half_period_std:
            #select for AVG and STD    
            query="SELECT AVG("+column_avg+"), count(*), STDDEV("+column_std+")  FROM "+ table +" WHERE datetime >= '"+ str(dt_from) +"' AND datetime <= '"+ str(dt_to)+"' AND siteid="+ str(siteid)
            try:
                curs.execute(query)
            except:
                print sys.exc_info()
                print "Unable to execute the query, exiting.\n"+query
                datGlist.append(None)
                continue
                
            tab1 = curs.fetchall()
            if len(tab1)==0:
                datGlist.append(None)
                datGlist_std.append(None)
            else:
                value, count, stdev = tab1[0]
                if not(strict_select):
                    datGlist.append(value)
                    datGlist_std.append(stdev)
                else:    
                    if (count >2):
                        datGlist.append(value)
                        datGlist_std.append(stdev)
                    else:
                        datGlist.append(None)
                        datGlist_std.append(None)
                        
        else: # integration time is different for avg and stdev
            #select for AVG    
            query="SELECT AVG("+column_avg+"), count(*) FROM "+ table +" WHERE datetime >= '"+ str(dt_from) +"' AND datetime <= '"+ str(dt_to)+"' AND siteid="+ str(siteid)
            #print query
            try:
                curs.execute(query)
            except:
                print sys.exc_info()
                print "Unable to execute the query, exiting.\n"+query
                datGlist.append(None)
                continue
                    
            tab1 = curs.fetchall()
            if len(tab1)==0:
                datGlist.append(None)
                #print "empty"
            else:
                value, count=tab1[0]
                if not(strict_select):
                    datGlist.append(value)
                else:    
                    if (count >2):
                        datGlist.append(value)
                    else:
                        #print "not enough"
                        datGlist.append(None)
            
            #select for STD    
            dt_from_std=aDT-half_period_std
            dt_to_std=aDT+half_period_std
            query="SELECT STDDEV("+column_std+"), count(*) FROM "+ table +" WHERE datetime >= '"+ str(dt_from_std) +"' AND datetime <= '"+ str(dt_to_std)+"' AND siteid="+ str(siteid)
            #print query
            try:
                curs.execute(query)
            except:
                print sys.exc_info()
                print "Unable to execute the query, exiting.\n"+query
                datGlist_std.append(None)
                continue
                    
            tab1 = curs.fetchall()
            if len(tab1)==0:
                datGlist_std.append(None)
            else:
                value, count=tab1[0]
                if not(strict_select):
                    datGlist_std.append(value)
                else:    
                    if (count >2):
                        datGlist_std.append(value)
                    else:
                        datGlist_std.append(None)

    conn.commit()
    conn.close()
    return(datGlist, datGlist_std)



#reads data from database for list of date values sumarized for given day, this version is for sat data, and siteid is needed
#returns one list - requested parameter list aligned to input D list
def db_getdata_sat_day_forD(dsn, table, column, datDlist, siteid, strict_select=False, sat_tstep=15):
    inst_weight=sat_tstep/60.
    datGlist=[]#global list
    
    if(db_utils.test_dsn(dsn)==False):
        print "Unable to connect to the database, exiting."
        return(datGlist)
    
    conn = psycopg2.connect(dsn)
    curs = conn.cursor()
    
    for aD in datDlist:
        query="SELECT SUM("+column+"), count(*) FROM "+ table +" WHERE date = '"+ str(aD) +"' AND siteid="+ str(siteid)
        try:
            curs.execute(query)
        except:
            print "Unable to execute the querry, skipping.\n"+query
            datGlist.append(None)
            continue
                
        tab1 = curs.fetchall()
        if len(tab1)==0:
            datGlist.append(None)
        else:
            value, count=tab1[0]
            if count == 0:
                datGlist.append(None)
            elif not(strict_select):
                datGlist.append(int(value*inst_weight))
            else:    
                if (count >10):
                    datGlist.append(int(value*inst_weight))
                else:
                    datGlist.append(None)

    conn.commit()
    conn.close()

    return(datGlist)



def db_get_site_latlonelev(siteID, dsn, sitetable, verbose=True):
    # get site longitude latitude from db
    try:
        conn = psycopg2.connect(dsn)
    except:
        print sys.exc_info()
        print "Unable to connect to the database, exiting."
        return(None, None, None)

    curs = conn.cursor()
    querry = "SELECT longitude, latitude, elev  FROM " + sitetable + " WHERE id =" + str(siteID)

    try:
        curs.execute(querry)
    except:
        print sys.exc_info()
        if verbose: print "Unable to execute the query for point coordinates, exiting."
        if verbose: print sys.exc_info()
        return(None, None, None)

    tab1 = curs.fetchall()
    if len(tab1) == 0:
        if verbose: print "Empty query result for point coordinates, exiting."
        return(None, None, None)
    else:
        longit = tab1[0][0]
        latit = tab1[0][1]
        elev = tab1[0][2]
    conn.close()
    return(longit, latit, elev)



def db_get_site_name_description(siteID, dsn, sitetable):
    # get site longitude latitude from db
    try:
        conn = psycopg2.connect(dsn)
    except:
        print sys.exc_info()
        print "Unable to connect to the database, exiting."
        return(None, None, None)

    curs = conn.cursor()
    querry = "SELECT name, description  FROM " + sitetable + " WHERE id =" + str(siteID)

    try:
        curs.execute(querry)
    except:
        print sys.exc_info()
        print "Unable to execute the query for point name, exiting."
        return(None, None)

    tab1 = curs.fetchall()
    if len(tab1) == 0:
        print "Empty query result for point name, exiting."
        return(None, None)
    else:
        name = tab1[0][0]
        description = tab1[0][1]
    conn.close()
    return(name, description)








#---------------------------------------------------------
# functions for new site_coordinates database



def db_get_site_lonlatalt(siteID, dsn, sitetable, verbose=True):
    # get site longitude latitude from db
    try:
        conn = psycopg2.connect(dsn)
    except:
        print sys.exc_info()
        print "Unable to connect to the database, exiting."
        return(None, None, None)

    curs = conn.cursor()
    querry = "SELECT longitude, latitude, altitude  FROM " + sitetable + " WHERE id =" + str(siteID)

    try:
        curs.execute(querry)
    except:
        print sys.exc_info()
        if verbose: print "Unable to execute the query for point coordinates, exiting."
        if verbose: print sys.exc_info()
        return(None, None, None)

    tab1 = curs.fetchall()
    if len(tab1) == 0:
        if verbose: print "Empty query result for point coordinates, exiting."
        return(None, None, None)
    else:
        longit = tab1[0][0]
        latit = tab1[0][1]
        elev = tab1[0][2]
    conn.close()
    return(longit, latit, elev)


def db_get_site_site_info_dict(siteID, dsn, sitetable):
    info_dict={}
    # get site longitude latitude from db
    try:
        conn = psycopg2.connect(dsn)
    except:
        print sys.exc_info()
        print "Unable to connect to the database, exiting."
        return None

    curs = conn.cursor()
    query = "SELECT id, latitude, longitude, altitude, short_name, name, country, customer, comment, country_code, order_code, company_name FROM " \
    + sitetable + " WHERE id =" + str(siteID)

    try:
        curs.execute(query)
    except:
        print sys.exc_info()
        print "Unable to execute the query for site info, exiting."
        conn.close()
        return None

    tab1 = curs.fetchall()
    if len(tab1) == 0:
        print "Empty query result for point name, exiting."
        conn.close()
        return None
    else:
        
        info_dict['id'] = tab1[0][0]
        info_dict['latitude'] = tab1[0][1]
        info_dict['longitude']  = tab1[0][2]
        info_dict['altitude'] = tab1[0][3]
        info_dict['short_name'] = tab1[0][4]
        info_dict['name'] = tab1[0][5]
        info_dict['country'] = tab1[0][6]
        info_dict['customer'] = tab1[0][7]
        info_dict['comment'] = tab1[0][8]
        info_dict['country_code'] = tab1[0][9]
        info_dict['order_code'] = tab1[0][10]
        info_dict['company_name'] = tab1[0][11]
                
    conn.close()
    return info_dict






if __name__ == "__main__":
    """
    #some tests
    def db_test():
    #    la,fi = 8.627056,45.8120278 #Ispra meteotower
        day_begin='20060101'
        day_end  ='20060101'
        slot_begin=26
        slot_end=73
        avg_period=60 #min
        
        dsn="dbname=meteo_sites host=localhost user=tomas"
        tab="r_isp_01min"
        col="gh"
        
        from general_utils import daytimeconv
        date_begin=daytimeconv.yyyymmdd2date(day_begin)
        date_end=daytimeconv.yyyymmdd2date(day_end)
        time_start=daytimeconv.slot2time(slot_begin)
        time_end=daytimeconv.slot2time(slot_end)
        
        timedelt=timedelta(minutes=avg_period)
        
        #test connection
        #print test_dsn(dsn) 
        #querry every existing value
        #print db_getdata(dsn, tab, col, date_begin, date_end, time_start, time_end)
        #querry averages 
    #    print db_getdata_avg(dsn, tab, col, date_begin, date_end, time_start, time_end, timedelt)
        
        #prepare datetime list
        myDTlist=[]
        curr_date=datetime.combine(date_begin,time_start)
        end_date=datetime.combine(date_end,time_end)
        while(curr_date<=end_date):
            myDTlist.append(curr_date)
            curr_date+=timedelt
        #querry datetime
        #print db_getdata_forDT(dsn, tab, col, myDTlist)
        #print db_getdata_forDT_ave(dsn, tab, col, myDTlist, timedelt)
        
        sitename='bai'
        ID=db_site_ID_by_name_s(dsn, sitename)
        print db_site_coord(dsn, ID)
        print db_site_Z(dsn, ID)
        #print db_site_name_s_by_ID(dsn, 9000)
        print db_dtables_for_siteID(dsn, ID)
        print db_dtables_for_siteID_param(dsn, ID, 'radiation')
    
    db_test()
    """