'''
Created on Feb 10, 2012

@author: tomas
'''

import MySQLdb
import sys


def getConnection(dbname,host,user,password=None):
    if (password is None) or (password == ''):
        conn=MySQLdb.connect(host=host,user=user,db=dbname)
    else:
        conn=MySQLdb.connect(host=host,user=user,passwd=password,db=dbname)
    return conn

def testConnection(dbname,host,user,password=None, verbose=True):
    try:
        conn=getConnection(dbname,host,user,password)
    except:
        if verbose: print sys.exc_info()
        return(False)
    conn.close()
    return(True)    


def db_table_exist(conn, dbname='', tablename=''):
    if (tablename==None) or (tablename==''):
        print "Missing table name ..."
        return (False)
    
    query = "SELECT table_name FROM information_schema.tables WHERE table_schema = '%s' AND table_name = '%s'" % (dbname, tablename)
    curs = conn.cursor()
    try:
        curs.execute(query)
    except:
        print "Unable to query table existence, exiting."
        print query
        print sys.exc_info()
        return(False)
    tab=curs.fetchall()
    if len(tab)>0:
        return (True)
    return (False)

def db_get_tables(conn, dbname=''):
    tables=[]

    query = "SELECT table_name FROM information_schema.tables WHERE table_schema = '%s' " % (dbname)
    curs = conn.cursor()
    try:
        curs.execute(query)
    except:
        print "Unable to query tables , exiting."
        print query
        print sys.exc_info()
        return tables

    tab=curs.fetchall()
    for t in tab:
        tables.append(t[0])

    return tables


def db_table_get_fields(conn,atable, only_names=True):
    cnames=[]
    columns=[]
    query="SHOW COLUMNS FROM "+atable+""
    curs = conn.cursor()
    try:
        curs.execute(query)
    except:
        print "Unable to query table existence, exiting."
        print query
        return([])
    tab=curs.fetchall()
    if len(tab)<1:
        return (cnames)
    for row in tab:
        columns.append(row)
        cnames.append(row[0])
    if only_names:
        return (cnames)
    else:
        return (columns)





def test():

    db="monitoring_db"
    host="venus"
#    p=13306
    user="artur"
    pwd="3grawUjar"
    tablename='weather_data'
    tablename='production_data'
#    tablename='inverter_data'
    
    if not (testConnection(dbname=db,host=host,user=user,password=pwd)):
        exit()

    conn=getConnection(dbname=db,host=host,user=user,password=pwd)
    
    if not (db_table_exist(conn, dbname=db, tablename=tablename)):
        exit()
    
    print db_get_tables(conn, dbname=db)
    
    for f in db_table_get_fields(conn,tablename, only_names=False):
        print f

    query="SELECT sourceId, count(*) FROM %s GROUP BY sourceId " % (tablename)
    cur = conn.cursor()
    cur.execute(query)
    data = cur.fetchall()
    for row in data :
        print row
        pass
    
    conn.close()



if __name__ == "__main__":
    test()
