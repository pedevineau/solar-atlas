#! /usr/bin/env python
''''''
import sys
import os
import subprocess
import datetime
import re
import psycopg2
from general_utils import daytimeconv
from general_utils.basic_logger import make_logger
logger = make_logger(__name__)

def getConnection(dbname,host,user,password=None):
	if (password is None) or (password == ''):
		dsn="dbname="+dbname+" host="+host+" user="+user
	else:
		dsn="dbname="+dbname+" host="+host+" user="+user+" password="+password+" connect_timeout=60"
	conn = psycopg2.connect(dsn)
	return conn

def getConnectionDSNString(dsn):
	conn = psycopg2.connect(dsn)
	return conn

def DSNListSelectValid(dsnList, returnFirstOnly=False):
	#reduce list of DSN(strings) to list with only valid/working databases
	dsnValidList=[]
	for dsn in dsnList:
		if test_dsn(dsn,verbose=False):
			dsnValidList.append(dsn)
			if returnFirstOnly:
				break
	return dsnValidList



def DSNDictToDSNString(DSNDict):
	#converts database DSN definition from dictionary to string
	
	DSNString_tokens = []

	#make dict with lowercase keys
	DSNDict_l = {}
	for k in DSNDict.keys():
		DSNDict_l[k.lower()] = DSNDict[k]
	
	DSNDict_lkeys = DSNDict_l.keys()
		
	#database name
	valid_keys = ["dbname", "database", "db"]
	for k in valid_keys:
		if k in DSNDict_lkeys:
			DSNString_tokens.append("dbname="+DSNDict_l[k])
	
	#host
	valid_keys = ["dbhost", "host"]
	for k in valid_keys:
		if k in DSNDict_lkeys:
			DSNString_tokens.append("host="+DSNDict_l[k])

	#user
	valid_keys = ["dbuser", "user"]
	for k in valid_keys:
		if k in DSNDict_lkeys:
			DSNString_tokens.append("user="+DSNDict_l[k])

	#user
	valid_keys = ["dbpassword", "password"]
	for k in valid_keys:
		if k in DSNDict_lkeys:
			DSNString_tokens.append("password="+DSNDict_l[k])

	DSNString = ' '.join(DSNString_tokens)
	return DSNString


#test the dsn validity
#e.g. dsn="dbname=meteo_sites host=hugin user=tomas"
def test_dsn(dsn, verbose=True):
	try:
		conn = psycopg2.connect(dsn)
	except:
		if verbose: print sys.exc_info()
		return(False)
	conn.close()
	return(True)	


#test existence of datatable
#e.g. dsn="dbname=meteo_sites host=hugin user=tomas", tablename="r_pa_10min"
def db_dtable_exist(dsn, tablename):
	if (tablename==None) or (tablename==''):
		print "Missing table name ..."
		return (False)
	
	if not(test_dsn(dsn)):
		print "Unable to connect to the database, exiting"
		return(False)
	conn = psycopg2.connect(dsn)
	query = "SELECT relname FROM pg_class WHERE relname = '"+tablename+"'"
	curs = conn.cursor()
	try:
		curs.execute(query)
	except:
		print "Unable to query table existence, exiting."
		print query
		conn.close()
		return(False)
	tab=curs.fetchall()
	conn.close()
	if len(tab)>0:
		return (True)
	return (False)

#test presence of datatable description
#e.g. dsn="dbname=meteo_sites host=hugin user=tomas", tablename="r_pa_10min", descr_table="data_tables"
def db_dtable_descr_exist(dsn, tablename, descr_table="data_tables"):
	if (tablename==None) or (tablename==''):
		print "Missing table name ..."
		return (False)
		
	if not(test_dsn(dsn)):
		print "Unable to connect to the database, exiting"
		return(False)
	conn = psycopg2.connect(dsn)
	query = "SELECT table_name FROM "+descr_table+" WHERE table_name = '"+tablename+"'"
	curs = conn.cursor()
	try:
		curs.execute(query)
	except:
		print "Unable to query table description existence, exiting."
		conn.close()
		return(False)
	tab=curs.fetchall()
	conn.close()
	if len(tab)>0:
		return (True)
	return (False)

#searches for columns in a given table
def db_table_get_fields(dsn,atable):
	cnames=[]
	if not(test_dsn(dsn, verbose=True)):
		print "cannot connect db"
		return (cnames)
	#select table fields
	conn = psycopg2.connect(dsn)
	curs = conn.cursor()
	query="SELECT column_name FROM information_schema.columns where table_name = '"+atable+"'"
	try:
		curs.execute(query)
	except:
		print "Unable to query table existence, exiting."
		print query
		conn.close()
		return([])
	tab=curs.fetchall()
	conn.close()
	if len(tab)<1:
		return (cnames)
	for row in tab:
		cnames.append(row[0])
	return (cnames)

#checks whether a table has a column with given name
def db_table_has_column(dsn_model, tabl_model, acolumn):
	cols = db_table_get_fields(dsn_model, tabl_model)
	if acolumn in cols:
		return True
	else:
		return False

#checks whether a table has a columns with given name
def db_table_has_columns(dsn_model, tabl_model, acolumns):
	cols = db_table_get_fields(dsn_model, tabl_model)
	res=True
	for acolumn in acolumns:
		res&=(acolumn in cols)
	return res


#return indexes of the table 
#e.g. dsn="dbname=meteo_sites host=localhost user=tomas", tablename="r_pa_10min"
def db_table_get_indexes(dsn, tablename):
	
	if (tablename==None) or (tablename==''):
		print "Missing table name ..."
		return (False)
	
	if not(test_dsn(dsn)):
		print "Unable to connect to the database, exiting"
		return(False)
	
	if not(db_dtable_exist(dsn, tablename)):
		print "Unable to find table, exiting"
		return (False)
		
	conn = psycopg2.connect(dsn)
	query = "SELECT indexname, indexdef FROM pg_indexes WHERE tablename = '"+tablename+"'"
	curs = conn.cursor()
	try:
		curs.execute(query)
	except:
		print "Unable to query indexes."
		print query
		conn.close()
		return(False)
	
	tab=curs.fetchall()
	conn.close()
	
	indexes={}
	if len(tab)<1:
		return (indexes)
	for row in tab:
		indexes[row[0]]=row[1]
	return (indexes)

#remove indexes stored in indexes dictionary 
#e.g. dsn="dbname=meteo_sites host=localhost user=tomas", indexes={'res_model_v12_date_idx': 'CREATE INDEX res_model_v12_date_idx ON res_model_v12 USING btree (date)'}
def db_remove_indexes(dsn, indexes):
	
	if not(test_dsn(dsn)):
		print "Unable to connect to the database, exiting"
		return(False)
	
	conn = psycopg2.connect(dsn)
	curs = conn.cursor()
	
	for index in indexes.keys():
		
		query = "DROP INDEX "+index
		try:
			curs.execute(query)
		except:
			print "Unable to drop index"
			print query
			conn.close()
			return(False)
	conn.commit()
	conn.close()
	return (True)

#create indexes stored in indexes dictionary 
#e.g. dsn="dbname=meteo_sites host=localhost user=tomas", indexes={'res_model_v12_date_idx': 'CREATE INDEX res_model_v12_date_idx ON res_model_v12 USING btree (date)'}
def db_create_indexes(dsn, indexes):
	if not(test_dsn(dsn)):
		print "Unable to connect to the database, exiting"
		return(False)
	
	conn = psycopg2.connect(dsn)
	curs = conn.cursor()
	
	for index in indexes.keys():
		index_query=indexes[index]
		try:
			curs.execute(index_query)
		except:
			print "Unable to create index"
			print index_query
			conn.close()
			return False

		conn.commit()
	conn.close()
	return True


#remove all indexes of a table 
#e.g. dsn="dbname=meteo_sites host=localhost user=tomas", tablename="r_pa_10min"
def db_table_remove_all_indexes(dsn, tablename):
	
	indexes=db_table_get_indexes(dsn, tablename)
	if len(indexes)>0:
		result=db_remove_indexes(dsn, indexes)
		return result
	return (True)

	
def db_table_remove_index(dsn, tablename, index):
	indexes=db_table_get_indexes(dsn, tablename)
	if index not in indexes.keys():
		print "index %s not found for table %s" % (index,tablename)
		return False
	else:
		result=db_remove_indexes(dsn, {index:''})
	return result

	
#vacuum table/ if table name not set vacuums whole db
def db_vacuum_table(dsn,atable=''):
	if (atable==None):
		print "Missing data table ..."
		return False
	
	if not(test_dsn(dsn)):
		print "Unable to connect to the database, exiting."
		return False
	
	conn = psycopg2.connect(dsn)
	conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
	curs = conn.cursor()
	query="VACUUM FULL ANALYSE "+ atable
	try:
		curs.execute(query)
	except:
		print "Unable to execute query"
		print query
	conn.close()

#cluster table/ if table name not set vacuums whole db
def db_cluster_table(dsn,atable='', aindex=''):
	if (atable==None) or (atable==''):
		print "Missing data table ..."
		return False
	
	if not(test_dsn(dsn)):
		print "Unable to connect to the database, exiting."
		return False
	
	conn = psycopg2.connect(dsn)
	conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
	curs = conn.cursor()
	if aindex!='':
		query="CLUSTER "+aindex+" ON "+ atable
	else:
		query="CLUSTER "+ atable
		
	try:
		curs.execute(query)
	except:
		print "Unable to execute query"
		print query
	conn.close()
	return True

#read coordinates of the site
def db_query(dsn, query):

	conn = psycopg2.connect(dsn)
	curs = conn.cursor()
	try:
		curs.execute(query)
	except:
		print "Unable to execute the querry, exiting."
		conn.close()
		return None
	
	result = curs.fetchall()
	conn.close()

	return result



def makeDBDump(dbCreditials, DbBackUpDir):
	'''
	Method to make back_up of the database. Input db definition:
	["db_name", "host", "user", "password"]
	Output is db_dum file in back_up directory as: DBdump_db_name_actualDate.sql.gz
	'''
	db_name = dbCreditials[0]; host = dbCreditials[1]; user = dbCreditials[2]; password = dbCreditials[3]
	aDT = datetime.datetime.now()
	TargetDumpFile = "DBdump_%s_%s.sql.gz"%(db_name, str(aDT.date()))
	DumpFilePath = os.path.join(DbBackUpDir, TargetDumpFile)
	logger.debug("%s back up to %s"%(db_name, DumpFilePath))
	os.putenv('PGPASSWORD', password)
	subprocess.call("pg_dump -iOx %s -h %s -U %s |gzip > %s"%(db_name, host, user, DumpFilePath),shell = True)
	os.unsetenv('PGPASSWORD')

def parsDBDumpFileName(DBNamesList, FileName):
	'''
	Method to parse dump_file name and retrieve the creation date.
	Input is also list of db_names used for matching.
	'''
	datePattern = "[1-2][0-9]{3}-[0-9]{2}-[0-9]{2}"
	PatStr = str()
	for dbName in DBNamesList:
		PatStr = PatStr+dbName+"|"
	dbPattern = PatStr[:-1]
	
	pattern = "("+dbPattern+")"+"[_]"+"("+datePattern+")"
	matcher = re.search(pattern, os.path.basename(FileName))
	if matcher == None:
		return None, None
	DBName = matcher.group(1)
	FileDateStr = matcher.group(2)
	FileDate = daytimeconv.yyyymmdd2date(daytimeconv.yyyy_mm_dd2yyyymmdd(FileDateStr))
	
	return DBName, FileDate


