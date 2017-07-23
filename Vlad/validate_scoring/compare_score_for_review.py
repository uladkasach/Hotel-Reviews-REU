import MySQLdb
import MySQLdb.cursors
import json

## get auth data
with open('auth/mysql_connection_data.json') as data_file:    
    auth = json.load(data_file)

def compare():
    ## open mysql connection
    conn = MySQLdb.connect(user=auth["user"], passwd=auth["password"], db=auth["database"], cursorclass = MySQLdb.cursors.SSCursor)
    cur = conn.cursor()

    ## query for 
    cur.execute("SELECT ExtID, Text FROM reviews")
    while row is not None:
        i += 1;
        # row[0]
        row = cur.fetchone()
    cur.close()

    conn.close()

