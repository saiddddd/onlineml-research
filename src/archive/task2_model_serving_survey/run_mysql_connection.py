from __future__ import print_function

import datetime
import pickle

import mysql
from mysql.connector import (connection, errorcode)


cnx = connection.MySQLConnection(user='pwang', password='pwang', host='localhost', database='online_ml')
cursor = cnx.cursor()

# Do some logic here

TABLES = {}

TABLES['modelsaving'] = (
    "create table `t_hoeffding_classifier_persist` ("
    "  `persist_time` date NOT NULL,"
    "  `model` BLOB NOT NULL"
    " )"
)

for table_name in TABLES:
    table_description = TABLES[table_name]

    try:
        print("creating table {}: ".format(table_name), end='')
        cursor.execute(table_description)
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_TABLE_EXISTS_ERROR:
            print("already exists.")
        else:
            print(err.msg)
    else:
        print("OK")

hello_message = "hello world!"
pickled_hello_message = pickle.dumps(hello_message)
print(type(pickled_hello_message))

insert_request = (
    "insert into t_hoeffding_classifier_persist "
    "(persist_time, model)"
    "VALUES (%s, %s)"
)
insert_value = (datetime.datetime.now(), pickled_hello_message,)

cursor.execute(insert_request, insert_value)

cnx.commit()

# query data
query_sql = (
    "select * from t_hoeffding_classifier_persist"
)

cursor.execute(query_sql)

for i in cursor:
    print(i)


# End of operation logic

cursor.close()
cnx.close()
