import pandas as pd
from psycopg2 import connect
from datetime import datetime, date

try:
    conn = connect(dbname='crime', user='postgres', host='localhost', password='badguy')
    cur = conn.cursor()
except:
    print "I am unable to connect to the database"
    quit()

# Header of file: Dates,Category,Descript,DayOfWeek,PdDistrict,Resolution,Address,X,Y
# Example contents: 015-05-13 23:53:00,WARRANTS,WARRANT ARREST,Wednesday,NORTHERN,"ARREST, BOOKED",
# OAK ST / LAGUNA ST,-122.425891675136,37.7745985956747
cur.execute('DROP TABLE IF EXISTS full_data;')
cur.execute('CREATE TABLE full_data ('
            'id serial PRIMARY KEY, '
            'incidntNum BIGINT, '
            'category VARCHAR, '
            'description TEXT, '
            'day_of_week VARCHAR, '
            'dates date, '
            'times time, '
            'pd_district VARCHAR, '
            'resolution TEXT, '
            'address TEXT, '
            'x DECIMAL, '
            'y DECIMAL, '
            'location TEXT, '
            'pdId BIGINT);')
conn.commit()

print("Reading in data...")
train = pd.read_csv('../CrimeData/SFPD_Incident.csv')
print("done")

print("Inserting data into database...")
count = 0
for index, row in train.iterrows():
    values = [row['IncidntNum'], row['Category'], row['Descript'].replace("'", "''"), row['DayOfWeek'],
              datetime.strptime(row['Date'], '%m/%d/%Y').date(), row['Time'], row['PdDistrict'], row['Resolution'],
              row['Address'], row['X'], row['Y'], row['Location'], row['PdId']]

    count = count+1
    if count % 10000 == 0:
        print('Inserted {0} records so far'.format(count))

    cur.execute('INSERT INTO full_data (incidntNum, category, description, day_of_week, dates, times, pd_district, '
                'resolution, address, x, y, location, pdId) '
               'VALUES (\'{0[0]}\', \'{0[1]}\', \'{0[2]}\', \'{0[3]}\', \'{0[4]}\', \'{0[5]}\', \'{0[6]}\', '
                '\'{0[7]}\', \'{0[8]}\', \'{0[9]}\', \'{0[10]}\', \'{0[11]}\', \'{0[12]}\');'
                .format(values))


conn.commit()

cur.close()
conn.close()
print("All done!")
