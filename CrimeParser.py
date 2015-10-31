import pandas as pd
from psycopg2 import connect

try:
    conn = connect(dbname='crime', user='postgres', host='localhost', password='badguy')
    cur = conn.cursor()
except:
    print "I am unable to connect to the database"

# Header of file: Dates,Category,Descript,DayOfWeek,PdDistrict,Resolution,Address,X,Y
# Example contents: 015-05-13 23:53:00,WARRANTS,WARRANT ARREST,Wednesday,NORTHERN,"ARREST, BOOKED",
# OAK ST / LAGUNA ST,-122.425891675136,37.7745985956747
cur.execute('DROP TABLE IF EXISTS train;')
cur.execute('CREATE TABLE train ('
            'id serial PRIMARY KEY, '
            'dates TIMESTAMP, '
            'category VARCHAR, '
            'description TEXT, '
            'day_of_week VARCHAR, '
            'pd_district VARCHAR, '
            'resolution TEXT, '
            'address TEXT, '
            'x DECIMAL, '
            'y DECIMAL);')

print("Reading in data...")
train = pd.read_csv('../CrimeData/train.csv')
print("done")

print("Inserting data into database...")
count = 0
for index, row in train.iterrows():
    values = [row['Dates'], row['Category'], row['Descript'].replace("'", "''"), row['DayOfWeek'], row['PdDistrict'],
              row['Resolution'], row['Address'], row['X'], row['Y']]

    if count % 10000 == 0:
        print('Inserted {0} records so far'.format(count))
    cur.execute('INSERT INTO train (dates, category, description, day_of_week, '
               'pd_district, resolution, address, x, y) '
               'VALUES (\'{0[0]}\', \'{0[1]}\', \'{0[2]}\', \'{0[3]}\', \'{0[4]}\', \'{0[5]}\', \'{0[6]}\', '
                '\'{0[7]}\', \'{0[8]}\');'
                .format(values))
    count = count+1


print("All done!")
