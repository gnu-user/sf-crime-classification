from psycopg2 import connect
import matplotlib.pyplot as plt

# Each type of crime by the hour/period of the day (e.g. prostitution goes up between 1 - 5 am)

try:
    conn = connect(dbname='crime', user='postgres', host='localhost', password='badguy')
    cur = conn.cursor()
except:
    print "I am unable to connect to the database"
    quit()

cur.execute("SELECT DISTINCT category FROM full_data;")
categories = cur.fetchall()


cur.execute("SELECT DISTINCT EXTRACT(YEAR FROM dates) FROM full_data;")
years = cur.fetchall()



for cat in categories:

    query = "WITH count_by_hour AS (" \
            "SELECT COUNT(id)::INT AS crime_count, EXTRACT(HOUR FROM times)::INT AS hours, " \
            "dates " \
            "FROM full_data " \
            "WHERE category = '{0}' " \
            "GROUP BY dates, hours" \
            ") SELECT AVG(crime_count) AS crime_avg, hours " \
            "FROM count_by_hour " \
            "GROUP BY hours " \
            "ORDER BY hours;".format(cat[0])

    cur.execute(query)
    results = cur.fetchall()

    if results:
        crime_avg, week = [list(r) for r in zip(*results)]

        plt.plot(week, crime_avg)
        plt.ylabel('Average Frequency')
        plt.xlabel('Hour of day')
        plt.title(cat[0])
        plt.rcParams['figure.figsize'] = [12, 9]
        plt.savefig('Avg_Over_Hours_'+cat[0].replace('/', '_')+'.png')
        plt.clf()
        print('Finished plot: '+cat[0])


cur.close()
conn.close()

