from psycopg2 import connect
import matplotlib.pyplot as plt

try:
    conn = connect(dbname='crime', user='postgres', host='localhost', password='badguy')
    cur = conn.cursor()
except:
    print "I am unable to connect to the database"
    quit()

cur.execute("SELECT DISTINCT category FROM train;")
categories = cur.fetchall()


cur.execute("SELECT DISTINCT EXTRACT(YEAR FROM dates) FROM train")
years = cur.fetchall()



for cat in categories:

    query = "WITH count_by_week AS (" \
            "SELECT COUNT(id)::INT AS crime_count, EXTRACT(WEEK FROM dates)::INT AS week, " \
            "EXTRACT(YEAR FROM dates) AS year " \
            "FROM full_data " \
            "WHERE category = '{0}' " \
            "GROUP BY year, week ORDER BY week ASC" \
            ") SELECT AVG(crime_count) AS crime_avg, week " \
            "FROM count_by_week " \
            "GROUP BY week " \
            "ORDER BY week;".format(cat[0])

    cur.execute(query)
    results = cur.fetchall()

    if results:
        crime_avg, week = [list(r) for r in zip(*results)]

        plt.plot(week, crime_avg)
        plt.ylabel('Average Frequency')
        plt.xlabel('Week of the Year')
        plt.title(cat[0])
        plt.rcParams['figure.figsize'] = [12, 9]
        plt.savefig('Avg_Over_Years_'+cat[0].replace('/', '_')+'.png')
        plt.clf()
        print('Finished plot: '+cat[0])


cur.close()
conn.close()

