import sqlite3
import csv
from datetime import datetime
import os
from subprocess import Popen, PIPE
from tqdm import tqdm
from shapely.wkt import loads as shapely_loads
import urllib.request


if __name__ == '__main__':
    print("download data files from Melbourne")
    if not os.path.exists("data/"):
        os.mkdir("data/")

    if not os.path.exists('data/bay_locations.csv'):
        urllib.request.urlretrieve('https://data.melbourne.vic.gov.au/api/views/wuf8-susg/rows.csv?accessType=DOWNLOAD',
                                   'data/bay_locations.csv')
    if not os.path.exists('data/On-street_Parking_Bay_Sensors.csv'):
        urllib.request.urlretrieve('https://data.melbourne.vic.gov.au/api/views/vh2v-4nfs/rows.csv?accessType=DOWNLOAD',
                                   'data/On-street_Parking_Bay_Sensors.csv')
    if not os.path.exists('data/Road_Corridor.csv'):
        urllib.request.urlretrieve('https://data.melbourne.vic.gov.au/api/views/wzzt-avwf/rows.csv?accessType=DOWNLOAD',
                                   'data/Road_Corridor.csv')
    if not os.path.exists('data/On-street_Car_Park_Bay_Restrictions.csv'):
        urllib.request.urlretrieve('https://data.melbourne.vic.gov.au/api/views/ntht-5rk7/rows.csv?accessType=DOWNLOAD',
                                   'data/On-street_Car_Park_Bay_Restrictions.csv')
    if not os.path.exists('data/On-street_Car_Parking_Sensor_Data_-_2017.csv'):
        urllib.request.urlretrieve('https://data.melbourne.vic.gov.au/api/views/u9sa-j86i/rows.csv?accessType=DOWNLOAD',
                                   'data/On-street_Car_Parking_Sensor_Data_-_2017.csv')
    print("Done.")

    db_file = 'dataset.db'
    if os.path.exists(db_file):
        os.remove(db_file)

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    with open("db_schema", "r") as f:
        commands = f.read().split(";")
        for command in commands:
            cursor.execute(command)

    with open("data/bay_locations.csv", "r") as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        next(reader)
        for the_geom, marker_id, meter_id, bay_id, last_edit, rd_seg_id, rd_seg_dsc in reader:
            point = shapely_loads(the_geom).centroid
            lat, lon = point.y, point.x
            cursor.execute("INSERT OR IGNORE INTO locations (marker, lat, lon) VALUES(\"%s\", \"%f\", \"%f\")" % (marker_id, lat, lon))

    tables = [
        ("data/Road_Corridor.csv", "roads", [], []),
        ("data/On-street_Car_Park_Bay_Restrictions.csv", "restrictions", [], [*('"StartTime%d"' % i for i in range(1,7)), *('"EndTime%d"' % i for i in range(1,7))]),
        ("data/On-street_Parking_Bay_Sensors.csv", "sensors", [], []),
        ("data/On-street_Car_Parking_Sensor_Data_-_2017.csv", "events", ['"ArrivalTime"', '"DepartureTime"'], []),
    ]

    for file, table, datetime_cols, time_cols in tables:

        output = Popen(["wc", "-l", file], stdout=PIPE)
        lines = int(Popen(["wc", "-l", file], stdout=PIPE).communicate()[0].split(b" ")[0])
        with tqdm(total=lines) as pbar:
            with open(file, "r") as f:
                reader = csv.reader(f, delimiter=',', quotechar='"')
                header = ['"' + h + '"' for h in next(reader)]
                datetime_idx = [header.index(t) for t in datetime_cols]
                time_idx = [header.index(t) for t in time_cols]
                for data in reader:
                    for idx in datetime_idx:
                        if len(data[idx]) == 0:
                            data[idx] = "0"
                        else:
                            data[idx] = datetime.strptime(data[idx], "%m/%d/%Y %I:%M:%S %p").strftime("%s")
                    for idx in time_idx:
                        if len(data[idx]) == 0:
                            data[idx] = "0"
                        else:
                            data[idx] = "%d" % (datetime.strptime(data[idx], "%H:%M:%S") - datetime(1900,1,1)).total_seconds()
                    data = ['"' + d + '"' for d in data]

                    if len(data) == len(header):
                        cursor.execute("INSERT INTO " + table + " (" + ", ".join(header) + ") VALUES(" + ", ".join(data) + ")")
                    else:
                        print("Data len is not header len!!")

                    pbar.update(1)

        conn.commit()

    indices = [
        ("devices", "DeviceId"),
        ("roads", "GISID"),
        ("restrictions", "BayID"),
        ("restrictions", "DeviceID"),
        ("sensors", "bay_id"),
        ("locations", "marker"),
        ("events", "DeviceId"),
        ("events", "ArrivalTime"),
        ("events", "DepartureTime"),
        ("events", "Area"),
        ("events", "Sign"),
        ("events", "\"Vehicle Present\""),
        ("events", "StreetMarker"),
    ]

    for i, col in enumerate(indices):
        print("create index for", *col, end="...")
        cursor.execute("CREATE INDEX i%d on %s (%s)" % (i, *col))
        print("done")

    conn.commit()

    print("create durations")
    for i in range(1,7):
        sign = "Description%d" % i
        duration = "Duration%d" % i
        cursor.execute("INSERT OR IGNORE INTO durations (sign, duration) "
                       "SELECT Description%d , Duration%d from restrictions "
                       "where Description%d<>''" % (i, i, i))

    rows = cursor.execute("select distinct(events.sign) "
                          "from events where not exists("
                          "select * from durations where events.sign=durations.sign)")

    patterns = {
        '1/2': 30,
        '1/2P': 30,
        '1/4P': 15,
        '1P': 60,
        '2P': 2*60,
        '3P': 3*60,
        '4P': 4*60,
        '10P': 4*60,
        'P10': 10,
        'P/10': 10,
        'P/15': 15,
        '30MINS': 30,
        '15mins': 15,
        '15Mins': 15,
        '1PMTR': 60,
    }
    for sign, in rows:
        if len(sign) == 0:
            continue
        found = False
        for p in sign.split(" "):
            if p in patterns:
                duration = patterns[p]
                cursor.execute("INSERT INTO durations (sign, duration) VALUES (\"%s\", %d)" % (sign, duration))
                found = True
                break
        if not found:
            print("no way to parse sign", sign)
            cursor.execute("INSERT INTO durations (sign, duration) VALUES (\"%s\", %d)" % (sign, 60))

    conn.commit()
    print("done")

    print("create devices based on the location table")
    rows = cursor.execute("select DeviceId, area, lat, lon "
                          "from events join locations on StreetMarker=marker "
                          "group by DeviceId ")
    for device_id, area, lat, lon in tqdm(rows.fetchall()):
        cursor.execute("INSERT INTO devices (DeviceId, Area, lat, lon) VALUES(%d, \"%s\", %f, %f)"
                       % (device_id, area, lat, lon))

    cursor.close()

    conn.commit()
    conn.close()
