import csv
import numpy as np
import time
import re
import ipaddress

path = "D:\\University\\Master\\M102CEM - Project\\Code\\Data\\Original_CSV\\16-09-23.csv"

#time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.localtime(epoch))


def toTime(epoch):
    return time.strftime("%H%M", time.localtime(int(epoch)))

def mac_to_int(mac):
    res = re.match('^((?:(?:[0-9a-f]{2}):){5}[0-9a-f]{2})$', mac.lower())
    if res is None:
        raise ValueError('invalid mac address')
    return int(res.group(0).replace(':', ''), 16)

def ip_to_int(ip):
    return int(ipaddress.IPv4Address(ip))

path_to_write = "D:\\University\\Master\\M102CEM - Project\\Code\\Converted.csv"

newCSV = []

with open(path) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    count = 0;
    for row in readCSV:
        if count != 0:
            newCSV.append([toTime(row[1]),row[2],mac_to_int(row[3]),mac_to_int(row[4]),ip_to_int(row[5]),ip_to_int(row[6]),row[7],row[8],row[9],row[10]])
        else:
            newCSV.append(row[1:])
        count += 1

with open(path_to_write, mode='w', newline='') as CSV:
    writer = csv.writer(CSV)
    for row in newCSV:
        writer.writerow(row)
