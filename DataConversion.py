import csv
import numpy as np
import time
import re
import ipaddress


#Convert the epoch time to just hours and minutes
def toTime(epoch):
    try:
        return time.strftime("%H%M", time.localtime(int(epoch)))
    except:
        return epoch

#Convert a MAC address to a 64bit integer
def mac_to_int(mac):
    res = re.match('^((?:(?:[0-9a-f]{2}):){5}[0-9a-f]{2})$', mac.lower())
    if res is None:
        return mac
    return int(res.group(0).replace(':', ''), 16)

#Convert an IP address to a 64bit integer
def ip_to_int(ip):
    try:
        return int(ipaddress.IPv4Address(ip))
    except:
        return ip


if __name__ == '__main__':

    path = "D:\\University\\Master\\M102CEM - Project\\Code\\Data\\Original_CSV\\16-09-23.csv"
    path_to_write = "D:\\University\\Master\\M102CEM - Project\\Code\\Converted.csv"
    newCSV = []

    #Open and read the CSV file
    with open(path) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            newCSV.append([toTime(row[1]),row[2],row[7],row[8],row[9],row[10]]) #Add only the data we need to a list
    
    #Open the new file and write the data to it
    with open(path_to_write, mode='w', newline='') as CSV:
        writer = csv.writer(CSV)
        for row in newCSV:
            writer.writerow(row)