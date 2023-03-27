import csv

with open('files/results/model_2.csv', newline='') as csvfile:
    print("\nModel 3-2")
    reader = csv.reader(csvfile, delimiter=';')
    sumTime = float(0)
    sumAcc = float(0)
    for row in reader:
        sumTime += float(row[0])
        sumAcc += float(row[1])
    print("Average Time: " + str(sumTime / 10.0))
    print("Average Accuracy: " + str(sumAcc / 10.0))
    
# with open('files/results/model_3-3.csv', newline='') as csvfile:
#     print("\nModel 3-3")
#     reader = csv.reader(csvfile, delimiter=';', quotechar='|')
#     sumTime = float(0)
#     sumAcc = float(0)
#     for row in reader:
#         sumTime += float(row[0])
#         sumAcc += float(row[1])
#     print("Average Time: " + str(sumTime / 10.0))
#     print("Average Accuracy: " + str(sumAcc / 10.0))

# with open('files/results/model_3-4.csv', newline='') as csvfile:
#     print("\nModel 3-4")
#     reader = csv.reader(csvfile, delimiter=';', quotechar='|')
#     sumTime = float(0)
#     sumAcc = float(0)
#     for row in reader:
#         sumTime += float(row[0])
#         sumAcc += float(row[1])
#     print("Average Time: " + str(sumTime / 10.0))
#     print("Average Accuracy: " + str(sumAcc / 10.0))


# Model 3-3 with 50 Epochs
# iteration;datetime;elapsed_time;acc
#1;2023-03-19_19-54-59;6111.247538089752;91.846764087677

# Model 3-4 with 50 Epochs
#iteration;datetime;elapsed_time;acc
#1;2023-03-19_23-06-01;6454.079206943512;89.31238055229187