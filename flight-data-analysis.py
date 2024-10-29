from pyspark.sql import SparkSession
from pyspark.sql.functions import col, unix_timestamp, abs, stddev, count, when

# Create a Spark session
spark = SparkSession.builder.appName("Advanced Flight Data Analysis").getOrCreate()

# Load datasets
flights_df = spark.read.csv("flights.csv", header=True, inferSchema=True)
airports_df = spark.read.csv("airports.csv", header=True, inferSchema=True)
carriers_df = spark.read.csv("carriers.csv", header=True, inferSchema=True)

# Define output paths
output_dir = "output/"
task1_output = output_dir + "task1_largest_discrepancy.csv"
task2_output = output_dir + "task2_consistent_airlines.csv"
task3_output = output_dir + "task3_canceled_routes.csv"
task4_output = output_dir + "task4_carrier_performance_time_of_day.csv"

# ------------------------
# Task 1: Flights with the Largest Discrepancy Between Scheduled and Actual Travel Time
# ------------------------
def task1_largest_discrepancy(flights_df, carriers_df):
    flights_df.createOrReplaceTempView("flights")
    carriers_df.createOrReplaceTempView("carriers")

    query = """
    SELECT f.FlightNum, c.CarrierName, f.Origin, f.Destination, 
           UNIX_TIMESTAMP(f.ScheduledArrival) - UNIX_TIMESTAMP(f.ScheduledDeparture) AS ScheduledTime,
           UNIX_TIMESTAMP(f.ActualArrival) - UNIX_TIMESTAMP(f.ActualDeparture) AS ActualTime,
           ABS((UNIX_TIMESTAMP(f.ScheduledArrival) - UNIX_TIMESTAMP(f.ScheduledDeparture)) - 
               (UNIX_TIMESTAMP(f.ActualArrival) - UNIX_TIMESTAMP(f.ActualDeparture))) AS Discrepancy
    FROM flights f
    JOIN carriers c ON f.CarrierCode = c.CarrierCode
    ORDER BY Discrepancy DESC
    """
    largest_discrepancy = spark.sql(query)
    largest_discrepancy.write.csv(task1_output, header=True)
    print(f"Task 1 output written to {task1_output}")

# ------------------------
# Task 2: Most Consistently On-Time Airlines Using Standard Deviation
# ------------------------
def task2_consistent_airlines(flights_df, carriers_df):
    flights_df.createOrReplaceTempView("flights")
    carriers_df.createOrReplaceTempView("carriers")

    query = """
    SELECT c.CarrierName, 
           COUNT(*) AS TotalFlights,
           STDDEV(UNIX_TIMESTAMP(f.ActualDeparture) - UNIX_TIMESTAMP(f.ScheduledDeparture)) AS DepartureDelayStd
    FROM flights f
    JOIN carriers c ON f.CarrierCode = c.CarrierCode
    GROUP BY c.CarrierName
    HAVING TotalFlights > 100
    ORDER BY DepartureDelayStd ASC
    """
    consistent_airlines = spark.sql(query)
    consistent_airlines.write.csv(task2_output, header=True)
    print(f"Task 2 output written to {task2_output}")

# ------------------------
# Task 3: Origin-Destination Pairs with the Highest Percentage of Canceled Flights
# ------------------------
def task3_canceled_routes(flights_df, airports_df):
    flights_df.createOrReplaceTempView("flights")
    airports_df.createOrReplaceTempView("airports")

    query = """
    SELECT a1.AirportName AS OriginAirport, a1.City AS OriginCity, 
           a2.AirportName AS DestinationAirport, a2.City AS DestinationCity,
           COUNT(CASE WHEN f.ActualDeparture IS NULL THEN 1 END) / COUNT(*) * 100 AS CancellationRate
    FROM flights f
    JOIN airports a1 ON f.Origin = a1.AirportCode
    JOIN airports a2 ON f.Destination = a2.AirportCode
    GROUP BY a1.AirportName, a1.City, a2.AirportName, a2.City
    ORDER BY CancellationRate DESC
    """
    canceled_routes = spark.sql(query)
    canceled_routes.write.csv(task3_output, header=True)
    print(f"Task 3 output written to {task3_output}")

# ------------------------
# Task 4: Carrier Performance Based on Time of Day
# ------------------------
def task4_carrier_performance_time_of_day(flights_df, carriers_df):
    flights_df.createOrReplaceTempView("flights")
    carriers_df.createOrReplaceTempView("carriers")

    query = """
    SELECT c.CarrierName,
           CASE 
               WHEN HOUR(f.ScheduledDeparture) BETWEEN 6 AND 12 THEN 'Morning'
               WHEN HOUR(f.ScheduledDeparture) BETWEEN 12 AND 18 THEN 'Afternoon'
               WHEN HOUR(f.ScheduledDeparture) BETWEEN 18 AND 24 THEN 'Evening'
               ELSE 'Night'
           END AS TimeOfDay,
           AVG(UNIX_TIMESTAMP(f.ActualDeparture) - UNIX_TIMESTAMP(f.ScheduledDeparture)) AS AvgDepartureDelay
    FROM flights f
    JOIN carriers c ON f.CarrierCode = c.CarrierCode
    GROUP BY c.CarrierName, TimeOfDay
    ORDER BY TimeOfDay, AvgDepartureDelay ASC
    """
    carrier_performance_time_of_day = spark.sql(query)
    carrier_performance_time_of_day.write.csv(task4_output, header=True)
    print(f"Task 4 output written to {task4_output}")

# ------------------------
# Call the functions for each task
# ------------------------
task1_largest_discrepancy(flights_df, carriers_df)
task2_consistent_airlines(flights_df, carriers_df)
task3_canceled_routes(flights_df, airports_df)
task4_carrier_performance_time_of_day(flights_df, carriers_df)

# Stop the Spark session
spark.stop()
