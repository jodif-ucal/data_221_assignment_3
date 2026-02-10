import matplotlib.pyplot as plt
import pandas as pd

crime_stats = pd.read_csv("csv_and_txt_files/crime.csv")

#==================================================================================================
#Histogram
plt.hist(crime_stats["ViolentCrimesPerPop"], bins=30, edgecolor="white")
plt.title("Number of violent crimes committed per population")
plt.xlabel("Violent crimes committed per population")
plt.ylabel("Frequency")
plt.show()
#==================================================================================================

#==================================================================================================
#Boxplot
plt.boxplot(crime_stats["ViolentCrimesPerPop"])
plt.title("Distribution of violent crimes committed per population")
plt.xlabel("The populations")
plt.ylabel("Number of crimes committed per population")
plt.show()
#==================================================================================================

#The histogram shows that the data is rather right-skewed, meaning that most of the data points
#are closer to the left than the middle. Each local maxima decreases as we move towards the right
#of the graph, except at the very end, where the most amount of crimes committed per population
#(between 0.96 and 1.0 per population) appears to happen in most regions. From all of this, we can
#tell that not a lot of the regions have a high amount of violent crime in them, and the regions that do have
#a good amount of violent crime in them decrease in frequency, except when it is over than 0.96,

#The boxplot makes the same suggestion as the histogram that this data is right-skewed, as the median
#amount of violent crimes committed per population is closer to the lower quartile than it is the upper quartile,
#at 0.4. One thing that the boxplot can indicate over the histogram is the occurrence of outliers,
#but there does not seem to be any present here, which makes the regions that do have over 0.96 violent
#crimes per population much stranger, especially when it is over 0.3 units away from the upper quartile.