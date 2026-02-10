import pandas as pd
import numpy as np

crime_stats = pd.read_csv("csv_and_txt_files/crime.csv")

mean_violent_crime = np.mean(crime_stats["ViolentCrimesPerPop"])
median_violent_crime = np.median(crime_stats["ViolentCrimesPerPop"])
sd_violent_crime = np.std(crime_stats["ViolentCrimesPerPop"])
min_value_violent_crime = np.min(crime_stats["ViolentCrimesPerPop"])
max_value_violent_crime = np.max(crime_stats["ViolentCrimesPerPop"])

print("Mean of violent crimes per population: ", mean_violent_crime, f" ({mean_violent_crime:.2f})")
print("Median of violent crimes per population: ", median_violent_crime)
print("Standard deviation of violent crimes per population: ", sd_violent_crime, f" ({sd_violent_crime:.2f})")
print("Maximum number of violent crimes per population: ", max_value_violent_crime)
print("Minimum number of violent crimes per population: ", min_value_violent_crime)

#Concerning the mean and the median, given that the standard deviation is 0.28, the median and the mean
#are actually pretty close to each other, so I would say it's rather symmetrical

#Extreme values like 0.02 and 1.0 in a dataset like this affects the mean more than the median. This
#is because the mean takes all the data points into account whilst the median is only the middle
#point of the data and hence is not affected by extreme values
