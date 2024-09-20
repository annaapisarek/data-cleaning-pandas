# project-shark-attack-data
Shark Attack Analysis for Insurance Risk Assessment

Project Overview
This project aims to analyze shark attack data to provide insights for an insurance company. Using the Global Shark Attack File (GSAF) dataset, we explore the relationships between various factors like human activities, time of year, and gender to assess the risks of shark attacks. The goal is to test specific hypotheses that can inform insurance risk policies.

Dataset Description
The dataset contains records of shark attack incidents, with information including:
Location: The geographical area where the attack took place.
Activity: The activity of the person during the incident (e.g., swimming, surfing).
Provoked vs. Unprovoked: The nature of the incident, whether it was provoked, unprovoked, or involved watercraft.
Gender and Age: Personal details of the victims when available.
Date and Time: Date and time of the incident.
Fatal vs. Non-fatal incidents

Mission and Hypotheses
The main objective of this analysis is to provide an insurance company with insights into shark attack trends, which could impact risk assessment policies.

Hypotheses:
Surfers are more likely to be attacked by sharks.
Shark attacks are more common in summer than in other seasons.
Men are much more likely to be attacked by a shark than women.

Data Cleaning Process
To ensure the dataset is suitable for analysis, several steps were taken to clean and prepare the data:
Missing data handling: 
Imputed or removed rows with significant missing values.
Date parsing:
Converted the date to a proper format and extracted useful time-based features (e.g., month, year).
Categorical variable standardization: Standardized activities, locations, and injury types to maintain consistency.
Text cleaning:
Applied regular expressions to clean fields like "Injury" and "Activity" for better analysis.
Numerical columns: Converted relevant fields like age and year to appropriate numeric types and handled outliers.

Analysis Approach
Exploratory Data Analysis (EDA): 
Initial exploration was done to understand the distribution of data, trends, and correlations.
Clustering and Grouping: Used clustering techniques to group similar activities and assess attack risks based on different categories.

Results

Hypothesis 1: Surfers show a higher proportion of attacks compared to other activities such as swimming or wading, but swimming has the highest rate of fatality.

Hypothesis 2: A peak in shark attacks was observed during summer months, with a noticeable increase in warm weather.

Hypothesis 3: Data suggests that men are significantly more likely to be victims of shark attacks compared to women.

Full detailed findings are provided in the analysis section of the project.
