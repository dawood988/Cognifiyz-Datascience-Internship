# Cognifyz Data Science Internship

## Table of Contents

1. [Overview](#overview)
2. [Level 1](#level-1)
   - [Task 1: Data Exploration and Preprocessing](#task-1-data-exploration-and-preprocessing)
   - [Task 2: Descriptive Analysis](#task-2-descriptive-analysis)
   - [Task 3: Geospatial Analysis](#task-3-geospatial-analysis)
3. [Level 2](#level-2)
   - [Task 1: Table Booking and Online Delivery](#task-1-table-booking-and-online-delivery)
   - [Task 2: Price Range Analysis](#task-2-price-range-analysis)
   - [Task 3: Feature Engineering](#task-3-feature-engineering)
4. [Installation & Dependencies](#installation--dependencies)
5. [Running the Project](#running-the-project)
6. [Results & Visualizations](#results--visualizations)
7. [Conclusion](#conclusion)
8. [Future Enhancements](#future-enhancements)
9. [Author](#author)
10. [License](#license)

## Overview

This project is part of the Cognifyz Data Science Internship program and focuses on analyzing restaurant data. The project involves data cleaning, preprocessing, exploratory data analysis, geospatial analysis, and feature engineering to gain insights into restaurant ratings, table booking, online delivery, and price range distribution.

## Level 1

### Task 1: Data Exploration and Preprocessing
- Load and inspect the dataset.
- Detect and correct encoding issues.
- Handle missing values and duplicates.
- Perform data type conversion and feature transformation.
- Analyze target variable distribution.

### Task 2: Descriptive Analysis
- Compute basic statistical measures (mean, median, standard deviation, etc.).
- Explore categorical variables such as "Country Code," "City," and "Cuisines."
- Identify top cuisines and cities with the highest number of restaurants.

### Task 3: Geospatial Analysis
- Visualize restaurant locations using latitude and longitude data.
- Analyze the distribution of restaurants across different regions.
- Determine correlations between location and rating.

## Level 2

### Task 1: Table Booking and Online Delivery
- Determine the percentage of restaurants that offer table booking and online delivery.
- Compare average ratings of restaurants with and without table booking.
- Analyze online delivery availability across different price ranges.

### Task 2: Price Range Analysis
- Identify the most common price range among restaurants.
- Calculate and visualize average ratings for different price ranges.

### Task 3: Feature Engineering
- Extract additional features such as restaurant name length and address length.
- Create new categorical features for table booking and online delivery.

## Installation & Dependencies

Ensure the following libraries are installed:
```sh
pip install numpy pandas matplotlib seaborn geopandas shapely unidecode
```

## Running the Project

1. Clone this repository:
   ```sh
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```sh
   cd cognifyz-data-science-internship
   ```
3. Run the script:
   ```sh
   python data_analysis.py
   ```

## Results & Visualizations

- **Class Distribution Plot**: Shows imbalances in restaurant ratings.
- **Geospatial Map**: Displays restaurant locations on a world map.
- **Bar Graphs & Pie Charts**: Represent data distributions for table booking, online delivery, and price ranges.
- **Correlation Heatmap**: Highlights relationships between features.

## Conclusion

This project provides insights into restaurant analytics, highlighting trends in ratings, pricing, and availability of services like table booking and online delivery. The analysis also demonstrates the impact of location on restaurant success.

## Future Enhancements

- Apply machine learning models to predict restaurant success.
- Integrate web scraping to collect real-time restaurant data.
- Deploy the project as a web application for interactive analysis.

## Author

- **Dawood M D**

## License

This project is open-source and available under the [MIT License](LICENSE).

