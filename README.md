# MLP Exoplanet Hunting
#### Requirements
* keras
* numpy
* requests
* scikit
* tensorflow
* tqdm

#### Usage
```
git clone https://github.com/drewszurko/keras-exoplanet-analysis.git
cd keras-exoplanet-analysis/ 
pip install -r requirements.txt
python main.py
```
#### Datasets
The datasets were created and open-sourced by Winter Delta (WΔ) and can be found on [Kaggle](https://www.kaggle.com/keplersmachines/kepler-labelled-time-series-data) and [GitHub](https://github.com/winterdelta/KeplerAI). They have also been pre-split into training and testing files, `exo_train.csv.gz` and `exo_test.csv.gz`.

#### Dataset Descriptions
`exo_train.csv.gz`

* 5087 observations
* 3198 features
* 37 confirmed exoplanet stars
* 5050 non-exoplanet stars
* Column 1 is the target column (0 = non-exoplanet star, 1 = confirmed exoplanet star)
* Columns 2-3198 are feature columns that display the star's flux values over time

`exo_test.csv.gz`

* 570 observations
* 3198 features
* 5 confirmed exoplanet stars
* 565 non-exoplanet stars
* Column 1 is the target column (0 = non-exoplanet star, 1 = confirmed exoplanet star)
* Columns 2-3198 are feature columns that display the star's flux values over time
