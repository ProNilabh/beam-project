# Data folder

Place the dataset file here:

    ENB2012_data.xlsx

This is the UCI Energy Efficiency dataset by Tsanas & Xifara (2012). You can download it from:

    https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx

Or copy it from your existing project folder if you already have it.

Without this file, training will fail. The Docker container expects the file at /app/data/ENB2012_data.xlsx, which is mounted from this folder.
