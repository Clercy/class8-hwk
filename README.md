# class8-hwk

My Question
Can you pinpoint the areas in the Boston region that would be ideal
to bring up a family?  I would base my findings on the following 3 criteria.  
Affordability(MEDV), schooling(PTRATIO) and peace of mind(CRIM).

My initial analysis was based upon figure PLT_MEDV_PTRATIO_CRIM_20193719145.png
from Class7hmwrk.  I was looking to find the best scenarios based on
attributes MEDV, PTRATIO and CRIM. Based on that chart my choice for best scenarios
were the positions between respondents 100 to 130 and 310 to 340.  We see areas where
MEDV is below or at the average.  There is a constant low crime rate and lower than average PTRATIO.
Based on the revised version of PLT_MEDV_PTRATIO_CRIM_20193112245.png in class8-hwk the
resorting of MEDV makes the visuals much clearer with positions between 130 and 340 as the
best areas.  

Using tools from this week's homework like the correlation heatmap Heatmap_2019314213316.png,
it shows a negative correlation between attributes PTRATIO, CRIM vs MEDV while
for PTRATIO vs CRIM have a positive correlation.

MEDV - PTRATIO correlation = -0.51
MEDV - CRIM correlation = -0.39
PTRATIO - CRIM correlation = 0.29

Figure GBR_Relative_Importance_2019314213317 uses GradientBoostingRegressor places
attributes PTRATIO and CRIM within top half of the attributes.

The Linear regression training/testing models returned r_score values of 32 and 30 which is low.  

Docker - How to run the script
Built my image
lcler@LAPTOP-D2T6EOO0 MINGW64 ~/Desktop/class8-hwk (master)
$ docker build -t scikit-image8 .

Checked if image was created
lcler@LAPTOP-D2T6EOO0 MINGW64 ~/Desktop/class8-hwk (master)
$ docker images
REPOSITORY            TAG                 IMAGE ID            CREATED             SIZE
scikit-image8         latest              0c01c97da15c        3 minutes ago       922MB

Running the image
lcler@LAPTOP-D2T6EOO0 MINGW64 ~/Desktop/class8-hwk (master)
$ docker run -t -v /${PWD}:/${PWD} -w/${PWD} scikit-image8

....will generate ex.

...Predicted True Price Scatter Chart on Full Data (Linear Regression)

RMS: 4.502022301087212

...Predicted True Price Scatter Chart on Full Data (Gradient Boosting Regressor)

...Price Target Data Histogram Generated

...Scatter Charts Feature vs Target Generated

...Decision Tree Regressor Scatter Chart Generated

...Heatmap Generated

...Preparing the data for training the model

X_train: (404, 2)
X_test: (102, 2)
Y_train: (404,)
Y_test: (102,)

Model performance testing and training performed on CRIM, PTRATIO vs MDEV

The model performance for training set
--------------------------------------
RMSE is 7.6144001885660035
R2 score is 0.3250530330112821


The model performance for testing set
--------------------------------------
RMSE is 7.389189284263164
R2 score is 0.30262587374686933

GradientBoostingRegressor Model MSE
Mean Squared Error: 6.8068

...Relative Importance and Training-Test Set Deviance Chart Generated




 -> Execution Completed




***** The python script will generate charts and generate values.
