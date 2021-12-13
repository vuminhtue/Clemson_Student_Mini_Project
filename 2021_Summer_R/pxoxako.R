### Mini-Project: Machine Learning using R w/ CCIT ###
###               by Phoebe Xoxakos               ###
###             pxoxako@g.clemson.edu             ###

## Clearly state the objective of the mini-project on Supervised Machine Learning.
# The objective of this mini-project is to predict Positive.affect (DV) from Freedom.to.make.life.choices (IV).

## Data w/ Explanation
## Read in Data
# The "World Happiness Report 2021" was downloaded from kaggle and imported into the Global Environment manually.
DataPanelWHR2021C2$Freedom.to.make.life.choices <- as.numeric(as.factor(DataPanelWHR2021C2$Freedom.to.make.life.choices))
DataPanelWHR2021C2$Positive.affect <- as.numeric(as.factor(DataPanelWHR2021C2$Positive.affect))

## Type of ML model output: Continuous or Classification?
# Continuous 

## Clean & Standardized the input data if needed
install.packages("caret")
install.packages(Ggally)

#2.4.2 Pre-processing with missing value
View(DataPanelWHR2021C2)
new_DataPanel <- na.omit(DataPanelWHR2021C2)

## Split data to training/testing (You can also use Cross Validation if needed, not required)
# 3.1 Data Spliting using Data Partition
set.seed(123)
ind1 <- createDataPartition(y=new_DataPanel$Freedom.to.make.life.choices,p=0.6,list=FALSE,times=1)
head(ind1,10)
#list=FALSE, prevent returning result as a list
#times=1 to create the resample size. Default value is 1.
training <- new_DataPanel[ind1,]
View(training)
testing  <- new_DataPanel[-ind1,]
View(testing)

## You can use any Regularization (variable selection) or PCA if needed (not required)
# Supervised Learning w/ Linear Regression
# I utilized linear regression, a form of supervised machine learning, to predict one value (Positive
# Affect, DV) based on the value of another variable (Feedom to make life chioces, IV) because it is 
# the most appropriate analysis for the data. 

set.seed(123)
# single liner model regression
#Fit a Linear model using method=lm
ModFit <- train(Positive.affect~Freedom.to.make.life.choices,data=training,
                preProcess=c("center","scale"),
                method="lm")
summary(ModFit$finalModel)

#Apply trained model to testing data set and evaluate output
prediction <- predict(ModFit,testing)
cor.test(prediction,testing$Positive.affect)
postResample(prediction,testing$Positive.affect)


## Evaluate the output using any of the given method in chapter 4
# The output indicates that Freedom to Make Life Choices (IV) does significantly impact one's Positive
# Affect (DV) (B = 62.80; t = 25.88; p < .001). 


## Confirm if your ML model is good or bad?
# The model is good. 

