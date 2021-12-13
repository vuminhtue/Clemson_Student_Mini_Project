# Gwendolyn Watson - Mini Project_CCIT Mini Project (July 6, 2021)

# 1 - Clearly state the objective of the mini project on Supervised Machine Learning 
## The purpose of this project is to use supervised machine learning to predict employee turnover.

# 2 - Brief explanation about the data 
## The data used for this project was imported from an Excel file downloaded from kaggle. The link to the data source can be found here: https://www.kaggle.com/davinwijaya/employee-turnover
## The outcome variable is employee turnover (turnover event or no turnover event).
## The predictors were employee tenure, employee gender, employee age, employee profession, the employee's previous company (pipeline company), presence of a training coach, supervisor gender, employee tax classification, employee transportation method to work, extraversion score, independences score, self-control score, anxiety score, and creativity score.

# 3 - Type of ML model output
## Classification (1 = turnover event, 0 = no turnover event)

# 4 - Read in data
library(readxl)
turnover <- read_excel("Clemson Spring 2021/Machine Learning in R Workshop/Potential Datasets for Mini Project/turnover_2.xlsx")
View(turnover_2)

# 5 - Clean and standardize the input data

library(caret)
library(GGally)
library(ggplot2)
ggpairs(data=turnover_2,aes(colour=as.factor(event)))

turnover_2$event <- as.factor(as.character(turnover_2$event))

##Check for and remove missing NA values
is.na(turnover_2)
View(turnover_2) #N = 1129
new_turnover_2 <- na.omit(turnover_2)
View(new_turnover_2) #N = 1129 - there was not any missing data so I will continue to use turnover_2

# 6 - Split data into training/testing
set.seed(123)
indTrain <- createDataPartition(y=turnover_2$event,p=0.6,list = FALSE)
turnover_training <- turnover_2[indTrain,]
turnover_testing  <- turnover_2[-indTrain,]
View(turnover_training)
View(turnover_testing)

# 7 - Construct ML model to training set and explain why that algorithm
## I chose to use logisitic regression because my model intends to predict a binary, categorical output(yes/no for turnover event).

##Build logistic regression modeling using GLM (generalized linear modeling) - ONLY WORKS FOR A BINARY OUTCOME
ModFit_glm <- train(event~.,data=turnover_2,method="glm")
summary(ModFit_glm$finalModel)

# 8 - Apply ML model to predict the output from testing set
predictions <- predict(ModFit_glm,turnover_testing)

# 9 - Evaluate the output 
## I used the Confusion Matrix, ROC plot, and AUC value to evaluate the ML output.
confusionMatrix(predictions, as.factor(turnover_testing$event)) #Accuracy = 65.85%, 95% CI = [0.6127, 0.7022], p value < .001

##Plot ROC and compute AUC
library(ROCR)
pred_prob <- predict(ModFit_glm,turnover_testing, type = "prob")
head(pred_prob)
data_roc <- data.frame(pred_prob = pred_prob[,'1'],
                       actual_label = ifelse(turnover_testing$event == '1', 1, 0))

roc <- prediction(predictions = data_roc$pred_prob,
                  labels = data_roc$actual_label)

plot(performance(roc, "tpr", "fpr"))
abline(0, 1, lty = 2)
auc <- performance(roc, measure = "auc")
auc@y.values
#used this website to help interpret the results: https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/

# 10 - Confirm if ML model is good or bad
## The purpose of this machine learning model was to predict whether or not an employee had a turnover event (i.e., terminated from job) based on predictors such as tenure, demographics, and personality characteristics. Because turnover is a binary outcome (yes vs. no), logistic regression was used. The results indicated that this was a decent logistic regression ML model. The confusion matrix reported an accuracy percentage of 65.85%, meaning that the turnover event was correctly classified in two-thirds of the cases.The ROC curve and AUC value (AUC = 70.91%) also reflect a model of fair quality. The ML model fairly accurately discriminates the employee cases into turnover event or no turnover event groups.