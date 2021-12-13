######################################
# Predicting Student Test Scores in R 
######################################
# The data used is taken from the following Kaggle page:
# https://www.kaggle.com/kwadwoofosu/predict-test-scores-of-students
# The csv file can bve found in my working directory on Palmetto: /home/armcdan/R

# The goal of this project is to use student dat to predict the outcome of
# their postest results (slightly unclear the exact definition of this...)

# Included in the data is information about the 
# school, class, and other characteristics of the indivdual students
# (More detail provided after the data is loaded)

# in this project I will use a random forest regression 
# to predict test scores of students
# I choose the random forest mainly due to its versatility and simplicity
# in this set we have mostly categorical features, along with a few numerical features
# however, the numerical features (namely the 'pretest') have high predictive power.
# Hopefully the random forest will effectively handle both the categorical and numercal features well.


############### Begin ###############
# clear variables and plots
rm(list=ls())
dev.off()

# May not need all these, but lets begin with loading some likely to use packages
library(ggplot2)
library(caret)
library(GGally)

# check (and set) working directory
getwd()
setwd("/home/armcdan/R")
getwd()


# load data file
# I've stored the relevant csv file in my wd
test_scores_df <- read.csv('test_scores.csv')


# check rows, columns number and names
dim(test_scores_df)
names(test_scores_df)

# what we will try to predict is the "posttest" values

head(test_scores_df)
summary(test_scores_df)

# before going further let's check for missing/inf values
sum(is.na(test_scores_df))
sum(apply(test_scores_df,c(1,2),is.nan))
sum(apply(test_scores_df,c(1,2),is.infinite))
# all give 0 so we have now need to worry about missing/inf values


# now that we've loaded the data and taken a quick look, 
# lets consider more closely what the features mean,
# and think about which ones may be good predictors

###################################
# Column Descriptions 
###################################
# "school" - identifier of the school, 
#            some schools may perform better than others overall 
#            so could be useful

# "school_setting" - urabn, rural, suburban. May be predictive
# "school_type" - non-public or public
# "classroom" - ID of classroom. Perhaps some classrooms have higher performance?
# "teaching_method" - experimental or standard
# "n_student" - number of students in the class
# "student_id" - Individual student ID, not a predictive quantity
# "gender" - male or female
# "lunch" - whether the student qualifies fro free lunch or not
#           likely a proxy for wealth/socioecenomic class

# "pretest" - pretest scores. Note sure if this refers to the students course 
#            grade prior to the test, or the score received on a practice test
#            either way I'd expect a high correlation to the posttest

# "posttest" - Our value to predict

# Note that we have 2 numerical features:
# n_student and pretest (along with the posttest target)

# 1 feature is not predictive (student_id)
# 5 features are categorical with 2-3 labels
#   (school_setting, school_type, teaching_method, gender, lunch)
# 2 features with many labels (classroom, school)

# The rest are categorical. This leads me to think that a random forest 
# or a K-nearest neighbors algorithm might be fitting
# (but lets not commit to anything just yet!)

###################################
# Exploratory Data Analysis (EDA) 
###################################


# Take a look at our numerical quantities
ggpairs(data=test_scores_df[c("n_student","pretest","posttest")])
# note the strong (as expected) correlation between pretest and posttest.

qplot(n_student, posttest, data=test_scores_df,
      color=factor(gender),
      shape=factor(school_type))

# for the features that have few categories, we can do some comparison plots to see 
# if there are any relations between the categories and scores
ggplot(test_scores_df, aes(lunch, posttest))+ geom_violin(aes(fill=lunch))
ggplot(test_scores_df, aes(school_type, posttest))+ geom_violin(aes(fill=school_type))
ggplot(test_scores_df, aes(gender, posttest))+ geom_violin(aes(fill=gender))
ggplot(test_scores_df, aes(teaching_method, posttest))+ geom_violin(aes(fill=teaching_method))
ggplot(test_scores_df, aes(school_setting, posttest))+ geom_violin(aes(fill=school_setting))

# of these, they all seem to have some predictive power except for gender, 
# which is essentially identical distributions for the two categories.

# For the final features classroom and class, it will be perhaps better to 
# look at the statistical quantities of the different categories (e.g. mean, std, etc.)

school_scores_mean <- aggregate(test_scores_df[c('pretest','posttest')], list(test_scores_df$school), mean)
class_scores_mean <- aggregate(test_scores_df[c('pretest', 'posttest')], list(test_scores_df$classroom), mean)


qplot(pretest, posttest, data=school_scores_mean)
qplot(pretest, posttest, data=class_scores_mean)


###################################
# Data Partition 
###################################
ind1 <- createDataPartition(y=test_scores_df$posttest,p=0.6,list=FALSE,times=1)
train <- test_scores_df[ind1,]
test  <- test_scores_df[-ind1,] 

# For the Training procedure we will remove features that we do 
# not deem to be important
# This will help with time/memory in building the Random Forest
# I will thus ignore: school_name, classroom, student_id, gender

# For comparison, I will test three models:
# "primary" - uses all reminaing features
# "pretest" - uses pretest as the only feature
# "nopretest" - same as primary, excluding pretest



primary_train = train[c("school_setting", "school_type","teaching_method","n_student","lunch","pretest", "posttest")]
primary_test = test[c("school_setting", "school_type","teaching_method","n_student","lunch","pretest", "posttest")]

pretest_train = train[c("pretest", "posttest")]
pretest_test = test[c("pretest", "posttest")]

nopretest_train = train[c("school_setting","school_type","teaching_method","n_student","lunch", "posttest")]
nopretest_test = test[c("school_setting","school_type","teaching_method","n_student","lunch", "posttest")]


ModFit_rf <- train(posttest~.,data=primary_train,method="rf",prox=TRUE)
predict_rf <- predict(ModFit_rf,primary_test)
postResample(predict_rf,primary_test$posttest)
qplot(primary_test$posttest, predict_rf)

ModFit_rf <- train(posttest~.,data=pretest_train,method="rf",prox=TRUE)
predict_rf <- predict(ModFit_rf,pretest_test)
postResample(predict_rf,pretest_test$posttest)
qplot(pretest_test$posttest, predict_rf)

ModFit_rf <- train(posttest~.,data=nopretest_train,method="rf",prox=TRUE)
predict_rf <- predict(ModFit_rf,nopretest_test)
postResample(predict_rf,nopretest_test$posttest)
qplot(nopretest_test$posttest, predict_rf)

# Our primary model which uses the 
# most feautures gives evaluation metrics of 
# RMSE      Rsquared  MAE 
# 3.2102805 0.9478829 2.5399172 
# I would say that generally speaking for this 
# fairly simple model this is an adequate performance.
# Very likely some more sophisticated analysis could imporve this however

# Using the most important feature 'pretest' as the only predictor
# actually preforms pretty well itself, giving 
# RMSE      Rsquared       MAE 
# 4.3297257 0.9054533 3.4544411
# While not as good as the primary model, this is still not
# bad for only having one feature. not surprising since we could see 
# from the outset that 'pretest' and 'posttest' were correlated

# The worst performance comes (unsurprisingly) when the 'pretest'
# data is no utilize. In this case the metrics are
# RMSE     Rsquared MAE 
# 5.399856 0.852189 3.986174 
# Here it would be beneficial to explore other options for
# ML models that could potentially improve the results
