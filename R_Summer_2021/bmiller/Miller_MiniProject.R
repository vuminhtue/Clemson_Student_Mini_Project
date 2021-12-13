# Requirement:
# 1. You can write the script in RStudio (on Palmetto's Open OnDemand or your PCs/Macs) 
#    and save it as your_username.R and send it to my email: tuev@clemson.edu. You can 
#    also upload to your github page and share to me
# 
# Note: in order to get the Certificate of Attendence, you should send me the 
# R script on or before 09/30/2021.
# Data
# a. You can use any data in your field of expertise (most preferred method). 
#    If so, please send along the dataset with the Rscript
# b. You can also use Kaggle data which can be found here (second preferred method). 
#    Please elaborate in the ipynb how did you retrieve the data
# c. You can use use Titanic data from our repo. Description for Titanic data can be 
#    found here (third preferred method)
# Method
# # In the R solution file I would like to see the following:
#  
# a. Clearly state the objective of the mini-project on Supervsised Machine Learning
#    My goal in this mini-project is to learn how to clean data and conduct a multivariate
#    regression using data from the General Social Survey 2014  

# b. Brief explanation about the data that you will be using: source, predictors, 
#    predictand
#     The data are from the 2014 General Social Survey (GSS)(Smith et al. 2017). 
      # The GSS is a biennial, nationally representative survey of Americans age 18 and over using face-to-face
      # interviews. The response rate is 69.2%. Although the total sample size is 3,842, a total
      # of 1,304 participants were presented the items on the "Identity Module." 
      # Dependent Variable: opposition to affirmative action.
      # Independent Variables: x1 - importance of racial identity, x2 - pride in racial identity,
      # x3 - race, x4 - gender, x5 - family income

# c. Type of ML model output: Continuous or Classification?
      # I will conduct a logistic regression model 
# d. Read in the data
library(caret)
library(GGally)
getwd()
setwd("C:/Users/bmill148/Documents/RProjects")
gss2014 <- read.csv('GSS_2014.csv')

View(gss2014)
names(gss2014)

# e. Clean & Standardized the input data if needed

# Dependent Variable: AFFRMACT NA=0, 8=DK, 9=no answer
# 1=strongly support, 2=support, 3=oppose, 4=strongly oppose
# Some people say that because of past discrimination, blacks should be 
# given preference in hiring and promotion. Others say that such preference 
# in hiring and promotion of blacks is wrong because it discriminates 
# against whites. What about your opinion? Are you for or against 
# preferential hiring and promotion of blacks? Do you favor preference 
# in hiring and promotion strongly or not strongly? Do you oppose preference
# in hiring and promotion strongly or not strongly?
summary(gss2014$AFFRMACT)
table(gss2014$AFFRMACT)
gss2014$opposeaff <- gss2014$AFFRMACT
gss2014$opposeaff[gss2014$opposeaff<1] <- NA
gss2014$opposeaff[gss2014$opposeaff>5] <- NA
gss2014$opposeaff[gss2014$opposeaff<3] <- 0
gss2014$opposeaff[gss2014$opposeaff>2] <- 1
table(gss2014$opposeaff)
sum(is.na(gss2014$opposeaff))
gss2014$opposeaff <- as.factor(gss2014$opposeaff)
levels(gss2014$opposeaff) <- c("Agree", "Oppose")
table(gss2014$opposeaff)

# IDRACIMP: -1=NA (n=2538), 98=DK (n=11), 99=No answer (n=6)
# How much is being [respondent's race] an important part of how you see yourself?
gss2014$rac_imp <- gss2014$IDRACIMP
gss2014$rac_imp[gss2014$rac_imp<0] <- NA
gss2014$rac_imp[gss2014$rac_imp>10] <- NA

# IDRACIMP: -1=NA (n=2538), 98=DK (n=11), 99=No answer (n=6)
# How proud are you to be [respondent's race]?
gss2014$rac_prd <- gss2014$IDRACPRD
gss2014$rac_prd[gss2014$rac_prd<0] <- NA
gss2014$rac_prd[gss2014$rac_prd>10] <- NA

# RACE: What race do you consider yourself?
# 1=white, 2=black, 3=other
gss2014$race_factor <- gss2014$RACE
gss2014$race_factor <- as.factor(gss2014$RACE)
levels(gss2014$race_factor) <- c("White", "Black", "Other")
print(gss2014$race_factor)

# SEX: Respondent's sex 1=male, 2=female
gss2014$gender <- as.factor(gss2014$SEX)
levels(gss2014$gender) <- c("male", "female")
print(gss2014$gender)

# INCOME06 In which of these groups did your total family income, from 
# all sources, fall last year before taxes, that is?
# RANGE: 1 (<$1k ) to 25 (>$150), 26=Refused, 98=DK
gss2014$f_income <- gss2014$INCOME06
gss2014$f_income[gss2014$f_income>25] <- NA
table(gss2014$f_income)

summary(glm(gss2014$opposeaff~gss2014$rac_imp+gss2014$rac_imp+gss2014$rac_prd
            +gss2014$race_factor+gss2014$gender+gss2014$f_income
            ,data=gss2014,family="binomial"))
# Omit NA
      # I decided to remove people that were NA due to the use of a split ballot
      # used in the GSS 2014. They have 3 ballots and not everyone is surveyed
      # for all items. There are weighting variables in the data (which I will 
      # need to learn to apply in R). For the sake of this project, I removed all NAs
newgss2014 <- na.omit(gss2014)


# f. Split data to training/testing (You can also use Cross Validation if needed, 
#    not required)
sum(is.na(newgss2014$opposeaff))
sum(is.nan(newgss2014$opposeaff))

ind1 <- createDataPartition(newgss2014$opposeaff,p=0.6,list=FALSE,times=1)
ind1
training <- newgss2014[ind1,]
testing <- newgss2014[-ind1,]

# g. You can use any Regularization (variable selection) or PCA if needed (not required)
      #Not using either

# h. Construct Machine Learning model to training set and explain why do you want 
#    to use that algorithm (any model is fine for me)

   # I selected this model because I am running a logistic regression to examine
   # the probabilities associated with supporting verses opposing race-based 
   # affirmative action. 
ModFit_glm <- train(opposeaff~rac_imp+rac_imp+rac_prd
                    +race_factor+gender+f_income
                    ,data=training,method="glm")

# i. Apply Machine Learning model to predict the output from testing set
summary(ModFit_glm$finalModel)

# j. Evaluate the output using any of the given method in chapter 4
predictions <- predict(ModFit_glm,testing)
confusionMatrix(predictions, testing$opposeaff)

# k. Confirm if your ML model is good opposed or bad?
   # The accuracy of the model is .785 and the p-value is .785, which indicates
   # this is a poor fitting model.  








