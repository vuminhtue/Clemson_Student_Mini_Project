# Training module:
library(GGally)
library(caret)
library(kernlab)
library(DMwR)
########################
# Load the datasets
########################

df_train = read.csv("train.csv")
df_test = read.csv("test.csv")

head(df_train)
head(df_test)

str(df_train)  # check the structure of the data


#######################
# Split 'train.csv' data into training dataset and cross-validation(cv) set
#######################

sum(is.na(df_train))  # check for missing values

## Treat missing values with the kNN approach

?knnImputation #fills in NA values with the values of the nearest neighbors
df_train = knnImputation(df_train)

sum(is.na(df_train))  # check for missing values



set.seed((1234))
train = df_train[sample(nrow(df_train),600),]
train
str(train)


cv = df_train[-train$PassengerId,]
cv
str(cv)




#######################
# Create the model using Logistic Regression
######################

glm.Surv=glm(Survived~Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,data=train, model = TRUE, family=binomial)
summary(glm.Surv)




########################
# Check the model on the cross validation data
########################

cv$Survived = NA
glm_pred_surv = predict(glm.Surv, newdata = cv, type="response")
glm_pred_surv

#Take the reference for survival as 50% probability
glm_pred_surv[glm_pred_surv > 0.5] = 1
glm_pred_surv[glm_pred_surv < 0.5] = 0
glm_pred_surv





##########################
# Use our model on the test data to predict the survival
##########################
test = df_test
test$Survived <- 0
str(test)


prediction_final <- predict(glm.Surv, test)
prediction_final
prediction_final[prediction_final>0.5] = 1
prediction_final[prediction_final<0.5] = 0
prediction_final


my_solution <- data.frame(PassengerId = test$PassengerId, Survived =prediction_final)
my_solution
test$Survived <- my_solution$Survived
table(test$Survived)
prop.table(table(test$Sex, test$Survived),1)
write.csv(my_solution, file = "my_solution.csv", row.names = FALSE)

