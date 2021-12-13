#1 Objective - To find factors that affect the water potability most
#2 Data Source - Kagggle https://www.kaggle.com/adityakadiwal/water-potability/download
#  Predictand - Potability
#  Predictors - "ph","Hardnes",Solids","Chloramines","Sulfate","Conductivity","Organic_carbon","Trihalomethanes","Turbidity"
#3 Type of ML output - Classification

#4 Data Read
data=read.csv("Downloads/water_potability.csv")
colnames(data)
data
summary(data)
dim(data)

#5 Data Cleaning

data_clean=na.omit(data)
dim(data_clean)

#Data Imputing(Standardizing)

NA2mean <- function(x) replace(x, is.na(x), mean(x, na.rm = TRUE)) 
impute_data <-replace(data, TRUE, lapply(data, NA2mean))
dim(impute_data)

library(caret)
PreImputeBag <- preProcess(data,method="bagImpute") 
imputeb_data <- predict(PreImputeBag,data)
dim(imputeb_data)

#6 Data-Split

dat1 <- createDataPartition(y=imputeb_data$Potability,p=0.7,list=FALSE,times=1)
training <- imputeb_data[dat1,]
testing <- imputeb_data[-dat1,]
dim(training)
dim(testing)

#7 Regularization - Ridge
library(glmnet) 
library(plotmo)
y <- training$Potability
x <- training[,-c(9,10)] 
x <- as.matrix(x)
cvfit_Ridge <- cv.glmnet(x,y,alpha=0) 
plot(cvfit_Ridge)
log(cvfit_Ridge$lambda.min) 
log(cvfit_Ridge$lambda.1se) 
coef(cvfit_Ridge,s=cvfit_Ridge$lambda.1se)

Fit_Ridge <- glmnet(x,y,alpha=0,standardize = TRUE) 
plot_glmnet(Fit_Ridge,label=TRUE,xvar="lambda",
col=seq(1,9),grid.col = 'lightgray')
xtest <- testing[,-c(9,10)] 
xtest <- as.matrix(xtest)

predict_Ridge <- predict(cvfit_Ridge,newx=xtest,s="lambda.1se") 
cor.test(predict_Ridge,testing$Potability)
postResample(predict_Ridge,testing$Potability)




#8 Machine Learning Model
# I am using Random Forest model as it has better efficiency for classification models.

prop.table(table(training$Potability))
prop.table(table(testing$Potability))

library(rpart)
library(rpart.plot)
fit <- rpart(Potability~., data = training, method = 'class')
rpart.plot(fit, extra = 106)


#9 Predicting output 

predict_unseen <-predict(fit, testing, type = 'class')


#10 Evaluating output
# I chose Confusion Matrix beacause the predictand is classified.
table_mat <- table(testing$Potability, predict_unseen)
table_mat
accuracy_Test <- sum(diag(table_mat)) / sum(table_mat)
accuracy_Test

#11 The model has achieved an efficiency of 65%. it is good when compared to other models,
# but is not an satisfactory model
