rm(list = ls())
dev.off()
#Objective: To apply machine learing algorithm to the fatigue test data
#and develop abd train a model that can demonstrate 
#the load displcaement behaviour of the specimen during fatigue test with good accuracy

#Source: Data used is collected from fatigue test of bistable composite laminate

#Predictor: Displacement 
#Predictand: Load



#ML model output: Continuous

#Impute Missing Values

Fatigue_Data<-read.csv("Fatigue_Test_Data.CSV")
Load_Displacement_Data <- Fatigue_Data[,-c(1,2,3)]
Displcanement_Time_Data<-Fatigue_Data[,-c(1,3,5)]
Load_Time_Data<-Fatigue_Data[,-c(1,3,4)]

#Impute Missing Values
PreImputeBag <- preProcess(Fatigue_Data,method="bagImpute")
Fatigue_Data_imp <- predict(PreImputeBag,Fatigue_Data)

#Data Partition
ind1 <- createDataPartition(y=Fatigue_Data_imp$Load,p=0.6,list=FALSE,times=1)
training <- Fatigue_Data_imp[ind1,]
testing  <- Fatigue_Data_imp[-ind1,] 

#Regularization
library(glmnet)
library(plotmo)
y <- training$Load
x <- training[,-c(1,5)]
x <- as.matrix(x)
cvfit_LASSO<- cv.glmnet(x,y,alpha=1)
plot(cvfit_LASSO)
log(cvfit_LASSO$lambda.min)
log(cvfit_LASSO$lambda.1se)
coef(cvfit_LASSO,s=cvfit_LASSO$lambda.min)
coef(cvfit_LASSO,s=cvfit_LASSO$lambda.1se)
Fit_LASSO <- glmnet(x,y,alpha=1)
plot_glmnet(Fit_LASSO,label=TRUE,xvar="lambda",
            col=seq(1,8),,grid.col = 'lightgray')

#Regularization demonstrates that there is no need to reduce the number of covariates

#Train the model using Polynomial Regression since correlation of load with displacement and time
#for bistable laminate can be represented with a polynonmial function with high accuracy

modFit_poly <- train(Load~poly(Disp,3)+poly(Elapsed.Time,3)++poly(Scan.Time,3),data=training,
                     preProcess=c("center","scale"),
                     method="lm")
summary(modFit_poly$finalModel)

#Make Prediction
prediction_poly <- predict(modFit_poly,testing)

#Evaluation
postResample(prediction_poly,testing$Load)

#This model model provides good result with very small valuea of RMSE and MAE
#It can be extended to include more poramaters like 
#stiffness, thermal and moisture expansion coefficients, temperature.
