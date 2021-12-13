#Objective of the mini-project on Supervised Machine Learning 
## Objective: To examine a data set from my experiment 
#             To analyze what factors could predict the pathogenicity of a mutation on MSH1 protein.
#             To build a ML model to predict the pathogenicity of a mutation on MSH1 protein.
#Brief explanation about the data that you will be using: source, predictors, predictand 
##Source:My lab results
##Predictors: Mutations Flag pathogenicity Folding_ddG_AVG Binding_ddG_AVG Evolutionary_score RMSD_AVG_2000 RMSDmut_wt_2000 RMSF_MUT_MT RMSF_mut_10_residues RMSF_wt_10_residues Neighbour_RMSF_MUT_WT Hond_mut Hond_wt Bfactor_wt rSASA_MSH2_only_mut rSASA_MSH2_only_wt Protein_Distance 
##Predictand: pathogenicity(bengin or pathogenic) of the mutation

#Type of ML model output: Continuous or Classification? 
##The out put is Classification

#Read in the data Clean & Standardized the input data if needed Split data to training/testing (You can also use Cross Validation if needed, not required) 
rm(list=ls())
dev.off()
library(caret)
library(glmnet)
library(plotmo)
mutation <- read.table("/Users/bohuawu/Study/mutation_All_5.txt",header = TRUE)
mutation1 <- mutation[2:24]
set.seed(123)
indT <- createDataPartition(y=mutation1$Flag,p=0.6,list=FALSE)

training <- mutation1[indT,]
testing  <- mutation1[-indT,]

#You can use any Regularization (variable selection) or PCA if needed (not required) 

y <- training$Flag
x <- training[,-c(1,2)]
x <- as.matrix(x)

cvfit_LASSO    <- cv.glmnet(x,y,alpha=1)
plot(cvfit_LASSO)

log(cvfit_LASSO$lambda.min)
log(cvfit_LASSO$lambda.1se)

coef(cvfit_LASSO,s=cvfit_LASSO$lambda.min)
coef(cvfit_LASSO,s=cvfit_LASSO$lambda.1se)

Fit_LASSO <- glmnet(x,y,alpha=1)
plot_glmnet(Fit_LASSO,label=TRUE,xvar="lambda",
            col=seq(3,24),,grid.col = 'lightgray')
xtest <- testing[,-c(1,2)]
xtest <- as.matrix(xtest)
predict_LASSO <- predict(cvfit_LASSO,newx=xtest,s="lambda.1se")

postResample(predict_LASSO,testing$Flag)


############

##Based on the results of lambda.1se, the following variables are selected for prediction
#Folding_ddG_AVG 
#Evolutionary_score 
#RMSF_MUT_MT        
#Bfactor_wt            
#rSASA_MSH2_only_mut   
#Protein_Distance   
#polarity              
              

#########################



training_mut <- training[,c(1,3,5,14,16,17,19,21)]
testing_mut  <- testing[,c(1,3,5,14,16,17,19,21)]

########################

#Construct Machine Learning model to training set and explain why do you want to use that algorithm (any model is fine for me) 
#Construct Machine Learning model to training set and explain why do you want to use that algorithm (any model is fine for me) 
ModFit_KNN <- train(pathogenicity~.,training_mut,method="knn",preProc=c("center","scale"),tuneLength=20)
#ModFit_KNN <- train(pathogenicity~.,training,method="knn",preProc=c("center","scale"),tuneLength=20)

ggplot(ModFit_KNN$results,aes(k,AccuracySD))+
  geom_point(color="blue")+
  labs(title=paste("Optimum K is ",ModFit_KNN$bestTune),
       y="Error")

#Apply Machine Learning model to predict the output from testing set 
predict_KNN<- predict(ModFit_KNN,newdata=testing_mut)

#Evaluate the output using any of the given method in chapter 4 
confusionMatrix(predict_KNN,as.factor(testing_mut$pathogenicity))



#Confirm if your ML model is good or bad?
##This ML model is good, because the accuracy of the model is 0.875, sensitivity os 0.8182, and the specificity is 0.9231. 
