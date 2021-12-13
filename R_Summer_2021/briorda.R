# Bryan Riordan | 06.30.21
# MiniProject
# 
# The objective of this project is to create a model comparing prices of computers
# with their respective part bases; speed, harddrive storage, RAM, screen size, and 
# possession of a cd-drive. The continuous data has been retrieved from Kaggle using
# a csv simply titled 'Computers' and is linked below. Data index has been reduced from
# roughly 6k to just 200 computers. The csv file was downloaded and then uploaded into
# personal dropbox files for use.
#
# Data Link: https://www.kaggle.com/kingburrito666/basic-computer-data-set
#
# My prediction is that the higher priced computers will have a higher speed, more storage,
# more RAM, larger screen size, and have a cd-drive. Furthermore, I believe that speeds 
# will carry the most weight on determining prices. A linear regression model will be used
# for each category against prices. While a multi-linear regression model can point out the
# correlation between all factors, by doing each we'll be able to determine which has 
# the most influence on the price.


# Clear Workspace
rm(list=ls())
dev.off()


# --- DATA SETUP --- #

# Read in Data
data = read.csv("Computers.csv")

# Data Partition
library(caret)
set.seed(74319429)
ind = createDataPartition(y=data$screen,p=0.6,list=FALSE,times=1)
training = data[ind,]
testing = data[-ind,]

# Cross Validation
fitControl = trainControl(method="cv",number=10)
model = train(price~speed,data=training,trControl=fitControl,method="lm")
print(model)
predict = predict(model,testing)
print(predict)

# Correlation Coefficient / Determination / RMS
cor(predict,testing$price)
cor.test(predict,testing$price)
postResample(predict,testing$price)


# --- MODELS --- #

# All --- Multi-Linear Regression Model
ModFit_MLR = train(price~speed+hd+ram+screen+cd,data=training,preProcess=c("center","scale"),method="lm")
summary(ModFit_MLR$finalModel)
prediction_MLR = predict(ModFit_MLR,testing)
postResample(prediction_MLR,testing$price)
cor.test(prediction_MLR,testing$price)
postResample(prediction_MLR,testing$price)

# Speed --- Linear Regression Model
ModFit_speed = train(price~speed,data=training,preProcess=c("center","scale"),method="lm")
summary(ModFit_speed$finalModel)
prediction_speed = predict(ModFit_speed,testing)
cor.test(prediction_speed,testing$price)
postResample(prediction_speed,testing$price)

# Harddrive --- Linear Regression Model
ModFit_hd = train(price~hd,data=training,preProcess=c("center","scale"),method="lm")
summary(ModFit_hd$finalModel)
prediction_hd = predict(ModFit_hd,testing)
cor.test(prediction_hd,testing$price)
postResample(prediction_hd,testing$price)

# RAM --- Linear Regression Model
ModFit_ram = train(price~ram,data=training,preProcess=c("center","scale"),method="lm")
summary(ModFit_ram$finalModel)
prediction_ram = predict(ModFit_ram,testing)
cor.test(prediction_ram,testing$price)
postResample(prediction_ram,testing$price)

# Screen Size --- Linear Regression Model
ModFit_screen = train(price~screen,data=training,preProcess=c("center","scale"),method="lm")
summary(ModFit_screen$finalModel)
prediction_screen = predict(ModFit_screen,testing)
cor.test(prediction_screen,testing$price)
postResample(prediction_screen,testing$price)

# CD Drive --- Linear Regression Model
ModFit_cd = train(price~cd,data=training,preProcess=c("center","scale"),method="lm")
summary(ModFit_cd$finalModel)
prediction_cd = predict(ModFit_cd,testing)
cor.test(prediction_cd,testing$price)
postResample(prediction_cd,testing$price)


# --- RESULTS / CONCLUSION --- #

# All ---- Prediction: 87.9% correlation
#          R^2: 77.3% correlation
# 
# Speed -- Predction: 51.2%
#          R^2: 26.2%
#
# HD ----- Prediction: 81.7%
#          R^2: 66.8%
#
# RAM ---- Prediction: 52.6%
#          R^2: 27.6%
#
# Screen - Prediction: 27.0%
#          R^2: 7.3%
#
# CD ----- Prediction: 8.6%
#          R^2: 0.7%
#
#
# The inital MLR regression model shows that there is a strong correlation between
# prices and its varying factors. With this being confirmed, the individual tests
# can be conducted with confidence in determining the major factor. Looking at the
# coefficient of determination for each test, the harddrive turned out to have the
# most influence in prices. My speed prediction was not only incorrect, but even
# fell behind the RAM factor. However, as expected, the screen size and cd drives
# hardly affected the results.
#
# Although I'm able to determine the major factor on prices here, the model doesn't
# quite reach the predicted correlation values. I do still think the model worked
# well enough and as intended, but may not be as accurate as it can be. This might
# be corrected through a different model or perhaps by using more items in the
# overall dataset.
