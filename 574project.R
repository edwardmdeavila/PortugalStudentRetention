rm(list=ls()); gc()
# setwd('/Users/edwardmdeavila/Desktop/574/')
setwd("/Users/eduardodeavila/Desktop/574/project")

clean_set <- read.csv('cleaned_dataset.csv', head = T, stringsAsFactors = T)

#### check for missing ####
library(Amelia)
missmap(clean_set) # no missing !

#### transform categories into dummies ####
library(fastDummies)
dummy_clean_set <- dummy_columns(clean_set, select_columns = c("school","sex", "address", "famsize", 
  "Pstatus", "Mjob", "Fjob", "reason", "guardian", "schoolsup",
  "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic"),
  remove_most_frequent_dummy = T,
  remove_selected_columns = T)
View(dummy_clean_set)

#### Collinearity check ####
library(corrplot)
par(mfrow = c(1, 1))
corrplot(cor(na.omit(data)), method="number", number.cex = .6)
# can change to method = "circle"

#### check the distribution of continuous variables ####
par(mfrow=c(1, 2)) # par -> parameters, mfrow, how many graphs side by side
# mfrow = c(# of rows, # of graphs per row)
hist(dummy_clean_set$age)
boxplot(dummy_clean_set$age) 

par(mfrow=c(1, 2)) 
hist(dummy_clean_set$G1) 
boxplot(dummy_clean_set$G1) 

par(mfrow=c(1, 2)) 
hist(dummy_clean_set$G2) 
boxplot(dummy_clean_set$G2) 

par(mfrow=c(1, 2)) 
hist(dummy_clean_set$G3) 
boxplot(dummy_clean_set$G3) 

par(mfrow=c(1, 2)) 
hist(dummy_clean_set$absences) 
boxplot(dummy_clean_set$absences) 

library(writexl)
write_xlsx(dummy_clean_set, 
           path="/Users/eduardodeavila/Desktop/574/project//project574.xlsx")

library(tidyverse)
data <- read.csv('project574.csv', head = T, stringsAsFactors = T)
data <- data %>% 
  select(G3, everything()) %>%
  select(-G1, -G2)
head(data)

set.seed(123)
n.train = floor( nrow(data)*0.6 )
ind.train = sample(1:nrow(data), n.train)
ind.test = setdiff(1:nrow(data), ind.train)


#### MLR ####
#### 60% training partition ####
library(tidyverse)
data <- read.csv('project574.csv', head = T, stringsAsFactors = T)
data <- data %>% 
  select(G3, everything()) %>%
  select(-G1, -G2)
head(data)


library(MASS)
obj.null = lm(G3 ~ 1, data = data[ind.train, ]) # only intercept, 1, is included in the model
obj.full = lm(G3 ~ ., data = data[ind.train, ])

#### forward selection ####
obj1 = step(obj.null, scope=list(lower=obj.null, upper=obj.full), direction='forward') # forward selection by Akaike information criterion (AIC)
summary(obj1) ## ending up with a full model
# r squared 0.3638

#### backward elimination ####
obj2 = step(obj.full, scope=list(lower=obj.null, upper=obj.full), direction='backward') # start with full and end with null; reversed comparing to forward
summary(obj2)
# r squared: 0.3652

#### stepwise selection ####
obj3 = step(obj.null, scope=list(lower=obj.null, upper=obj.full), direction='both') # start with full and end with null
summary(obj3)
# r squared: 0.3638

#### best subset #### 
library(leaps)
obj4 = regsubsets(G3 ~ ., data = data[ind.train, ], nvmax=20) ## allow up to 20 variables in the model;
summary(obj4)

par(mfrow=c(1,1)) # par(mfrow=c(m,n)) allows us to put m*n figures in a single plot; if m=n=1, only one figure in the plot
plot(obj4, scale="adjr2") # black color indicates a variable is used in the model
plot(obj4, scale="bic")

# compare prediction results of forward, backward, stepwise
library(Metrics)

# forward
yhat1 = predict(obj1, newdata = data[ind.test, ])
rmse(data[ind.test, 'G3'], yhat1) ## RMSE for test data
# 2.9937

# backward
yhat2 = predict(obj2, newdata = data[ind.test, ])
rmse(data[ind.test, 'G3'], yhat2)
# 3.005

# stepwise
yhat3 = predict(obj3, newdata = data[ind.test, ])
rmse(data[ind.test, 'G3'], yhat3)
# 2.9937

#### Winner: Backwards Elimination ####

#### KNN ####
# detach("package:class", unload=TRUE)
##### functions needed for knn prediction #######
library(dplyr) #Package simplifies functions
library(FNN) #Needed to run KNN prediction
one.pred = function(xnew, xtrain, ytrain, k, algorithm) {
  ind = knnx.index(xtrain, matrix(xnew, 1), k=k, algorithm=algorithm)
  mean(ytrain[ind])
}

knn.predict = function(Xtrain, ytrain, Xtest, k=5, algorithm = 'kd_tree') {
  ypred = apply(Xtest, 1, one.pred, xtrain = Xtrain, ytrain = ytrain, k=k, algorithm=algorithm)
  return(ypred)
}

knn.predict.bestK = function(Xtrain, ytrain, Xtest, ytest, k.grid = 1:20, algorithm='kd_tree') {
  fun.tmp = function(x) {
    yhat = knn.predict(Xtrain, ytrain, Xtest, k = x, algorithm=algorithm) # run knn for each k in k.grid
    rmse = (yhat - ytest)^2 %>% mean() %>% sqrt()
    return(rmse)
  }
  ## create a temporary function (fun.tmp) that we want to apply to each value in k.grid
  error = unlist(lapply(k.grid, fun.tmp))
  out = list(k.optimal = k.grid[which.min(error)], error.min = min(error))
  return(out)
}
#################################################

#### Define x & y ####
Xtrain = data[ind.train,2:ncol(data)]
Xtest = data[ind.test,2:ncol(data)]
ytrain = data[ind.train,1]
ytest = data[ind.test,1]

#### Specify K values ####
ypred = knn.predict(Xtrain, ytrain, Xtest, k = 3)
plot(ytest, ypred)
abline(0, 1, col='red') #Gives linear line: Intercept & Slope

objknn = knn.predict.bestK(Xtrain, ytrain, Xtest, ytest, k.grid = 1:18) 
objknn # optimal k = 13

## rerun with the best k
ypred1 = knn.predict(Xtrain, ytrain, Xtest, k = objknn$k.optimal)
plot(ytest, ypred1)
abline(0, 1, col='red')

(ytest - ypred1)^2 %>% mean() %>% sqrt()
sqrt(mean((ytest - ypred1)^2))

knn.predict.bestK(Xtrain, ytrain, Xtest, ytest) # rmse = 3.19913
