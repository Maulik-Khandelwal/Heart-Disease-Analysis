library(dplyr)

#sigmoid function, inverse of logit
sigmoid <- function(z){1/(1+exp(-z))}

#cost function
cost <- function(theta, X, y){
  m <- length(y) # number of training examples
  h <- sigmoid(X %*% theta)
  J <- (t(-y)%*%log(h)-t(1-y)%*%log(1-h))/m
  J
}

#gradient function
grad <- function(theta, X, y){
  m <- length(y) 
  
  h <- sigmoid(X%*%theta)
  grad <- (t(X)%*%(h - y))/m
  grad
}


#logistic regression
logisticReg <- function(X, y){
  X <- na.omit(X)
  y <- na.omit(y)
  X <- mutate(X, bias =1)
  X <- as.matrix(X[, c(ncol(X), 1:(ncol(X)-1))])
  y <- as.matrix(y)
  theta <- matrix(rep(0, ncol(X)), nrow = ncol(X))
  costOpti <- optim(theta, fn = cost, gr = grad, X = X, y = y)
  return(costOpti$par)
}

# probability of getting 1
logisticProb <- function(theta, X){
  X <- na.omit(X)
  X <- mutate(X, bias =1)
  X <- as.matrix(X[,c(ncol(X), 1:(ncol(X)-1))])
  return(sigmoid(X%*%theta))
}

# y prediction
logisticPred <- function(prob){
  return(round(prob, 0))
}

#accuracy function
accuracy = function(y_test,pred){
  return(length(which(y_test==pred))/length(y_test))
}

df <- read.csv("D:/DS-CP/heart.csv")
head(df)

set.seed(23)
train = sample(1:nrow(df), 0.7*nrow(df))
test = -train
training_data = df[train,]
testing_data = df[test,]
train_y = training_data$target
train_X = training_data[, !(colnames(training_data) %in% c("target"))]
test_y = testing_data$target
test_X = testing_data[, !(colnames(training_data) %in% c("target"))]


# training
theta <- logisticReg(train_X, train_y)
a <- logisticProb(theta,test_X)
a
ypred <- logisticPred(a)

#accuracy
acc <- accuracy(test_y,ypred)
print(acc)

ypred <- as.factor(ypred)
test_y <- as.factor(test_y)
library(caret)
library(e1071)
confusionMatrix(ypred , test_y)

