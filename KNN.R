euclideanDist = function(a, b){
  d = 0
  for(i in c(1:(length(a)-1) ))
  {
    d = d + (a[[i]]-b[[i]])^2
  }
  d = sqrt(d)
  return(d)
}

knn_predict = function(X_train, y_train, X_test, k_value){
  list_labels = c((unique(y_train)))
  num_labels = length(list_labels)
  pred <- c()  
  for(i in c(1:nrow(X_test))){   
    eu_dist = c()          
    eu_char = c()
    counters = rep(c(0),num_labels)
    for(j in c(1:nrow(X_train))){
      eu_dist <- c(eu_dist, euclideanDist(X_test[i,], X_train[j,]))
      eu_char <- c(eu_char, (y_train[j]))
    }
    
    eu <- data.frame(eu_char, eu_dist) 
    
    
    eu <- eu[order(eu$eu_dist),]
    eu <- eu[1:k_value,]
    
    
    for(k in c(1:nrow(eu))){
      for(i in 1:num_labels){
        if((eu[k,'eu_char']) == list_labels[i]){
          counters[i] = counters[i] + 1
        }
      }
    }
    
    pred <- c(pred, list_labels[which.max(counters)])
    
  }
  return(pred) 
}


accuracy_knn = function(y_test, preds){
  return(length(which(y_test==preds))/length(y_test))
}



df <- read.csv("D:/DS-CP/heart.csv")
head(df)

library(caret)
preproc <- preProcess(df[,c(1:13,14)],method = c("range"))
df2 <- predict(preproc, df[,c(1:13,14)])
head(df2)

set.seed(2)
train = sample(1:nrow(df2), 0.7*nrow(df2))
test = -train
training_data = df2[train,]
testing_data = df2[test,]
train_y = training_data$target
train_X = training_data[, !(colnames(training_data) %in% c("target"))]
test_y = testing_data$target
test_X = testing_data[, !(colnames(training_data) %in% c("target"))]


preds = knn_predict(train_X, train_y, test_X, 5)
accuracy = accuracy_knn(test_y, preds)
print(accuracy)

preds <- as.factor(preds)
test_y <- as.factor(test_y)
library(caret)
library(e1071)
confusionMatrix(preds , test_y)

