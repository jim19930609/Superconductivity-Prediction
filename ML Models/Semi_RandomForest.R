library("randomForest")

# Readin Data into DataFrame
sem = read.csv('C:\\Users\\jim19\\Desktop\\Statistics_Project\\train.csv')

# Split into Traing & Test
smp_size <- floor(0.75 * nrow(sem))
train_ind <- sample(seq_len(nrow(sem)), size = smp_size)

sem_train = sem[train_ind, ]
sem_test = sem[-train_ind, ]

X_train = as.matrix(sem_train[-82])
Y_train = as.matrix(sem_train["critical_temp"])

X_test = as.matrix(sem_test[-82])
Y_test = as.matrix(sem_test["critical_temp"])


# Search for best mtry parameter
best_mtry = 0
minerr = Inf
steps = 5

n = length(names(sem_train))
lowend = max(sqrt(n)-steps, 0)
highend = min(sqrt(n)+steps, n)
for (i in lowend:highend){
  model = randomForest(sem_train$critical_temp ~ ., 
                        data=sem_train, 
                        mtry=i,
                        ntree=100)
  err = mean(model$mse)
  if (minerr > err) {
    minerr = err
    best_mtry = i
  }
  print(err)
}

sem.rf = randomForest(x=X_train,
                      y=Y_train,
                      mtry=10,
                      ntree=100,
                      importance=TRUE)
# Convergence
plot(sem.rf)

# Variable Importance
importance(x=sem.rf)
varImpPlot(sem.rf)

# Train Error
pred_train = predict(object=sem.rf, newdata=sem_train)
TrainRMSE = sqrt(sum((pred_train - Y_train)^2)/length(pred_train))
TrainRMSE

# Test Error
pred_test = predict(object=sem.rf, newdata=sem_test)
TestRMSE = sqrt(sum((pred_test - Y_test)^2)/length(pred_test))
TestRMSE
