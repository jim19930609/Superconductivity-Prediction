library("randomForest")

# Readin Data into DataFrame
sem = read.csv('C:\\Users\\jim19\\Desktop\\Statistics_Project\\ML Models\\train.csv')

# Split into Traing & Test
smp_size <- floor(0.8 * nrow(sem))
train_ind <- sample(seq_len(nrow(sem)), size = smp_size)

sem_train = sem[train_ind, ]
sem_test = sem[-train_ind, ]

X_train = as.matrix(sem_train[-82])
Y_train = as.matrix(sem_train["critical_temp"])

X_test = as.matrix(sem_test[-82])
Y_test = as.matrix(sem_test["critical_temp"])

# Fit Model
# Parameters Tuning
smp_size2 <- floor(0.8 * nrow(sem_train))
train_ind2 = sample(seq_len(nrow(sem_train)), size = smp_size2)

sem_train2 = sem_train[train_ind2, ]
sem_test2 = sem_train[-train_ind2, ]

X_train2 = as.matrix(sem_train2[-82])
Y_train2 = as.matrix(sem_train2["critical_temp"])

X_test2 = as.matrix(sem_test2[-82])
Y_test2 = as.matrix(sem_test2["critical_temp"])

# Search for best mtry parameter
steps = 10

n = length(names(sem_train2))
lowend = max(sqrt(n)-steps, 0)
highend = min(sqrt(n)+steps, n)

record = c()
iter = 0
for (i in lowend:highend){
  print(iter)
  iter = iter + 1
  
  model = randomForest(sem_train$critical_temp ~ ., 
                        data=sem_train, 
                        mtry=i,
                        ntree=1000)
  
  pred_test2 = predict(object=model, newdata=sem_test2)
  TestRMSE = sqrt(sum((pred_test2 - Y_test2)^2)/length(pred_test2))
  
  record = c(record, TestRMSE)
}

# Optimize mtry
plot(record)

# Train with Optimized mtry
sem.rf = randomForest(x=X_train,
                      y=Y_train,
                      mtry=2,
                      ntree=5000,
                      importance=TRUE)
# Convergence
plot(sem.rf)

# Stats for time efficiency
ptm = proc.time()
sem.rf = randomForest(x=X_train,
                      y=Y_train,
                      mtry=2,
                      ntree=500,
                      importance=TRUE)
proc.time() - ptm

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
