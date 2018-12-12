library("gbm")

# Readin Data into DataFrame
sem = read.csv('C:\\Users\\jim19\\Desktop\\Statistics_Project\\ML Models\\train.csv')

# Split into Traing & Test
smp_size = floor(0.8 * nrow(sem))
train_ind = sample(seq_len(nrow(sem)), size = smp_size)

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

record = c()
iter = 0
for (shrink in c(0.01, 0.05, 0.1)) {
  for (depth in c(1, 5, 9)) {
    for (mino in c(5, 10, 15)) {
      print(iter)
      iter = iter + 1
      
      sem.gbm = gbm.fit(x=X_train,
                    y=Y_train,
                    distribution = "gaussian",
                    n.trees = 5000,
                    shrinkage = shrink, 
                    interaction.depth = depth,
                    n.minobsinnode=mino,
                    bag.fraction=0.5,
                    keep.data=TRUE)
      
      # Test Error
      pred_test2 = predict(sem.gbm, sem_test2, n.trees = 3000)
      TestRMSE = sqrt(sum((pred_test2 - Y_test2)^2)/length(pred_test2))
      TestRMSE
      
      record = c(record, TestRMSE)
    }
  }
}

# Find Optimized Parameters
print(record)

# Train with the optimized parameters
sem.gbm = gbm.fit(x=X_train,
                  y=Y_train,
                  distribution = "gaussian",
                  n.trees = 10000,
                  shrinkage = 0.1, 
                  interaction.depth = 9,
                  n.minobsinnode= 5,
                  bag.fraction=0.5,
                  keep.data=TRUE)

# Determine Best number of Maximum trees
best.iter = gbm.perf(sem.gbm, 
                     plot.it = TRUE)
print(best.iter)

# Stats for time efficiency
ptm = proc.time()
sem.gbm = gbm.fit(x=X_train,
                  y=Y_train,
                  distribution = "gaussian",
                  n.trees = 3000,
                  shrinkage = 0.1, 
                  interaction.depth = 9,
                  n.minobsinnode= 5,
                  bag.fraction=0.5,
                  keep.data=TRUE)
proc.time() - ptm

# Summary and Plots
summary(sem.gbm)

# Train Error
pred_train = predict(sem.gbm, sem_train, n.trees = best.iter)
TrainMSE = sqrt(sum((pred_train - Y_train)^2)/length(pred_train))
TrainMSE

# Test Error
pred_test = predict(sem.gbm, sem_test, n.trees = best.iter)
TestMSE = sqrt(sum((pred_test - Y_test)^2)/length(pred_test))
TestMSE

