library("gbm")

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

# Fit Model
sem.gbm = gbm.fit(x=X_train,
              y=Y_train,
              distribution = "gaussian",
              n.trees = 5000,
              shrinkage = 0.01, 
              interaction.depth = 5,
              n.minobsinnode=10,
              bag.fraction=0.5,
              keep.data=TRUE)

# Best number of trees
best.iter = gbm.perf(sem.gbm, 
             plot.it = TRUE)

print(best.iter)

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

