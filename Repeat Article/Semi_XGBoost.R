library("xgboost")

rsq = function (x, y) cor(x, y) ^ 2

# Readin Data into DataFrame
sem = read.csv('C:\\Users\\jim19\\Desktop\\Statistics_Project\\ML Models\\train.csv')

# Average RMSE in 25 iters
TrainRMSE_list = c()
TrainRSQ_list  = c()
TestRMSE_list  = c()
TestRSQ_list   = c()
for (iter in 1:25) {
  # Split into Traing & Test
  smp_size <- floor(0.67 * nrow(sem))
  train_ind <- sample(seq_len(nrow(sem)), size = smp_size)
  
  sem_train = sem[train_ind, ]
  sem_test = sem[-train_ind, ]
  
  X_train = as.matrix(sem_train[-82])
  Y_train = as.matrix(sem_train["critical_temp"])
  
  X_test = as.matrix(sem_test[-82])
  Y_test = as.matrix(sem_test["critical_temp"])
  
  # Apply Naive Bayes
  model_xg = xgboost(data = X_train, 
                     label = Y_train,
                     eta = 0.02,
                     max_depth = 16, 
                     min_child_weight = 1,
                     colsample_bytree = 0.5,
                     subsample = 0.5,
                     nrounds = 750,
                     objective = "reg:linear")
  
  # Training MSE
  pred_train = predict(model_xg, X_train)
  TrainRMSE = sqrt(sum((pred_train - Y_train)^2)/length(pred_train))
  TrainRSQ  = rsq(pred_train, Y_train)
  
  # Test MSE
  pred_test = predict(model_xg, X_test)
  TestRMSE = sqrt(sum((pred_test - Y_test)^2)/length(pred_test))
  TestRSQ  = rsq(pred_test, Y_test)
  
  # Collect Results
  TrainRMSE_list = c(TrainRMSE_list, TrainRMSE)
  TrainRSQ_list = c(TrainRSQ_list, TrainRSQ)
  TestRMSE_list = c(TestRMSE_list, TestRMSE)
  TestRSQ_list = c(TestRSQ_list, TestRSQ)
}

# Runtime
ptm = proc.time()
model_xg = xgboost(data = X_train, 
                   label = Y_train,
                   eta = 0.02,
                   max_depth = 16, 
                   min_child_weight = 1,
                   colsample_bytree = 0.5,
                   subsample = 0.5,
                   nrounds = 750,
                   objective = "reg:linear")
proc.time() - ptm

# Average RMSE and RSQ
AveTrainRMSE = mean(TrainRMSE_list)
AveTrainRSQ  = mean(TrainRSQ_list)
AveTestRMSE  = mean(TestRMSE_list)
AveTestRSQ   = mean(TestRSQ_list)

AveTrainRMSE
AveTrainRSQ
AveTestRMSE
AveTestRSQ

# Plot TrueTc - PredTc for test batch(last iter)
plot(pred_test, Y_test)
lines(seq(1,200),seq(1,200),col="red")

# Importance Table
importance_table = xgb.importance(model = model_xg, 
                                  data = X_train,
                                  label = Y_train)
importance_table
